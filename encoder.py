import numpy as np
from datetime import datetime

import polars as pl
from sklearn.preprocessing import OrdinalEncoder


class HorseEncoder():

    STR_COLUMNS = [
        "name", "jockey", "trainer",
        "sex", "turn", "weather", "field", "field_condition", "race_condition", "race_name"
    ]
    start_of_day = datetime.strptime("0001-01-01", "%Y-%m-%d")

    def __init__(self):
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    def format(self, df):
        return df.filter(
            ~pl.col("race_condition").is_in(['15頭', '16頭', '12頭', '10頭', '14頭', '13頭'])
        )

    def fit(self, df):
        self.encoder.fit(df.select(HorseEncoder.STR_COLUMNS).to_numpy())
        return self

    def transform(self, df):
        np_encoded_T = self.encoder.transform(df.select(HorseEncoder.STR_COLUMNS).to_numpy()).T
        return (
            df.lazy()
            .with_columns([
                pl.col("race_id").cast(pl.Float64),
                pl.col("horse_id").cast(pl.Float64),
                pl.col("jockey_id").cast(pl.Float64),
                pl.col("trainer_id").cast(pl.Float64),
                encode_racedate().alias("race_date"),
                encode_starttime().alias("start_time"),
                encode_seconds().alias("time"),
                pl.col("corner").str.extract_all(r'\d+').alias("corners"),
                encode_tanshou().alias("tanshou"),
                encode_hukushou().alias("hukushou"),
                encode_prize().alias("prize"),
                encode_score().alias("score"),
                (pl.col("last3f") - pl.col("last3f").min().over("race_id")).alias("last3frel"),
                (pl.col("penalty") - pl.col("penalty").min().over("race_id")).alias("penaltyrel"),
                (pl.col("weight") - pl.col("weight").min().over("race_id")).alias("weightrel"),
                (pl.col("penalty") / pl.col("weight")).alias("penaltywgt"),
                (pl.col("odds") / pl.col("result")).alias("oddsrslt"),
                *[pl.Series(col, values) for col, values in zip(HorseEncoder.STR_COLUMNS, np_encoded_T)],
            ])
            .with_columns([
                encode_cos(pl.col("race_date").dt.ordinal_day(), max=365).alias("cos_racedate"),
                encode_cos((pl.col("start_time") - HorseEncoder.start_of_day).dt.total_seconds(), max=86400).alias("cos_starttime"),
                (pl.col("time") - pl.col("time").min().over("race_id")).alias("margin"),
                pl.col("corners").list.reverse().list.get(0, null_on_oob=True).cast(pl.Float64).alias("corner4"),
                pl.col("corners").list.reverse().list.get(1, null_on_oob=True).cast(pl.Float64).alias("corner3"),
                pl.col("corners").list.reverse().list.get(2, null_on_oob=True).cast(pl.Float64).alias("corner2"),
                pl.col("corners").list.reverse().list.get(3, null_on_oob=True).cast(pl.Float64).alias("corner1"),
            ])
            .select(pl.exclude("corners"))
            .collect()
        )

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    def format_fit_transform(self, df):
        df_formatted = self.format(df)
        return self.fit_transform(df_formatted)

def encode_cos(num_col, max):
    angle = 90 - (num_col / max) * 360
    radian = angle * np.pi / 180
    return radian.cos()

def encode_racedate():
    return (
        pl.format(
            "{}-{}-{}",
            pl.col("year"),
            pl.col("race_date").str.extract(r'(\d+)月'),
            pl.col("race_date").str.extract(r'(\d+)日')
        )
        .str.to_date()
    )

def encode_starttime():
    return (
        pl.col("start_time")
        .str.extract_all(r'\d+')
        .list.eval(
            pl.element().str.zfill(2)
        )
        .list.join(":")
        .str.strptime(pl.Datetime, "%H:%M")
    )

def extract_int(col_name, i):
    return pl.col(col_name).str.extract_all(r'\d+').list.get(i, null_on_oob=True).str.to_integer(strict=False)

def encode_seconds():
    return (
        extract_int("time", 0) * 60 +
        extract_int("time", 1) +
        extract_int("time", 2) * 0.1
    )

def encode_tanshou():
    return (
        pl.when(pl.col("result") != 1).then(0.0)
        .when(pl.col("horse_no") == pl.col("tanno1")).then(100.0/pl.col("tan1"))
        .when(pl.col("horse_no") == pl.col("tanno2")).then(100.0/pl.col("tan2"))
    )

def encode_hukushou():
    return (
        pl.when(3 < pl.col("result")).then(0.0)
        .when(pl.col("horse_no") == pl.col("hukuno1")).then(100.0/pl.col("hukuno1"))
        .when(pl.col("horse_no") == pl.col("hukuno2")).then(100.0/pl.col("hukuno2"))
        .when(pl.col("horse_no") == pl.col("hukuno3")).then(100.0/pl.col("hukuno3"))
    )

def encode_prize():
    return (
        pl.when(5 < pl.col("result")).then(0.0)
        .when(pl.col("result") == 1).then(pl.col("prize1"))
        .when(pl.col("result") == 2).then(pl.col("prize2"))
        .when(pl.col("result") == 3).then(pl.col("prize3"))
        .when(pl.col("result") == 4).then(pl.col("prize4"))
        .when(pl.col("result") == 5).then(pl.col("prize5"))
    )

def encode_score():
    return (
        pl.when(5 < pl.col("result")).then(0)
        .when(pl.col("result") == 1).then(20)
        .when(pl.col("result") == 2).then(8)
        .when(pl.col("result") == 3).then(5)
        .when(pl.col("result") == 4).then(3)
        .when(pl.col("result") == 5).then(2)
    )

# if __name__ == "__main__":
#     import netkeiba
#     horse_encoder = HorseEncoder()
#     race_data, horses = netkeiba.scrape_shutuba("200806010101")
#     df_horses = pd.DataFrame(horses, columns=netkeiba.HORSE_COLUMNS)
#     df_races = pd.DataFrame([race_data], columns=netkeiba.RACE_PRE_COLUMNS)
#     df_original = pd.merge(df_horses, df_races, on='race_id', how='left')
#     df_format = horse_encoder.format(df_original)
#     df_encoded = horse_encoder.fit_transform(df_format)
#     print(df_encoded)

#     race_data, horses = netkeiba.scrape_results("200806010108") # ３連単なし
#     df_horses = pd.DataFrame(horses, columns=netkeiba.HORSE_COLUMNS)
#     df_races = pd.DataFrame([race_data], columns=netkeiba.RACE_AFTER_COLUMNS)
#     df_original = pd.merge(df_horses, df_races, on='race_id', how='left')
#     df_format = horse_encoder.format(df_original)
#     df_encoded = horse_encoder.transform(df_format)
#     print(df_encoded)

#     race_data, horses = netkeiba.scrape_results("200808010709") # 単勝２頭（同着）
#     df_horses = pd.DataFrame(horses, columns=netkeiba.HORSE_COLUMNS)
#     df_races = pd.DataFrame([race_data], columns=netkeiba.RACE_AFTER_COLUMNS)
#     df_original = pd.merge(df_horses, df_races, on='race_id', how='left')
#     df_format = horse_encoder.format(df_original)
#     df_encoded = horse_encoder.transform(df_format)
#     print(df_encoded)

#     race_data, horses = netkeiba.scrape_results("201302010505") # 枠連なし、複勝なぜか２頭
#     df_horses = pd.DataFrame(horses, columns=netkeiba.HORSE_COLUMNS)
#     df_races = pd.DataFrame([race_data], columns=netkeiba.RACE_AFTER_COLUMNS)
#     df_original = pd.merge(df_horses, df_races, on='race_id', how='left')
#     df_format = horse_encoder.format(df_original)
#     df_encoded = horse_encoder.transform(df_format)
#     print(df_encoded)

# (Pdb) df
# shape: (738149, 59)
# ┌────────┬──────┬──────────┬────────────────────┬─────┬──────┬───────┬─────────┬──────┐
# │ result ┆ gate ┆ horse_no ┆ name               ┆ ... ┆ uma2 ┆ puku  ┆ san1    ┆ san2 │
# │ ---    ┆ ---  ┆ ---      ┆ ---                ┆     ┆ ---  ┆ ---   ┆ ---     ┆ ---  │
# │ f32    ┆ i8   ┆ i8       ┆ str                ┆     ┆ f64  ┆ i32   ┆ f32     ┆ f64  │
# ╞════════╪══════╪══════════╪════════════════════╪═════╪══════╪═══════╪═════════╪══════╡
# │ 1.0    ┆ 1    ┆ 2        ┆ メジロアリエル     ┆ ... ┆ null ┆ 10600 ┆ null    ┆ null │
# │ 2.0    ┆ 3    ┆ 5        ┆ ヒロアンジェロ     ┆ ... ┆ null ┆ 10600 ┆ null    ┆ null │
# │ 3.0    ┆ 2    ┆ 3        ┆ キャスタスペルミー ┆ ... ┆ null ┆ 10600 ┆ null    ┆ null │
# │ 4.0    ┆ 4    ┆ 7        ┆ デルマベガ         ┆ ... ┆ null ┆ 10600 ┆ null    ┆ null │
# │ ...    ┆ ...  ┆ ...      ┆ ...                ┆ ... ┆ ...  ┆ ...   ┆ ...     ┆ ...  │
# │ 13.0   ┆ 3    ┆ 6        ┆ テイエムイダテン   ┆ ... ┆ null ┆ 2580  ┆ 10260.0 ┆ null │
# │ 14.0   ┆ 4    ┆ 8        ┆ ショウナンアリアナ ┆ ... ┆ null ┆ 2580  ┆ 10260.0 ┆ null │
# │ 15.0   ┆ 5    ┆ 9        ┆ アールラプチャー   ┆ ... ┆ null ┆ 2580  ┆ 10260.0 ┆ null │
# │ 16.0   ┆ 7    ┆ 13       ┆ グレイトゲイナー   ┆ ... ┆ null ┆ 2580  ┆ 10260.0 ┆ null │
# └────────┴──────┴──────────┴────────────────────┴─────┴──────┴───────┴─────────┴──────┘
# (Pdb) df_encoded
# shape: (737916, 72)
# ┌────────┬──────┬──────────┬─────────┬─────┬─────────┬─────────┬─────────┬─────────┐
# │ result ┆ gate ┆ horse_no ┆ name    ┆ ... ┆ corner1 ┆ corner2 ┆ corner3 ┆ corner4 │
# │ ---    ┆ ---  ┆ ---      ┆ ---     ┆     ┆ ---     ┆ ---     ┆ ---     ┆ ---     │
# │ f32    ┆ i8   ┆ i8       ┆ f64     ┆     ┆ str     ┆ str     ┆ str     ┆ str     │
# ╞════════╪══════╪══════════╪═════════╪═════╪═════════╪═════════╪═════════╪═════════╡
# │ 1.0    ┆ 1    ┆ 2        ┆ 65515.0 ┆ ... ┆ null    ┆ null    ┆ 1       ┆ 1       │
# │ 2.0    ┆ 3    ┆ 5        ┆ 50525.0 ┆ ... ┆ null    ┆ null    ┆ 10      ┆ 7       │
# │ 3.0    ┆ 2    ┆ 3        ┆ 14124.0 ┆ ... ┆ null    ┆ null    ┆ 13      ┆ 8       │
# │ 4.0    ┆ 4    ┆ 7        ┆ 40762.0 ┆ ... ┆ null    ┆ null    ┆ 2       ┆ 2       │
# │ ...    ┆ ...  ┆ ...      ┆ ...     ┆ ... ┆ ...     ┆ ...     ┆ ...     ┆ ...     │
# │ 13.0   ┆ 3    ┆ 6        ┆ 38454.0 ┆ ... ┆ null    ┆ null    ┆ 6       ┆ 7       │
# │ 14.0   ┆ 4    ┆ 8        ┆ 26765.0 ┆ ... ┆ null    ┆ null    ┆ 9       ┆ 9       │
# │ 15.0   ┆ 5    ┆ 9        ┆ 4554.0  ┆ ... ┆ null    ┆ null    ┆ 4       ┆ 4       │
# │ 16.0   ┆ 7    ┆ 13       ┆ 18195.0 ┆ ... ┆ null    ┆ null    ┆ 9       ┆ 11      │
# └────────┴──────┴──────────┴─────────┴─────┴─────────┴─────────┴─────────┴─────────┘