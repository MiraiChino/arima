import math

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


class HorseEncoder():

    STR_COLUMNS = [
        "name", "jockey", "trainer",
        "sex", "turn", "weather", "field", "field_condition", "race_condition", "race_name"
    ]

    def __init__(self):
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    def fit(self, df):
        self.encoder.fit(df[HorseEncoder.STR_COLUMNS])
        return self

    def transform(self, df):
        result = df.copy()
        result["cos_racedate"] = encode_racedate(result["race_date"])
        result["cos_starttime"] = encode_starttime(df["start_time"])
        result["margin"] = encoded_margin(df)
        result["time"] = encoded_time(df["time"])
        result["tanshou"] = calc_tanshou(df)
        result["hukushou"] = calc_hukushou(df)
        result["prize"] = calc_prize(df)
        result["score"] = calc_score(df)
        result["last3frel"] = relative_min(df["last3f"])
        result["penaltyrel"] = relative_min(df["penalty"])
        result["weightrel"] = relative_min(df["weight"])
        result["penaltywgt"] = df["penalty"] / df["weight"]
        result["oddsrslt"] = df["odds"] / df["result"]
        result[HorseEncoder.STR_COLUMNS] = self.encoder.transform(df[HorseEncoder.STR_COLUMNS])
        return result

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    def format(self, df):
        result = df.copy()
        result["race_id"] = df["race_id"].astype(int)
        result["horse_id"] = df["horse_id"].astype(float)
        result["jockey_id"] = df["jockey_id"].astype(float)
        result["trainer_id"] = df["trainer_id"].astype(float)
        result["race_date"] = format_date(df["race_date"], df["year"])
        result["corner3"] = format_corner3(df["corner"])
        result["corner4"] = format_corner4(df["corner"])
        result["race_condition"] = format_racecondition(df)
        return result

def to_seconds(x):
    if not x:
        return None
    min, secdot = x.split(':')
    sec, ms100 = secdot.split('.')
    return int(min)*60 + int(sec) + int(ms100)*0.1

def to_cos(x, max):
    return np.cos(math.radians(90 - (x / max)*360))

def to_score(x):
    no = x["result"]
    if not no:
        return None
    elif 1 <= no <= 5:
        return [20, 8, 5, 3, 2][int(no)-1]
    else:
        return 0

def get_prize(x):
    no = x["result"]
    if not no:
        return None
    elif 1 <= no <= 5:
        return x[f"prize{int(no)}"]
    else:
        return 0

def get_tanshou(x):
    no = x["result"]
    if not no:
        return None
    elif no == 1:
        if x["horse_no"] == x["tanno1"]:
            return 100/x[f"tan1"]
        elif x["horse_no"] == x["tanno2"]:
            return 100/x[f"tan2"]
    else:
        return 0

def get_hukushou(x):
    no = x["result"]
    if not no:
        return None
    elif 1 <= no <= 3:
        if x["horse_no"] == x["hukuno1"]:
            return 100/x[f"huku1"]
        elif x["horse_no"] == x["hukuno2"]:
            return 100/x[f"huku2"]
        elif x["horse_no"] == x["hukuno3"]:
            return 100/x[f"huku3"]
    else:
        return 0

def format_jockey(s_jockey):
    return s_jockey.str.replace('^[△▲☆★◇](.*)', r'\1', regex=True)

def format_date(s_racedate, s_year):
    s_mmdd = s_racedate.replace('(\d+)月(\d+)日\(.\)', r'\1/\2', regex=True)
    s_yyyy = s_year.astype(str)
    result = pd.to_datetime(s_yyyy + s_mmdd, format='%Y%m/%d')
    return result

def format_corner(s_corner):
    s_corner = s_corner.replace('^(\d+)-(\d+)-(\d+)-(\d+)$', r'\3-\4', regex=True)
    s_corner = s_corner.replace('^(\d+)-(\d+)-(\d+)$', r'\2-\3', regex=True)
    df_corner34 = s_corner.str.split('-', expand=True)
    return df_corner34

def format_corner3(s_corner):
    df_corner34 = format_corner(s_corner)
    if len(df_corner34.columns) == 2:
        corner3 = df_corner34[0]
        corner3 = corner3.mask(corner3 == '', np.nan)
        corner3 = corner3.mask(corner3 == None, np.nan)
        return corner3.astype(float)
    else:
        return None

def format_corner4(s_corner):
    df_corner34 = format_corner(s_corner)
    if len(df_corner34.columns) == 2:
        corner4 = df_corner34[1]
        corner4 = corner4.mask(corner4 == '', np.nan)
        corner4 = corner4.mask(corner4 == None, np.nan)
        return corner4.astype(float)
    else:
        return None

def format_racecondition(df):
    mask_bad_rc = df.eval("race_condition in ['15頭', '16頭', '12頭', '10頭', '14頭', '13頭']")
    # mask_bad_rc = df.eval("race_condition in ['未勝利', '新馬', '５００万下', '１０００万下', '15頭', '16頭', '12頭', '10頭', '14頭', '13頭']")
    result = df["race_condition"].mask(mask_bad_rc, df["race_name"])
    return result

def encode_racedate(s_racedate):
    return s_racedate.dt.dayofyear.apply(to_cos, max=365)

def encode_starttime(s_starttime):
    result = pd.to_timedelta(s_starttime + ":00").dt.total_seconds()
    result = result.apply(to_cos, max=86400)
    return result

def calc_prize(df):
    return df.apply(get_prize, axis="columns")

def calc_score(df):
    return df.apply(to_score, axis="columns")

def calc_hukushou(df):
    return df.apply(get_hukushou, axis="columns")

def calc_tanshou(df):
    return df.apply(get_tanshou, axis="columns")

def yield_df_race(df):
    return df.groupby(["year", "place_code", "hold_num", "day_num", "race_num"])

def encoded_margin(df):
    s_margin_list = []
    for raceid, df_race in yield_df_race(df):
        s_timesec = df_race["time"].apply(to_seconds)
        time_1st = s_timesec.iloc[0]
        s_margin_list.append(s_timesec - time_1st)
    return pd.concat(s_margin_list)

def encoded_time(s_time):
    return s_time.apply(to_seconds)

def relative_min(series):
    return series - series.min()

if __name__ == "__main__":
    import netkeiba
    horse_encoder = HorseEncoder()
    race_data, horses = netkeiba.scrape_shutuba("200806010101")
    df_horses = pd.DataFrame(horses, columns=netkeiba.HORSE_COLUMNS)
    df_races = pd.DataFrame([race_data], columns=netkeiba.RACE_PRE_COLUMNS)
    df_original = pd.merge(df_horses, df_races, on='race_id', how='left')
    df_format = horse_encoder.format(df_original)
    df_encoded = horse_encoder.fit_transform(df_format)
    print(df_encoded)

    race_data, horses = netkeiba.scrape_results("200806010108") # ３連単なし
    df_horses = pd.DataFrame(horses, columns=netkeiba.HORSE_COLUMNS)
    df_races = pd.DataFrame([race_data], columns=netkeiba.RACE_AFTER_COLUMNS)
    df_original = pd.merge(df_horses, df_races, on='race_id', how='left')
    df_format = horse_encoder.format(df_original)
    df_encoded = horse_encoder.transform(df_format)
    print(df_encoded)

    race_data, horses = netkeiba.scrape_results("200808010709") # 単勝２頭（同着）
    df_horses = pd.DataFrame(horses, columns=netkeiba.HORSE_COLUMNS)
    df_races = pd.DataFrame([race_data], columns=netkeiba.RACE_AFTER_COLUMNS)
    df_original = pd.merge(df_horses, df_races, on='race_id', how='left')
    df_format = horse_encoder.format(df_original)
    df_encoded = horse_encoder.transform(df_format)
    print(df_encoded)

    race_data, horses = netkeiba.scrape_results("201302010505") # 枠連なし、複勝なぜか２頭
    df_horses = pd.DataFrame(horses, columns=netkeiba.HORSE_COLUMNS)
    df_races = pd.DataFrame([race_data], columns=netkeiba.RACE_AFTER_COLUMNS)
    df_original = pd.merge(df_horses, df_races, on='race_id', how='left')
    df_format = horse_encoder.format(df_original)
    df_encoded = horse_encoder.transform(df_format)
    print(df_encoded)

    #     result  gate  horse_no  name  horse_id  sex  age  ...  corner4  cos_racedate  cos_starttime  tanshou  hukushou  prize  score
    # 0      1.0     6        11   0.0      12.0  2.0    3  ...     12.0      0.085965       -0.55557     2340       570   1000     20
    # 1      2.0     2         4   7.0       8.0  2.0    3  ...      1.0      0.085965       -0.55557        0       200    400      8
    # 2      3.0     1         2   9.0      14.0  2.0    3  ...     10.0      0.085965       -0.55557        0      1430    250      5
    # 3      4.0     3         5   4.0      13.0  2.0    3  ...      3.0      0.085965       -0.55557        0         0    150      3
    # 4      5.0     2         3  11.0       7.0  2.0    3  ...      3.0      0.085965       -0.55557        0         0    100      2
    # 5      6.0     7        14   3.0       5.0  2.0    3  ...     10.0      0.085965       -0.55557        0         0      0      0
    # 6      7.0     8        15  15.0      10.0  0.0    3  ...     12.0      0.085965       -0.55557        0         0      0      0
    # 7      8.0     6        12   8.0       4.0  2.0    3  ...      6.0      0.085965       -0.55557        0         0      0      0
    # 8      9.0     8        16   1.0       6.0  2.0    3  ...      3.0      0.085965       -0.55557        0         0      0      0
    # 9     10.0     4         8  14.0      11.0  2.0    3  ...      6.0      0.085965       -0.55557        0         0      0      0
    # 10    11.0     5         9  10.0      15.0  2.0    3  ...      2.0      0.085965       -0.55557        0         0      0      0
    # 11    12.0     3         6  13.0       1.0  2.0    3  ...      6.0      0.085965       -0.55557        0         0      0      0
    # 12    13.0     4         7  12.0       9.0  2.0    3  ...      6.0      0.085965       -0.55557        0         0      0      0
    # 13    14.0     7        13   5.0       2.0  2.0    3  ...     14.0      0.085965       -0.55557        0         0      0      0
    # 14    15.0     5        10   2.0       3.0  1.0    3  ...     15.0      0.085965       -0.55557        0         0      0      0
    # 15     NaN     1         1   6.0       0.0  1.0    3  ...      NaN      0.085965       -0.55557        0         0      0      0