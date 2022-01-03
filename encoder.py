import math

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


class HorseEncoder():

    STR_COLUMNS = ["name", "sex", "jockey", "barn", "turn", "weather", "field", "field_condition", "race_condition", "race_name"]

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
        result["prize"] = calc_prize(df)
        result["score"] = calc_score(df)
        result[HorseEncoder.STR_COLUMNS] = self.encoder.transform(df[HorseEncoder.STR_COLUMNS])
        return result

    def fit_transform(self, df):
        return self.fit(df).transform(df)

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
    if 1 <= no <= 5:
        return [20, 8, 5, 3, 2][no-1]
    else:
        return 0

def get_prize(x):
    no = x["result"]
    if not no:
        return None
    if 1 <= no <= 5:
        return x[f"prize{no}"]
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
        return df_corner34[0].astype(float)
    else:
        return None

def format_corner4(s_corner):
    df_corner34 = format_corner(s_corner)
    if len(df_corner34.columns) == 2:
        return df_corner34[1].astype(float)
    else:
        return None

def format_racecondition(df):
    mask_bad_rc = df.eval("race_condition in ['未勝利', '新馬', '５００万下', '１０００万下', '15頭', '16頭', '12頭', '10頭', '14頭', '13頭']")
    result = df["race_condition"].mask(mask_bad_rc, df["race_name"])
    return result

def format(df):
    result = df.copy()
    result["jockey"] = format_jockey(df["jockey"])
    result["race_date"] = format_date(df["race_date"], df["year"])
    result["corner3"] = format_corner3(df["corner"])
    result["corner4"] = format_corner4(df["corner"])
    result["race_condition"] = format_racecondition(df)
    return result

def encode_racedate(s_racedate):
    return s_racedate.dt.dayofyear.apply(to_cos, max=365)

def encode_starttime(s_starttime):
    result = pd.to_timedelta(s_starttime + ":00").dt.total_seconds()
    result = result.apply(to_cos, max=86400)
    return result

def calc_prize(df):
    return df.apply(get_prize, axis="columns")

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

def calc_score(df):
    return df.apply(to_score, axis="columns")

if __name__ == "__main__":
    import netkeiba
    horses = [horse for horse in netkeiba.scrape_shutuba("202206010111")]
    df_original = pd.DataFrame(horses, columns=netkeiba.COLUMNS)
    df_format = format(df_original)
    netkeiba_encoder = HorseEncoder()
    netkeiba_encoder.fit(df_format)
    df_encoded = netkeiba_encoder.transform(df_format)
    print(df_encoded.T)

    horses = [horse for horse in netkeiba.scrape_results("202106050811")]
    df_original = pd.DataFrame(horses, columns=netkeiba.COLUMNS)
    df_format = format(df_original)
    df_encoded = netkeiba_encoder.transform(df_format)
    print(df_encoded.T)
