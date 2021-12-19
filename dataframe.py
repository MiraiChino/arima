import math
from itertools import chain

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

def to_seconds(x):
    min, secdot = x.split(':')
    sec, ms100 = secdot.split('.')
    return int(min)*60 + int(sec) + int(ms100)*0.1

def to_cos(x, max):
    return np.cos(math.radians(90 - (x / max)*360))

def s_cosracedate(s_racedate):
    return s_racedate.dt.dayofyear.apply(to_cos, max=365)

def s_jockey(df):
    return df["jockey"].str.replace('^[△▲☆★◇](.*)', r'\1', regex=True)

def s_racedate(df):
    s_mmdd = df["race_date"].replace('(\d+)月(\d+)日\(.\)', r'\1/\2', regex=True)
    s_yyyy = df["year"].astype(str)
    result = pd.to_datetime(s_yyyy + s_mmdd, format='%Y%m/%d')
    return result

def s_cosstarttime(df):
    s_starttime = pd.to_timedelta(df["start_time"] + ":00").dt.total_seconds()
    result = s_starttime.apply(to_cos, max=86400)
    return result

def s_revised_race_condition(df):
    mask_bad_rc = df.eval("race_condition in ['未勝利', '新馬', '５００万下', '１０００万下', '15頭', '16頭', '12頭', '10頭', '14頭', '13頭']")
    result = df["race_condition"].mask(mask_bad_rc, df["race_name"])
    return result

def yield_s_corner34(df):
    s_corner = df["corner"].replace('^(\d+)-(\d+)-(\d+)-(\d+)$', r'\3-\4', regex=True)
    s_corner = s_corner.replace('^(\d+)-(\d+)-(\d+)$', r'\2-\3', regex=True)
    df_corner34 = s_corner.str.split('-', expand=True)
    yield "corner3", df_corner34[0]
    yield "corner4", df_corner34[1]

def yield_s_encoded(df, categories):
    df["race_condition"] = s_revised_race_condition(df)
    for category in categories:
        label_encoder.fit(df[category])
        yield category, label_encoder.transform(df[category])

def yield_df_race(df):
    return df.groupby(["year", "place_code", "hold_num", "day_num", "race_num"])

def yield_s_time(df):
    s_margin_list = []
    for raceid, df_race in yield_df_race(df):
        s_timesec = df_race["time"].apply(to_seconds)
        time_1st = s_timesec.iloc[0]
        s_margin_list.append(s_timesec - time_1st)
    yield "margin", pd.concat(s_margin_list)
    yield "time", df["time"].apply(to_seconds)

def format(df):
    result = df.copy()
    result["race_date"] = s_racedate(df)
    result["cos_racedate"] = s_cosracedate(result["race_date"])
    result["cos_starttime"] = s_cosstarttime(df)
    result["jockey"] = s_jockey(df)
    str_columns = ["name", "sex", "jockey", "barn", "turn", "weather", "field_condition", "race_condition"]
    for column, series in chain(yield_s_encoded(df, str_columns), yield_s_corner34(df), yield_s_time(df)):
        result[column] = series
    result = result.drop(columns=["corner", "race_name", "start_time", "field", "year"])
    return result
