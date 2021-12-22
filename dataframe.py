import math
from itertools import chain

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
pandarallel.initialize()

def to_seconds(x):
    min, secdot = x.split(':')
    sec, ms100 = secdot.split('.')
    return int(min)*60 + int(sec) + int(ms100)*0.1

def to_cos(x, max):
    return np.cos(math.radians(90 - (x / max)*360))

def get_prize(x):
    no = x["result"]
    if 1 <= no <= 5:
        return x[f"prize{no}"]
    else:
        return 0

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

def s_prize(df):
    return df.parallel_apply(get_prize, axis="columns")

def yield_s_corner34(df):
    s_corner = df["corner"].replace('^(\d+)-(\d+)-(\d+)-(\d+)$', r'\3-\4', regex=True)
    s_corner = s_corner.replace('^(\d+)-(\d+)-(\d+)$', r'\2-\3', regex=True)
    df_corner34 = s_corner.str.split('-', expand=True)
    yield "corner3", df_corner34[0].astype(float)
    yield "corner4", df_corner34[1].astype(float)

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
    result["prize"] = s_prize(df)
    str_columns = ["name", "sex", "jockey", "barn", "turn", "weather", "field", "field_condition", "race_condition"]
    for column, series in chain(yield_s_encoded(df, str_columns), yield_s_corner34(df), yield_s_time(df)):
        result[column] = series
    result = result.drop(columns=["corner", "race_name", "start_time", "year",\
                                    "prize1", "prize2", "prize3", "prize4", "prize5"])
    return result

def agg_history(x, f, pattern, df):
    result = []
    name = x["name"]
    now_date = x["race_date"]
    history = df.query(f"name == {name} and race_date < '{now_date}'").iloc[::-1]
    for i in pattern:
        if type(i) is int:
            data = history.head(i)
        elif i == "all":
            data = history
        result.append(f(data, x))
    return pd.Series(result)

def interval(history, now):
    return (now["race_date"] - history["race_date"]).mean() / np.timedelta64(1, 'D')

def same_count(column):
    def wrapper(history, now):
        return (history[column] == now[column]).sum()
    return wrapper

def ave(column):
    def wrapper(history, now):
        return history[column].mean()
    return wrapper

def time(ave_time):
    def wrapper(history, now):
        values = []
        for index, row in history.iterrows():
            average = ave_time[(row["field"], row["distance"], row["field_condition"])]
            values.append((row["time"] - average) / average)
        return np.mean(values)
    return wrapper

def diff(column):
    def wrapper(history, now):
        return (history[column] - now[column]).mean()
    return wrapper

def feature(df):
    ave_time = {key: race["time"].mean() for key, race in df.groupby(["field", "distance", "field_condition"])}
    hist_pattern = [1, 2, 3, 4, 5, 10, "all"]
    feat_pattern = {
        "horse_interval": interval,
        "horse_place": same_count("place_code"),
        "horse_odds": ave("odds"),
        "horse_pop": ave("pop"),
        "horse_result": ave("result"),
        "horse_jockey": same_count("jockey"),
        "horse_penalty": ave("penalty"),
        "horse_distance": diff("distance"),
        "horse_weather": same_count("weather"),
        "horse_fc": same_count("field_condition"),
        "horse_time": time(ave_time),
        "horse_margin": ave("margin"),
        "horse_corner3": ave("corner3"),
        "horse_corner4": ave("corner4"),
        "horse_last3f": ave("last3f"),
        "horse_weight": ave("weight"),
        "horse_wc": ave("wieght_change"),
        "horse_prize": ave("prize"),
    }

    df_copied = df.copy()
    df_copied = df_copied.drop(columns=["result", "hold_num", "day_num", "race_num"])
    features = [df_copied]
    for column_name, func in feat_pattern.items():
        print(f"calculating {column_name} ...", flush=True)
        df_feat = df.parallel_apply(agg_history, args=(func, hist_pattern, df), axis="columns")
        df_feat.columns = [f"{column_name}_{x}" for x in hist_pattern]
        features.append(df_feat)
    return pd.concat(features, axis="columns")
