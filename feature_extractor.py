import sqlite3

import dill as pickle
import numpy as np
import pandas as pd

import config
from encoder import HorseEncoder


def ave(column):
    def wrapper(history, now, index):
        return (history[:, index(column)]).mean()
    return wrapper

def interval(history, now, index):
    return (now[index("race_date")] - history[:, index("race_date")]).astype("timedelta64[D]").mean() / np.timedelta64(1, 'D')

def same_count(column):
    def wrapper(history, now, index):
        return (history[:, index(column)] == now[index(column)]).sum()
    return wrapper

def diff(column):
    def wrapper(history, now, index):
        return (history[:, index(column)] - now[index(column)]).mean()
    return wrapper

def calc_ave_time(df):
    return {key: race["time"].mean() for key, race in df.groupby(["field", "distance", "field_condition"])}

def time(ave):
    def wrapper(history, now, index):
        def average_time(time, f, d, fc, ave):
            a = ave[(f, d, fc)]
            return (time - a) / a
        ht = history[:, index("time")]
        hf = history[:, index("field")]
        hd = history[:, index("distance")]
        hfc = history[:, index("field_condition")]
        times = np.vectorize(average_time)(ht, hf, hd, hfc, ave)
        return np.mean(times)
    return wrapper

def extract_samevalue(a, target_index):
    values = np.unique(a[:, target_index])
    for value in values:
        extracted_nparray = a[np.where(a[:, target_index] == value)]
        yield value, extracted_nparray

def agg_history(funcs, hist_pattern, horse_history, index):
    no_hist = np.empty((len(funcs)*len(hist_pattern),))
    no_hist[:] = np.nan
    result = []
    for i in range(0, len(horse_history)):
        row = horse_history[i, :]
        past_rows = horse_history[:, index("race_date")] < row[index("race_date")]
        past_hist = horse_history[np.where(past_rows)][::-1]
        if past_hist.any():
            try:
                last_jrace_fresult = np.array([f(past_hist[:j, :], row, index) for f in funcs for j in hist_pattern])
                result.append(last_jrace_fresult)
            except Exception as e:
                print(e)
                result.append(no_hist)
        else:
            result.append(no_hist)
    return np.array(result)

def df2np(df):
    a = df.values
    columns = df.columns.tolist()
    index = lambda x: columns.index(x)
    return a, columns, index

def yield_history_aggdf(df, hist_pattern, feat_pattern):
    funcs = list(feat_pattern.values())
    past_columns = [f"{col}_{x}" for col in list(feat_pattern.keys()) for x in hist_pattern]
    a, columns, index = df2np(df)

    for name, hist in extract_samevalue(a, target_index=index("name")):
        a_agghist = agg_history(funcs, hist_pattern, hist, index)
        hist_df = pd.DataFrame(np.concatenate([hist, a_agghist], axis=1), columns=columns+past_columns)
        yield name, hist_df

def search_history(name, df_encoded, hist_pattern, feat_pattern, feature_db):
    funcs = list(feat_pattern.values())
    past_columns = [f"{col}_{x}" for col in list(feat_pattern.keys()) for x in hist_pattern]

    hist = pd.read_sql_query(f"SELECT * FROM horse WHERE name=={name}", feature_db)
    hist["race_date"] = pd.to_datetime(hist["race_date"])
    hist = hist.sort_values("race_date")
    hist = hist.loc[:, :'score']

    row_target = df_encoded[df_encoded["name"] == name]
    if "id" not in row_target.columns:
        row_target["id"] = None
    if "index" not in row_target.columns:
        row_target["index"] = None
    hist = pd.concat([hist, row_target])
    a, columns, index = df2np(hist)
    a_agghist = agg_history(funcs, hist_pattern, a, index)
    hist_df = pd.DataFrame(np.concatenate([hist, a_agghist], axis=1), columns=columns+past_columns)
    return hist_df.tail(1)

def prepare(output_db, input_db="netkeiba.sqlite", encoder_file="encoder.pickle"):
    with sqlite3.connect(input_db) as conn:
        df_original = pd.read_sql_query("SELECT * FROM horse", conn)
        if "index" in df_original.columns:
            df_original = df_original.drop(columns="index")
        df_original["id"] = df_original.index
        
    horse_encoder = HorseEncoder()
    df_format = horse_encoder.format(df_original)
    df_encoded = horse_encoder.fit_transform(df_format)
    with open(encoder_file, "wb") as f:
        pickle.dump(horse_encoder, f)

    ave_time = calc_ave_time(df_encoded)
    hist_pattern = config.hist_pattern
    feat_pattern = config.feature_pattern(ave_time)
    df_avetime = pd.Series(ave_time).rename_axis(["field", "distance", "field_condition"]).reset_index(name="ave_time")
    with sqlite3.connect(output_db) as conn:
        df_avetime.to_sql("ave_time", conn, if_exists="replace", index=False)

    hist_df_list = []
    for name, hist_df in yield_history_aggdf(df_encoded, hist_pattern, feat_pattern):
        print("\r"+str(name),end="")
        hist_df_list.append(hist_df)
    df_feat = pd.concat(hist_df_list)
    df_feat = df_feat.sort_values("id").reset_index()
    df_feat = pd.concat([df.sort_values("horse_no") for _, df in df_feat.groupby(config.RACE_COLUMNS)])
    with sqlite3.connect(output_db) as conn:
        df_feat.to_sql("horse", conn, if_exists="replace", index=False)

if __name__ == "__main__":
    prepare(
        output_db=config.feat_db, 
        input_db=config.netkeiba_db, 
        encoder_file=config.encoder_file, 
    )
