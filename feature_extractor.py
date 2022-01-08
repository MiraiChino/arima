import argparse
import sqlite3

import dill as pickle
import numpy as np
import pandas as pd

import feature_params
from encoder import HorseEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indb", dest="input_db", required=True, type=str,
                        help="Example netkeiba.sqlite")
    parser.add_argument("--outdb", dest="output_db", required=True, type=str,
                        help="Example feature.sqlite")
    parser.add_argument("-outencoder", dest="encoder_file", required=True, type=str,
                        help="Example encoder.pickle")
    parser.add_argument("-outparams", dest="params_file", required=True, type=str,
                        help="Example params.pickle")
    return parser.parse_args()

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

def average_time(time, f, d, fc, ave):
    a = ave[(f, d, fc)]
    return (time - a) / a

def time(ave):
    def wrapper(history, now, index):
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
            except:
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
    hist = hist.loc[:, :'corner4']

    row_target = df_encoded[df_encoded["name"] == name]
    hist = pd.concat([hist, row_target])
    a, columns, index = df2np(hist)
    a_agghist = agg_history(funcs, hist_pattern, a, index)
    hist_df = pd.DataFrame(np.concatenate([hist, a_agghist], axis=1), columns=columns+past_columns)
    hist_df = hist_df.set_index("id")
    return hist_df

def prepare(output_db, input_db="netkeiba.sqlite", encoder_file="encoder.pickle", params_file="params.pickle"):
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
    hist_pattern = [1, 2, 3, 4, 5, 10, 999999]
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
        "horse_wc": ave("weight_change"),
        "horse_prize": ave("prize"),
    }
    params = feature_params.Params(ave_time, hist_pattern, feat_pattern)
    with open(params_file, "wb") as f:
        pickle.dump(params, f)

    hist_df_list = []
    for name, hist_df in yield_history_aggdf(df_encoded, hist_pattern, feat_pattern):
        print("\r"+str(name),end="")
        hist_df_list.append(hist_df)
    df_feat = pd.concat(hist_df_list).sort_values("id").reset_index()
    df_feat = pd.concat([df.sort_values("horse_no") for _, df in df_feat.groupby(feature_params.RACE_CULMNS)])
    with sqlite3.connect(output_db) as conn:
        df_feat.to_sql("horse", conn, if_exists="replace", index=False)

if __name__ == "__main__":
    args = parse_args()
    prepare(
        output_db=args.output_db, 
        input_db=args.input_db, 
        encoder_file=args.encoder_file, 
        params_file=args.params_file,
    )
