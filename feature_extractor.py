import sqlite3

import dill as pickle
import numpy as np
import pandas as pd

import config
import utils
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

def same_ave(*same_columns, target="prize"):
    def wrapper(history, now, index):
        i1 = index(target)
        same_conditions = [(history[:, index(c)] == now[index(c)]) for c in same_columns]
        same_hist = history[np.logical_and.reduce(same_conditions)]
        if same_hist.any():
            prizes = same_hist[:, i1]
            return prizes.mean()
        else:
            return 0
    return wrapper

def drize(distance, prize, now_distance):
    return (1 - abs(distance - now_distance) / now_distance) * prize

def distance_prize():
    def wrapper(history, now, index):
        i1 = index("distance")
        distances = history[:, i1]
        prizes = history[:, index("prize")]
        dprizes = np.vectorize(drize)(distances, prizes, now[i1])
        return np.mean(dprizes)
    return wrapper

def same_drize(*same_columns):
    def wrapper(history, now, index):
        i1 = index("distance")
        same_conditions = [(history[:, index(c)] == now[index(c)]) for c in same_columns]
        same_hist = history[np.logical_and.reduce(same_conditions)]
        if same_hist.any():
            distances = same_hist[:, i1]
            prizes = same_hist[:, index("prize")]
            dprizes = np.vectorize(drize)(distances, prizes, now[i1])
            return np.mean(dprizes)
        else:
            return 0
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

def agg_history_i(i, funcs, hist_pattern, history, index):
    no_hist = np.empty((len(funcs)*len(hist_pattern),))
    no_hist[:] = np.nan
    row = history[i, :]
    past_rows = history[:, index("race_date")] < row[index("race_date")]
    past_hist = history[np.where(past_rows)][::-1]
    if past_hist.any():
        try:
            last_jrace_fresult = np.array([f(past_hist[:1+j, :], row, index) for f in funcs for j in hist_pattern])
            return last_jrace_fresult
        except Exception as e:
            print(e)
            return no_hist
    else:
        return no_hist

def agg_history(funcs, hist_pattern, history, index):
    result = []
    for i in range(0, len(history)):
        history_i = agg_history_i(i, funcs, hist_pattern, history, index)
        result.append(history_i)
    return np.array(result)

def df2np(df):
    a = df.values
    columns = df.columns.tolist()
    index = lambda x: columns.index(x)
    return a, columns, index

def yield_history_aggdf(df, target, hist_pattern, feat_pattern):
    funcs = list(feat_pattern.values())
    past_columns = [f"{col}_{x}" for col in list(feat_pattern.keys()) for x in hist_pattern]
    a, columns, index = df2np(df)

    for name, hist in extract_samevalue(a, target_index=index(target)):
        a_agghist = agg_history(funcs, hist_pattern, hist, index)
        hist_df = pd.DataFrame(np.concatenate([hist, a_agghist], axis=1), columns=columns+past_columns)
        yield name, hist_df

def search_history(target_row, hist_pattern, feat_pattern, df):
    row_df = pd.DataFrame([target_row]).reset_index()
    if "id" not in row_df.columns:
        row_df["id"] = None
    if "index" not in row_df.columns:
        row_df["index"] = None
    hist_df = row_df.copy(deep=True)
    condition = " or ".join(f"{column}=={target_row[column]}" for column in feat_pattern.keys())
    condition_df = df.query(condition)

    for column in feat_pattern.keys():
        funcs = list(feat_pattern[column].values())
        past_columns = [f"{col}_{x}" for col in list(feat_pattern[column].keys()) for x in hist_pattern]
        hist = condition_df.query(f"{column}=={target_row[column]}")
        hist = pd.concat([hist, row_df])
        a, columns, index = df2np(hist)
        targetrow_agg = agg_history_i(len(a)-1, funcs, hist_pattern, a, index)
        row_df_column = pd.DataFrame([targetrow_agg], columns=past_columns)
        hist_df = pd.concat([hist_df, row_df_column], axis="columns")
    return hist_df

def prepare(output_db, input_db="netkeiba.sqlite", encoder_file="encoder.pickle"):
    with sqlite3.connect(input_db) as conn:
        reader = pd.read_sql_query("SELECT * FROM horse", conn, chunksize=10000)
        chunks = []
        for i, df in enumerate(reader):
            print(i+1, df.shape)
            df_chunk = utils.reduce_mem_usage(df)
            chunks.append(df_chunk)
        df_original = pd.concat(chunks, ignore_index=True)
        if "index" in df_original.columns:
            df_original = df_original.drop(columns="index")
        df_original["id"] = df_original.index
        
    horse_encoder = HorseEncoder()
    df_format = horse_encoder.format(df_original)
    df_encoded = horse_encoder.fit_transform(df_format)
    df_encoded = utils.reduce_mem_usage(df_encoded)
    with open(encoder_file, "wb") as f:
        pickle.dump(horse_encoder, f)

    ave_time = calc_ave_time(df_encoded)
    hist_pattern = config.hist_pattern
    feat_pattern = config.feature_pattern(ave_time)
    df_avetime = pd.Series(ave_time).rename_axis(["field", "distance", "field_condition"]).reset_index(name="ave_time")
    with sqlite3.connect(output_db) as conn:
        df_avetime.to_sql("ave_time", conn, if_exists="replace", index=False)

    cols = list(feat_pattern.keys())
    df_feats = {}
    for column in cols:
        hist_list = []
        for name, hist_df in yield_history_aggdf(df_encoded, column, hist_pattern, feat_pattern[column]):
            print(f"\r{column}:{name}", end="")
            hist_list.append(hist_df)
        df_feats[column] = pd.concat(hist_list)
        df_feats[column] = df_feats[column].sort_values("id").reset_index()
        df_feats[column] = pd.concat([df.sort_values("horse_no") for _, df in df_feats[column].groupby(config.RACE_COLUMNS)])
        df_feats[column] = utils.reduce_mem_usage(df_feats[column])
    df_feat = df_feats[cols[0]]
    for column in cols[1:]:
        cols_to_use = df_feats[column].columns.difference(df_feat.columns).tolist() + ["id"]
        df_feat = pd.merge(df_feat, df_feats[column][cols_to_use], on="id")
    with sqlite3.connect(output_db) as conn:
        df_feat.to_sql("horse", conn, if_exists="replace", index=False)

if __name__ == "__main__":
    prepare(
        output_db=config.feat_db,
        input_db=config.netkeiba_db,
        encoder_file=config.encoder_file,
    )
