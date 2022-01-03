import numpy as np
import pandas as pd


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