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

def yield_horse_history(a, name_index):
    horse_names = np.unique(a[:, name_index])
    for name in horse_names:
        hist = a[np.where(a[:, name_index] == name)]
        yield name, hist

def X_result(funcs, X, horse_history, index):
    no_hist = np.empty((len(funcs)*len(X),))
    no_hist[:] = np.nan
    result = []
    for i in range(0, len(horse_history)):
        row = horse_history[i, :]
        past_rows = horse_history[:, index("race_date")] < row[index("race_date")]
        past_hist = horse_history[np.where(past_rows)][::-1]
        if past_hist.any():
            last_jrace_fresult = np.array([f(past_hist[:j, :], row, index) for f in funcs for j in X])
            result.append(last_jrace_fresult)
        else:
            result.append(no_hist)
    return np.array(result)

def yield_calculated(df, hist_pattern, feat_pattern):
    a = df.values
    columns = df.columns.tolist()
    index = lambda x: columns.index(x)
    funcs = list(feat_pattern.values())

    for name, hist in yield_horse_history(a, index("name")):
        np_df = X_result(funcs, hist_pattern, hist, index)
        past_columns = [f"{col}_{x}" for col in list(feat_pattern.keys()) for x in hist_pattern]
        hist_df = pd.DataFrame(np.concatenate([hist, np_df], axis=1), columns=columns+past_columns)
        hist_df = hist_df.set_index("id")
        yield name, hist_df
