import argparse
from pathlib import Path

import dill as pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import config
import utils
from encoder import HorseEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true")
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
    return {key: race["time"].astype(float).mean() for key, race in df.groupby(["field", "distance", "field_condition"])}

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
            import traceback
            print(traceback.format_exc())
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

def prepare():
    df_races = pd.read_feather(config.netkeiba_race_file)
    df_horses = pd.read_feather(config.netkeiba_horse_file)
    df_original = pd.merge(df_horses, df_races, on='race_id', how='left')
    if "index" in df_original.columns:
        df_original = df_original.drop(columns="index")

    horse_encoder = HorseEncoder()
    df_format = horse_encoder.format(df_original)
    df_encoded = horse_encoder.fit_transform(df_format)
    with open(config.encoder_file, "wb") as f:
        pickle.dump(horse_encoder, f)
    
    ave_time = calc_ave_time(df_encoded)
    hist_pattern = config.hist_pattern
    feat_pattern = config.feature_pattern(ave_time)
    df_avetime = pd.Series(ave_time).rename_axis(["field", "distance", "field_condition"]).reset_index(name="ave_time")
    df_avetime.to_feather(config.avetime_file)

    cols = list(feat_pattern.keys())
    for column in cols:
        for name, hist_df in yield_history_aggdf(df_encoded, column, hist_pattern, feat_pattern[column]):
            hist_df.to_feather(f"feat/{column}_{int(name)}.feather")
            print(f"\r{column}:{name}", end="")

def out():
    feat_pattern = config.feature_pattern(0)
    cols = list(feat_pattern.keys())

    df_feats = {}
    all_feat_files = [p for p in sorted(Path("feat").iterdir(), key=lambda p: p.stat().st_mtime) if p.suffix == ".feather"]
    for column in cols:
        feat_files = [str(p) for p in all_feat_files if column in p.name]
        dfs = []
        print(column)
        for file in tqdm(feat_files):
            df_chunk = pd.read_feather(file)
            dfs.append(df_chunk)
        df_feats[column] = pd.concat(dfs)
        print(f"{column} concatted")
        df_feats[column] = pd.concat([df.sort_values("horse_no") for _, df in df_feats[column].groupby(["race_date", "race_id"])])
        df_feats[column] = df_feats[column].reset_index(drop=True)
        df_feats[column]["id"] = df_feats[column].index
        print(df_feats[column][["id", "year", "race_date", "race_id", "horse_no"]])
    df_feat = df_feats[cols[0]]
    for column in cols[1:]:
        cols_to_use = df_feats[column].columns.difference(df_feat.columns).tolist() + ["id"]
        df_feat = pd.merge(df_feat, df_feats[column][cols_to_use], on="id")
    df_feat = utils.reduce_mem_usage(df_feat)
    df_feat.to_feather(config.feat_file)

def update():
    if not Path("netkeiba.log").exists():
        print(f"no {config.encoder_file}")
        return
    with open("netkeiba.log", 'r') as f:
        filenames = f.readlines()

    dfs = []
    for filename in filenames:
        df_races = pd.read_feather(f"{filename}.races.feather")
        df_horses = pd.read_feather(f"{filename}.horses.feather")
        df = pd.merge(df_horses, df_races, on='race_id', how='left')
        dfs.append(df)
    df_original = pd.concat(dfs)
    if "index" in df_original.columns:
        df_original = df_original.drop(columns="index")

    if not Path(config.encoder_file).exists():
        print(f"no {config.encoder_file}")
        return
    with open(config.encoder_file, "rb") as f:
        horse_encoder = pickle.load(f)
    df_format = horse_encoder.format(df_original)
    df_encoded = horse_encoder.transform(df_format)

    df_avetime = pd.read_feather(config.avetime_file)
    ave_time = {(f, d, fc): t for f, d, fc, t in df_avetime.to_dict(orient="split")["data"]}
    feat_pattern = config.feature_pattern(ave_time)
    cols = list(feat_pattern.keys())
    for column in cols:
        print(column)
        past_columns = [f"{col}_{x}" for col in list(feat_pattern[column].keys()) for x in config.hist_pattern]
        funcs = list(feat_pattern[column].values())

        for name in tqdm(df_encoded[column].unique()):
        # for _, row in tqdm(list(df_encoded.iterrows())):
            try:
                # name = int(row[column])
                df_feat = pd.read_feather(f"feat/{column}_{int(name)}.feather")
                pre_hist = df_feat.drop(columns=past_columns)
                # TODO: 必ず追加するようになっているので、同じ行があったら削除する
                # TODO: 同じ馬を何回も計算してる
                rows = df_encoded.query(f"{column}=={name}")
                hist = pd.concat([pre_hist, rows])
                hist_nodup = hist.drop_duplicates(subset=['result', 'gate', 'horse_no', 'name', 'race_date', 'prize'])
                if len(hist_nodup) <= len(pre_hist):
                    continue
                print(len(hist_nodup), len(pre_hist))
                a, columns, index = df2np(hist_nodup)
                print(df_feat[["result", "horse_no", "name", "prize", "horse_prize_999999"]])
                print(hist_nodup[["result", "horse_no", "name", "prize"]])
                import pdb; pdb.set_trace()
                # ここからは未確認
                for i in range(len(pre_hist), len(hist_nodup)):
                    targetrow_agg = agg_history_i(i, funcs, config.hist_pattern, a, index)
                    row_df_column = pd.DataFrame([targetrow_agg], columns=past_columns)
                    import pdb; pdb.set_trace()
                    
                    # hist_df = pd.concat([hist_df, row_df_column], axis="columns")
                    # df_agg = search_history(row, config.hist_pattern, feat_pattern, df_hist)
                    # new_feat = pd.concat([df_feat, df_agg]).reset_index(drop=True)
            except:
                import traceback
                print(traceback.format_exc())
                print(f"didn't update feat/{column}_{name}.feather")
            else:
                pass
                # new_feat.to_feather(f"feat/{column}_{name}.feather")

if __name__ == "__main__":
    args = parse_args()
    if args.update:
        update()
        # out()
    else:
        prepare()
        out()