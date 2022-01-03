import pickle
import sqlite3
from collections import namedtuple

import lightgbm as lgb
import pandas as pd
from lightgbm import Dataset

import encoder
from feature_extractor import (ave, calc_ave_time, diff, interval, same_count,
                               time, yield_history_aggdf)

NONEED_COLUMNS = [
    "index", "id", "result", "time", "margin", "pop", "odds", "last3f", \
    "weight", "weight_change", "corner", "corner3", "corner4", \
    "year", "hold_num", "race_num", "day_num", "race_date", "race_name", \
    "start_time", "prize1", "prize2", "prize3", "prize4", "prize5", "prize", "score"
]
RACE_CULMNS = ["year", "place_code", "hold_num", "day_num", "race_num"]

def prepare_dataset(df, target, noneed_columns=NONEED_COLUMNS):
    if target in noneed_columns:
        noneed_columns.remove(target)
    query = df.groupby(RACE_CULMNS)["name"].count().values.tolist()
    x = df.drop(columns=noneed_columns)
    y = x.pop(target)
    dataset = Dataset(x, y, group=query)
    return dataset

def prepare(output_db, input_db="netkeiba.sqlite", encoder_file="netkeiba.encoder", params_file="netkeiba.params"):
    with sqlite3.connect(input_db) as conn:
        df_original = pd.read_sql_query("SELECT * FROM horse", conn)
    df_format = encoder.format(df_original)
    netkeiba_encoder = encoder.HorseEncoder()
    netkeiba_encoder.fit(df_format)
    df_encoded = netkeiba_encoder.transform(df_format)
    with open(encoder_file, "wb") as f:
        pickle.dump(netkeiba_encoder, f)

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
    Params = namedtuple("Params",["ave_time", "hist_pattern", "feat_pattern"])
    params = Params(ave_time, hist_pattern, feat_pattern)
    with open(params_file, "wb") as f:
        pickle.dump(params, f)

    with sqlite3.connect(output_db) as conn:
        for name, hist_df in yield_history_aggdf(df_encoded, hist_pattern, feat_pattern):
            hist_df = hist_df.set_index("id")
            print("\r"+str(name),end="")
            hist_df.to_sql("horse", conn, if_exists="append", index_label="id")

if __name__ == "__main__":
    prepare("feature.sqlite")
    with sqlite3.connect("feature.sqlite") as conn:
        df_feat = pd.read_sql_query("SELECT * FROM horse", conn).sort_values("id").reset_index()

    train = prepare_dataset(df_feat.query("'2008-01-01' <= race_date <= '2017-12-31'"), target="score")
    valid = prepare_dataset(df_feat.query("'2018-01-01' <= race_date <= '2020-12-31'"), target="score")
    test = prepare_dataset(df_feat.query("'2021-01-01' <= race_date <= '2021-12-31'"), target="score")
    lgb_params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "lambdarank_truncation_level": 10,
        "ndcg_eval_at": [1, 2, 3, 4, 5],
        "boosting_type": "gbdt",
        "learning_rate": 0.01,
    }
    callbacks = [
        lgb.log_evaluation(10),
        lgb.early_stopping(50),
    ]
    rank_model = lgb.train(
        lgb_params,
        train,
        num_boost_round=10000,
        valid_sets=valid,
        callbacks=callbacks,
    )
    with open("rank.model", "wb") as f:
        pickle.dump(rank_model, f)

    train = prepare_dataset(df_feat.query("'2008-01-01' <= race_date <= '2017-12-31'"), target="prize")
    valid = prepare_dataset(df_feat.query("'2018-01-01' <= race_date <= '2020-12-31'"), target="prize")
    test = prepare_dataset(df_feat.query("'2021-01-01' <= race_date <= '2021-12-31'"), target="prize")
    lgb_params = {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'metric': 'rmse',
        'learning_rate': 0.01
    }
    reg_model = lgb.train(
        lgb_params,
        train,
        num_boost_round=10000,
        valid_sets=valid,
        callbacks=callbacks,
    )
    with open("reg.model", "wb") as f:
        pickle.dump(reg_model, f)
