import sqlite3

import dill as pickle
import lightgbm as lgb
import pandas as pd
from lightgbm import Dataset

import config
import feature_params


def prepare_dataset(df, target, noneed_columns=feature_params.NONEED_COLUMNS):
    if target in noneed_columns:
        noneed_columns.remove(target)
    query = df.groupby(feature_params.RACE_COLUMNS)["name"].count().values.tolist()
    x = df.drop(columns=noneed_columns)
    y = x.pop(target)
    dataset = Dataset(x, y, group=query)
    return dataset

if __name__ == "__main__":
    with sqlite3.connect(config.feat_db) as conn:
        df_feat = pd.read_sql_query("SELECT * FROM horse", conn)
    print(df_feat.head().T)
    print(df_feat.tail().T)
    train = prepare_dataset(df_feat.query(config.train_query), target=config.rank_target)
    valid = prepare_dataset(df_feat.query(config.valid_query), target=config.rank_target)
    test = prepare_dataset(df_feat.query(config.test_query), target=config.rank_target)

    rank_model = lgb.train(
        config.rank_params,
        train,
        num_boost_round=10000,
        valid_sets=valid,
        callbacks=[
            lgb.log_evaluation(10),
            lgb.early_stopping(50, first_metric_only=True),
        ],
    )
    with open(config.rank_file, "wb") as f:
        pickle.dump(rank_model, f)

    train = prepare_dataset(df_feat.query(config.train_query), target=config.reg_target)
    valid = prepare_dataset(df_feat.query(config.valid_query), target=config.reg_target)
    test = prepare_dataset(df_feat.query(config.test_query), target=config.reg_target)

    reg_model = lgb.train(
        config.reg_params,
        train,
        num_boost_round=10000,
        valid_sets=valid,
        callbacks=[
            lgb.log_evaluation(10),
            lgb.early_stopping(50),
        ],
    )
    with open(config.reg_file, "wb") as f:
        pickle.dump(reg_model, f)
