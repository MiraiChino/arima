import sqlite3

import dill as pickle
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import Dataset

import config
import utils


def prepare_dataset(df, target):
    noneed_columns = config.NONEED_COLUMNS.copy()
    if target in noneed_columns:
        noneed_columns.remove(target)
    query = df.groupby(config.RACE_COLUMNS)["name"].count().values.tolist()
    print(len(query))
    x = df.drop(columns=noneed_columns)
    y = x.pop(target)
    dataset = Dataset(x, y, group=query)
    return dataset

if __name__ == "__main__":
    with sqlite3.connect(config.feat_db) as conn:
        reader = pd.read_sql_query("SELECT * FROM horse", conn, chunksize=10000)
        chunks = []
        for i, df in enumerate(reader):
            print(i+1, df.shape)
            df_feat_chunk = utils.reduce_mem_usage(df)
            chunks.append(df_feat_chunk)
        df_feat = pd.concat(chunks, ignore_index=True)
    print(df_feat.head().T)
    print(df_feat.tail().T)
    train = prepare_dataset(df_feat.query(config.train_query), target=config.rank_target)
    valid = prepare_dataset(df_feat.query(config.valid_query), target=config.rank_target)
    columns = train.data.columns

    rank_model = lgb.train(
        config.rank_params,
        train,
        num_boost_round=10000,
        valid_sets=valid,
        callbacks=[
            lgb.log_evaluation(10),
            lgb.early_stopping(300),
        ],
    )
    print(pd.Series(rank_model.feature_importance(importance_type='gain'), index=columns).sort_values(ascending=False)[:50])
    with open(config.rank_file, "wb") as f:
        pickle.dump(rank_model, f)

    train = prepare_dataset(df_feat.query(config.train_query), target=config.reg_target)
    valid = prepare_dataset(df_feat.query(config.valid_query), target=config.reg_target)
    columns = train.data.columns

    reg_model = lgb.train(
        config.reg_params,
        train,
        num_boost_round=10000,
        valid_sets=valid,
        callbacks=[
            lgb.log_evaluation(10),
            lgb.early_stopping(300),
        ],
    )
    print(pd.Series(reg_model.feature_importance(importance_type='gain'), index=columns).sort_values(ascending=False)[:50])
    with open(config.reg_file, "wb") as f:
        pickle.dump(reg_model, f)
