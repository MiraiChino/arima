import sqlite3

import dill as pickle
import lightgbm as lgb
import pandas as pd
from lightgbm import Dataset

import feature_params


def prepare_dataset(df, target, noneed_columns=feature_params.NONEED_COLUMNS):
    if target in noneed_columns:
        noneed_columns.remove(target)
    query = df.groupby(feature_params.RACE_CULMNS)["name"].count().values.tolist()
    x = df.drop(columns=noneed_columns)
    y = x.pop(target)
    dataset = Dataset(x, y, group=query)
    return dataset

if __name__ == "__main__":
    with sqlite3.connect("feature.sqlite") as conn:
        df_feat = pd.read_sql_query("SELECT * FROM horse", conn)
    print(df_feat.head().T)
    print(df_feat.tail().T)
    train = prepare_dataset(df_feat.query("'2008-01-01' <= race_date <= '2017-12-31'"), target="score")
    valid = prepare_dataset(df_feat.query("'2018-01-01' <= race_date <= '2020-12-31'"), target="score")
    test = prepare_dataset(df_feat.query("'2021-01-01' <= race_date <= '2021-12-31'"), target="score")
    lgb_params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "lambdarank_truncation_level": 10,
        "ndcg_eval_at": [5, 4, 3, 2, 1],
        "boosting_type": "gbdt",
        "learning_rate": 0.01,
    }
    rank_model = lgb.train(
        lgb_params,
        train,
        num_boost_round=10000,
        valid_sets=valid,
        callbacks=[
            lgb.log_evaluation(10),
            lgb.early_stopping(50, first_metric_only=True),
        ],
    )
    with open("rank_model.pickle", "wb") as f:
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
        callbacks=[
            lgb.log_evaluation(10),
            lgb.early_stopping(50),
        ],
    )
    with open("reg_model.pickle", "wb") as f:
        pickle.dump(reg_model, f)
