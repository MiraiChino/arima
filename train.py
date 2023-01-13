import dill as pickle
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import Dataset

import config


def prepare_dataset(df, target):
    noneed_columns = config.NONEED_COLUMNS.copy() + ["prizeper"]
    if target in noneed_columns:
        noneed_columns.remove(target)
    cols = df.columns.tolist()
    for c in noneed_columns:
        if c not in cols:
            noneed_columns.remove(c)
    query = df.groupby(config.RACEDATE_COLUMNS)["name"].count().values.tolist()
    print(f"target: {target}"
    print(f"{len(query)}races")
    x = df.drop(columns=noneed_columns)
    y = x.pop(target)
    print(x[["race_id", "horse_no"]])
    dataset = Dataset(x, y, group=query)
    return dataset

def percent_prize(lower, middle, upper):
    def wrapper(x):
        if x == 0:
            return 0
        for i, p in enumerate(lower):
            if x <= p:
                return i+1
        for i, p in enumerate(middle):
            if x <= p:
                return i+11
        for i, p in enumerate(upper):
            if x <= p:
                return i+20
    return wrapper


if __name__ == "__main__":
    df_feat = pd.read_feather(config.feat_file)
    print(df_feat.head()[["id", "year", "race_date", "race_id", "horse_no"]])
    print(df_feat.tail()[["id", "year", "race_date", "race_id", "horse_no"]])
    lower9 = np.percentile(df_feat.query("0<prize")["prize"], [i for i in range(10,100,10)])
    middle = lower9[-1]
    middle9 = np.percentile(df_feat.query(f"{middle}<prize")["prize"], [i for i in range(10,100,10)])
    upper = middle9[-1]
    upper9 = np.percentile(df_feat.query(f"{upper}<prize")["prize"], [i for i in range(10,101,10)])
    df_feat["prizeper"] = df_feat["prize"].map(percent_prize(lower9, middle9, upper9))

    for i, query in enumerate(config.splits):
        print(f"Folds: {i}")
        train = prepare_dataset(df_feat.query(query["train"]), target=config.rankscore_target)
        valid = prepare_dataset(df_feat.query(query["valid"]), target=config.rankscore_target)
        columns = train.data.columns

        rankscore_model = lgb.train(
            config.rankscore_params,
            train,
            num_boost_round=10000,
            valid_sets=valid,
            callbacks=[
                lgb.log_evaluation(10),
                lgb.early_stopping(300),
            ],
        )
        print(pd.Series(rankscore_model.feature_importance(importance_type='gain'), index=columns).sort_values(ascending=False)[:50])
        with open(f"{i}_{config.rankscore_file}", "wb") as f:
            pickle.dump(rankscore_model, f)

        train = prepare_dataset(df_feat.query(query["train"]), target=config.rankprize_target)
        valid = prepare_dataset(df_feat.query(query["valid"]), target=config.rankprize_target)
        columns = train.data.columns

        rankprize_model = lgb.train(
            config.rankprize_params,
            train,
            num_boost_round=10000,
            valid_sets=valid,
            callbacks=[
                lgb.log_evaluation(10),
                lgb.early_stopping(300),
            ],
        )
        print(pd.Series(rankprize_model.feature_importance(importance_type='gain'), index=columns).sort_values(ascending=False)[:50])
        with open(f"{i}_{config.rankprize_file}", "wb") as f:
            pickle.dump(rankprize_model, f)

        train = prepare_dataset(df_feat.query(query["train"]), target=config.reg_target)
        valid = prepare_dataset(df_feat.query(query["valid"]), target=config.reg_target)
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
        with open(f"{i}_{config.reg_file}", "wb") as f:
            pickle.dump(reg_model, f)
