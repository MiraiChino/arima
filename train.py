import dill as pickle
import lightgbm as lgb
import pandas as pd
from lightgbm import Dataset

import config
import utils


def prepare_dataset(df, target):
    noneed_columns = config.NONEED_COLUMNS.copy()
    if target in noneed_columns:
        noneed_columns.remove(target)
    query = df.groupby(config.RACEDATE_COLUMNS)["name"].count().values.tolist()
    print(len(query))
    x = df.drop(columns=noneed_columns)
    y = x.pop(target)
    dataset = Dataset(x, y, group=query)
    return dataset

if __name__ == "__main__":
    df_feat = pd.read_feather(config.feat_file)
    df_feat = utils.reduce_mem_usage(df_feat)
    print(df_feat.head()[["id", "year", "race_date", "race_id", "horse_no"]])
    print(df_feat.tail()[["id", "year", "race_date", "race_id", "horse_no"]])

    for i, query in enumerate(config.splits):
        print(f"Folds: {i}")
        train = prepare_dataset(df_feat.query(query["train"]), target=config.rank_target)
        valid = prepare_dataset(df_feat.query(query["valid"]), target=config.rank_target)
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
        with open(f"{i}_{config.rank_file}", "wb") as f:
            pickle.dump(rank_model, f)

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
