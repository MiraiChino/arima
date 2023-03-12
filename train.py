import itertools
from datetime import datetime
from pathlib import Path

import dill as pickle
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoost, Pool
from lightgbm import Dataset
from loguru import logger

import config


def prepare_dataset(df, target):
    noneed_columns = config.NONEED_COLUMNS.copy() + ['prizeper']
    if target in noneed_columns:
        noneed_columns.remove(target)
    cols = df.columns.tolist()
    for c in noneed_columns:
        if c not in cols:
            noneed_columns.remove(c)
    query = df.groupby(config.RACEDATE_COLUMNS)['name'].count().values.tolist()
    logger.info(f'target: {target}')
    logger.info(f'{len(query)}races')
    x = df.drop(columns=noneed_columns)
    y = x.pop(target)
    logger.info(x[['race_id', 'horse_no']])
    logger.info(f'{len(x.columns)=}')
    return x, y, query

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
    now = datetime.now().strftime('%Y%m%d_%H%M')
    logger.add(f"temp/train_{now}.log")

    df_feat = pl.read_ipc(config.feat_file)
    # logger.info(df_feat.head().glimpse())
    logger.info(df_feat.head().select(['year', 'race_date', 'race_id', 'horse_no', 'result']))
    logger.info(df_feat.tail().select(['year', 'race_date', 'race_id', 'horse_no', 'result']))
    df_feat = df_feat.with_column(pl.col('start_time').dt.strftime("%H:%M")).to_pandas()
    lower9 = np.percentile(df_feat.query('0<prize')['prize'], [i for i in range(10,100,10)])
    middle = lower9[-1]
    middle9 = np.percentile(df_feat.query(f'{middle}<prize')['prize'], [i for i in range(10,100,10)])
    upper = middle9[-1]
    upper9 = np.percentile(df_feat.query(f'{upper}<prize')['prize'], [i for i in range(10,101,10)])
    logger.info(f"prize_percentile: {lower9=}, {middle9=}, {upper9=}")
    df_feat['prizeper'] = df_feat['prize'].map(percent_prize(lower9, middle9, upper9))

    l2_valid_x, l2_valid_y, l2_valid_query = prepare_dataset(df_feat.query(config.stacking_valid), target=config.stacking_model.target)
    catfeatures_indices = [l2_valid_x.columns.get_loc(c) for c in config.cat_features if c in l2_valid_x]
    l2_valid_x_fillna = l2_valid_x.copy()
    l2_valid_x_fillna.iloc[:, catfeatures_indices] = l2_valid_x_fillna.iloc[:, catfeatures_indices].fillna(-1).astype(int)
    l2_valid = Pool(data=l2_valid_x_fillna, label=l2_valid_y, group_id=l2_valid_x['race_id'].tolist(), cat_features=catfeatures_indices)

    predicted_l1_valid_xs = []
    predicted_l2_valid_xs = []
    queries = []
    l1_valid_xs = []
    l1_valid_ys = []
    for i, query in enumerate(config.splits):
        logger.info(f'Folds: {i}')
        logger.info(query)
        l1_preds_i = []
        l2_preds_i = []
        for model in config.lgb_models:
            logger.info(f'train: {model}')
            train_x, train_y, train_query = prepare_dataset(df_feat.query(query.train), target=model.target)
            valid_x, valid_y, valid_query = prepare_dataset(df_feat.query(query.valid), target=model.target)
            train = Dataset(train_x, train_y, group=train_query)
            valid = Dataset(valid_x, valid_y, group=valid_query)
            model_file = Path(f'models/{i}_{model.file}')
            if model_file.exists():
                logger.info(f"loading {model_file}")
                with open(model_file, "rb") as f:
                    m = pickle.load(f)
            else:
                m = lgb.train(
                    model.params,
                    train,
                    num_boost_round=10000,
                    valid_sets=valid,
                    callbacks=[
                        lgb.log_evaluation(10),
                        lgb.early_stopping(300),
                    ],
                )
            l1_pred = m.predict(valid_x, num_iteration=m.best_iteration)
            l1_preds_i.append(l1_pred)
            l2_pred = m.predict(l2_valid_x, num_iteration=m.best_iteration)
            l2_preds_i.append(l2_pred)

            logger.info(pd.Series(m.feature_importance(importance_type='gain'), index=train_x.columns).sort_values(ascending=False)[:50])
            if not model_file.exists():
                with open(f'models/{i}_{model.file}', 'wb') as f:
                    pickle.dump(m, f)

        for model in config.cat_models:
            logger.info(f'train: {model}')
            train_x, train_y, train_query = prepare_dataset(df_feat.query(query.train), target=model.target)
            valid_x, valid_y, valid_query = prepare_dataset(df_feat.query(query.valid), target=model.target)
            train_x.iloc[:, catfeatures_indices] = train_x.iloc[:, catfeatures_indices].fillna(-1).astype(int)
            valid_x.iloc[:, catfeatures_indices] = valid_x.iloc[:, catfeatures_indices].fillna(-1).astype(int)
            train = Pool(data=train_x, label=train_y, group_id=train_x['race_id'].tolist(), cat_features=catfeatures_indices)
            valid = Pool(data=valid_x, label=valid_y, group_id=valid_x['race_id'].tolist(), cat_features=catfeatures_indices)
            model_file = Path(f'models/{i}_{model.file}')
            
            if model_file.exists():
                logger.info(f"loading {model_file}")
                with open(model_file, "rb") as f:
                    m = pickle.load(f)
            else:
                m = CatBoost(model.param)
                m.fit(
                    train,
                    eval_set=valid,
                    verbose_eval=10,
                )
            l1_pred = m.predict(valid, prediction_type='RawFormulaVal')
            l1_preds_i.append(l1_pred)
            l2_pred = m.predict(l2_valid, prediction_type='RawFormulaVal')
            l2_preds_i.append(l2_pred)

            logger.info(pd.Series(m.get_feature_importance(data=train), index=train_x.columns).sort_values(ascending=False)[:50])
            if not model_file.exists():
                with open(model_file, 'wb') as f:
                    pickle.dump(m, f)
        predicted_l1_valid_xs.append(np.column_stack(l1_preds_i))
        predicted_l2_valid_xs.append(np.column_stack(l2_preds_i))
        valid_x, valid_y, valid_query = prepare_dataset(df_feat.query(query.valid), target=config.stacking_model.target)
        queries.append(valid_query)
        l1_valid_xs.append(valid_x)
        l1_valid_ys.append(valid_y)

    # stacking
    model = config.stacking_model
    logger.info(f'train: {model}')
    train_x = np.hstack((np.vstack(l1_valid_xs), np.vstack(predicted_l1_valid_xs))) # (544868, 752+4)
    train_y = np.hstack(l1_valid_ys) # (544868,)
    train_query = np.fromiter(itertools.chain(*queries), int)
    valid_x = np.hstack((l2_valid_x, np.mean(predicted_l2_valid_xs, axis=0))) # (142983, 752+4)
    valid_y = l2_valid_y # (142983,)
    valid_query = l2_valid_query
    train = Dataset(train_x, train_y, group=train_query)
    valid = Dataset(valid_x, valid_y, group=valid_query)
    m = lgb.train(
        model.params,
        train,
        num_boost_round=10000,
        valid_sets=valid,
        callbacks=[
            lgb.log_evaluation(10),
            lgb.early_stopping(300),
        ],
    )
    columns = l2_valid_x.columns.tolist() + ["lgbrankscore", "lgbrankprize", "lgbregprize", "catrankprize"]
    logger.info(pd.Series(m.feature_importance(importance_type='gain'), index=columns).sort_values(ascending=False)[:50])
    with open(f'models/{model.file}', 'wb') as f:
        pickle.dump(m, f)
