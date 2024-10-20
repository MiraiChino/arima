from datetime import datetime
from pathlib import Path

import dill as pickle
import lightgbm
import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from lightgbm import Dataset
from sklearn.linear_model import LassoCV, SGDRegressor, ARDRegression, HuberRegressor, BayesianRidge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.tree import ExtraTreeRegressor

import config


class LogSummaryWriterCallback:
    
    def __init__(self, period=1, writer=None):
        self.period = period
        self.writer = writer
    
    def __call__(self, env):
        if (self.period > 0) and (env.evaluation_result_list) and (((env.iteration+1) % self.period)==0):
            if (self.writer is not None):
                scalars = {}
                for (name, metric, value, is_higher_better) in env.evaluation_result_list:
                    if (metric not in scalars.keys()):
                        scalars[metric] = {}
                    scalars[metric][name] = value
                    
                for key in scalars.keys():
                    self.writer.add_scalars(key, scalars[key], env.iteration+1)
            else:
                print(env.evaluation_result_list)

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
    # logger.info(f'columns: {x.columns.to_list()}')
    logger.info(f'{len(x.columns)=}')
    logger.info(x[['race_id', 'horse_no']])
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

def calculate_prize_percentiles(df_feat):
    logger.info('Calculating prize percentiles...')

    # 10%刻みでパーセンタイルを計算する
    lower9 = np.percentile(df_feat.query('0<prize')['prize'], [i for i in range(10, 100, 10)])
    middle = lower9[-1]
    
    # middleより大きい値でパーセンタイル計算
    middle9 = np.percentile(df_feat.query(f'{middle}<prize')['prize'], [i for i in range(10, 100, 10)])
    upper = middle9[-1]
    
    # upperより大きい値でパーセンタイル計算
    upper9 = np.percentile(df_feat.query(f'{upper}<prize')['prize'], [i for i in range(10, 101, 10)])

    logger.info(f'Percentiles calculated: lower9={lower9}, middle9={middle9}, upper9={upper9}')
    return lower9, middle9, upper9

def prepare_train_valid_dataset(df_feat, config):
    train_x, train_y, train_query = prepare_dataset(df_feat.query(config.train), target=config.target)
    valid_x, valid_y, valid_query = prepare_dataset(df_feat.query(config.valid), target=config.target)
    return train_x, train_y, train_query, valid_x, valid_y, valid_query

def load_model(model_file):
    logger.info(f"loading {model_file}")
    with open(model_file, "rb") as f:
        return pickle.load(f)

def save_model(model, model_file):
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f'Model saved to {model_file}')


def train_model(model_class, train_x, train_y, params, scaler=None):
    logger.info(f'Starting training for model: {model_class.__name__} with params {params}')

    model = model_class(**params)
    if scaler:
        logger.info('Scaling comlete for feature')
        model.fit(scaler.transform(train_x), train_y)
    else:
        model.fit(train_x, train_y)
    logger.info(f'Training complete for model: {model_class.__name__}')
    return model

def lgb(df, config, reg=False):
    logger.info(f'model: {config}')
    train_x, train_y, train_query, valid_x, valid_y, valid_query = prepare_train_valid_dataset(df, config)

    if reg:
        model = lightgbm_model(config, train_x, train_y, valid_x, valid_y)
    else:
        model = lightgbm_model(config, train_x, train_y, valid_x, valid_y, train_query, valid_query)

    logger.info('Predicting on validation set...')
    pred_valid_x = model.predict(valid_x, num_iteration=model.best_iteration)
    return model, pred_valid_x

def lightgbm_model(config, train_x, train_y, valid_x, valid_y, train_query=None, valid_query=None):
    model_file = Path(f'models/{config.file}')
    if model_file.exists():
        logger.info(f'Loading model from {model_file}')
        model = load_model(model_file)
    else:
        logger.info('Training new LightGBM model...')
        train = lightgbm.Dataset(train_x, train_y, group=train_query)
        valid = lightgbm.Dataset(valid_x, valid_y, group=valid_query)
        model = lightgbm.train(
            config.params,
            train,
            num_boost_round=10000,
            valid_sets=valid,
            callbacks=[
                lightgbm.log_evaluation(10),
                lightgbm.early_stopping(500, first_metric_only=True),
                LogSummaryWriterCallback(period=1, writer=SummaryWriter(log_dir=Path("temp", 'logs')))
            ],
        )
        save_model(model, model_file)
    return model

def regression(model_class, df, config, scaler=None):
    logger.info(f'model: {config}')
    train_x, train_y, _, valid_x, _, _ = prepare_train_valid_dataset(df, config)
    model_file = Path(f'models/{config.file}')

    if model_file.exists():
        model = load_model(model_file)
    else:
        model, _ = train_model(model_class, train_x, train_y, config.params, scaler)
        save_model(model, model_file)

    logger.info('Predicting on validation set...')
    pred_valid_x = model.predict(valid_x)
    return model, pred_valid_x

def kneighbors_regression(df, config):
    logger.info(f'model: {config}')
    train_x, train_y, _, valid_x, _, _ = prepare_train_valid_dataset(df, config)
    model_file = Path(f'models/{config.file}')

    if model_file.exists():
        model = UsearchKNeighborsRegressor()
        model.load(model_file)
    else:
        model, _ = train_model(UsearchKNeighborsRegressor, train_x, train_y, config.params)
        model.save(model_file)

    logger.info('Predicting on validation set...')
    pred_valid_x = model.predict(valid_x)
    return model, pred_valid_x

if __name__ == "__main__":
    now = datetime.now().strftime('%Y%m%d_%H%M')
    logger.add(f"temp/train_{now}.log")

    df_feat = pl.read_ipc(config.feat_file)
    df_feat = df_feat.with_columns(pl.col('start_time').dt.strftime("%H:%M"))
    logger.info(f"df_feat.columns: {df_feat.columns}")
    logger.info(df_feat.head().select(['year', 'race_date', 'race_id', 'horse_no', 'result']))
    logger.info(df_feat.tail().select(['year', 'race_date', 'race_id', 'horse_no', 'result']))

    df_feat = df_feat.to_pandas()
    lower9, middle9, upper9 = calculate_prize_percentiles(df_feat)
    logger.info(f"prize_percentile: {lower9=}, {middle9=}, {upper9=}")
    df_feat['prizeper'] = df_feat['prize'].map(percent_prize(lower9, middle9, upper9))

    l2_valid_x, l2_valid_y, l2_valid_query = prepare_dataset(df_feat.query(config.l2_stacking_lgb_rank.valid), target=config.l2_stacking_lgb_rank.target)

    ## Layer 1
    predicted_l1_valid_xs = []
    predicted_l2_valid_xs = []
    l1_valid_xs = []
    l1_valid_ys = []
    l1_preds_i = []
    l2_preds_i = []

    # Layer 1: LightGBM LambdaRank Prize
    logger.info(f'--- Layer 1: LightGBM LambdaRank Prize ---')
    lgb_rank_prize, l1_pred = lgb(df=df_feat, config=config.l1_lgb_rank_prize)
    l2_pred = lgb_rank_prize.predict(l2_valid_x, num_iteration=lgb_rank_prize.best_iteration)
    l1_preds_i.append(l1_pred)
    l2_preds_i.append(l2_pred)

    # Layer 1: LightGBM LambdaRank Score
    logger.info(f'--- Layer 1: LightGBM LambdaRank Score ---')
    lgb_rank_score, l1_pred = lgb(df=df_feat, config=config.l1_lgb_rank_score)
    l2_pred = lgb_rank_score.predict(l2_valid_x, num_iteration=lgb_rank_score.best_iteration)
    l1_preds_i.append(l1_pred)
    l2_preds_i.append(l2_pred)

    # Layer 1: LightGBM Regression
    logger.info(f'--- Layer 1: LightGBM Regression ---')
    lgb_regression, l1_pred = lgb(df=df_feat, config=config.l1_lgb_regression, reg=True)
    l2_pred = lgb_regression.predict(l2_valid_x, num_iteration=lgb_regression.best_iteration)
    l1_preds_i.append(l1_pred)
    l2_preds_i.append(l2_pred)

    # Scaler
    logger.info(f'--- Scaler ---')
    train_x, train_y, _, valid_x, _, _ = prepare_train_valid_dataset(df_feat, config.l1_sgd_regression)
    scaler = StandardScaler()
    scaler.fit(train_x)
    scaled_l2_valid_x = scaler.transform(l2_valid_x)
    save_model(scaler, Path(f'models/{config.scaler_file}'))

    # Layer 1: SGD Regressor
    logger.info(f'--- Layer 1: SGD Regressor ---')
    sgd_regression, l1_pred = regression(SGDRegressor, df=df_feat, config=config.l1_sgd_regression, scaler=scaler)
    l2_pred = sgd_regression.predict(scaled_l2_valid_x)
    l1_preds_i.append(l1_pred)
    l2_preds_i.append(l2_pred)

    # Layer 1: Lasso Regression
    logger.info(f'--- Layer 1: Lasso Regression ---')
    lasso, l1_pred = regression(LassoCV, df=df_feat, config=config.l1_lasso_regression, scaler=scaler)
    l2_pred = lasso.predict(scaled_l2_valid_x)
    l1_preds_i.append(l1_pred)
    l2_preds_i.append(l2_pred)

    # Layer1: ARD Regression
    logger.info(f'--- Layer 1: ARD Regression ---')
    ard, l1_pred = regression(ARDRegression, df=df_feat, config=config.l1_ard_regression, scaler=scaler)
    l2_pred = ard.predict(scaled_l2_valid_x)
    l1_preds_i.append(l1_pred)
    l2_preds_i.append(l2_pred)

    # Layer1: Huber Regression
    logger.info(f'--- Layer 1: Huber Regression ---')
    huber, l1_pred = regression(HuberRegressor, df=df_feat, config=config.l1_huber_regression, scaler=scaler)
    l2_pred = huber.predict(scaled_l2_valid_x)
    l1_preds_i.append(l1_pred)
    l2_preds_i.append(l2_pred)

    # Layer 1: BayesianRidge Regression
    logger.info(f'--- Layer 1: BayesianRidge Regression ---')
    bayesian_ridge, l1_pred = regression(BayesianRidge, df=df_feat, config=config.l1_br_regression, scaler=scaler)
    l2_pred = bayesian_ridge.predict(scaled_l2_valid_x)
    l1_preds_i.append(l1_pred)
    l2_preds_i.append(l2_pred)

    # Layer 1: ExtraTreeRegressor
    logger.info(f'--- Layer 1: ExtraTreeRegressor ---')
    extra_tree, l1_pred = regression(ExtraTreeRegressor, df=df_feat, config=config.l1_etr_regression)
    l2_pred = extra_tree.predict(l2_valid_x)
    l1_preds_i.append(l1_pred)
    l2_preds_i.append(l2_pred)

    # Layer 1: ElasticNet
    logger.info(f'--- Layer 1: ElasticNet ---')
    elasticnet, l1_pred = regression(ElasticNet, df=df_feat, config=config.l1_en_regression, scaler=scaler)
    l2_pred = elasticnet.predict(scaled_l2_valid_x)
    l1_preds_i.append(l1_pred)
    l2_preds_i.append(l2_pred)

    predicted_l1_valid_xs.append(np.column_stack(l1_preds_i))
    predicted_l2_valid_xs.append(np.column_stack(l2_preds_i))
    l1_valid_query = config.l1_lgb_rank_prize.valid
    valid_x, valid_y, valid_query = prepare_dataset(df_feat.query(l1_valid_query), target=config.l2_stacking_lgb_rank.target)
    l1_valid_xs.append(valid_x)
    l1_valid_ys.append(valid_y)

    # Layer2: Stacking LightGBM LambdaRank
    logger.info(f'--- Layer2: Stacking LightGBM LambdaRank ---')
    model = config.l2_stacking_lgb_rank
    logger.info(f'train: {model}')
    train_x = np.hstack((np.vstack(l1_valid_xs), np.vstack(predicted_l1_valid_xs)))
    train_y = np.hstack(l1_valid_ys)
    l2_train_query = valid_query
    valid_x = np.hstack((l2_valid_x, np.mean(predicted_l2_valid_xs, axis=0)))
    valid_y = l2_valid_y
    train = Dataset(train_x, train_y, group=l2_train_query)
    valid = Dataset(valid_x, valid_y, group=l2_valid_query)
    m = lightgbm.train(
        model.params,
        train,
        num_boost_round=10000,
        valid_sets=valid,
        callbacks=[
            lightgbm.log_evaluation(10),
            lightgbm.early_stopping(1000),
            LogSummaryWriterCallback(period=1, writer=SummaryWriter(log_dir=Path("temp", 'logs')))
        ],
    )
    logger.info(pd.Series(m.feature_importance(importance_type='gain'), index=m.feature_name()).sort_values(ascending=False)[:50])
    with open(f'models/{model.file}', 'wb') as f:
        pickle.dump(m, f)
