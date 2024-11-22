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
from sklearn.linear_model import LassoCV, SGDRegressor, ARDRegression, HuberRegressor, BayesianRidge, ElasticNet, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
# from sklearn.svm import LinearSVR
# from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import ndcg_score, mean_squared_error

import config
from knn import UsearchKNeighborsRegressor


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

def nddcg_at(k, y_pred, y_valid, query):
    # グループごとにNDCGを計算
    start = 0
    scores = []
    for group_size in query:
        end = start + group_size
        true_relevance = y_valid[start:end]     # グループごとの真のラベル
        predicted_scores = y_pred[start:end]  # グループごとの予測スコア
        score = ndcg_score([true_relevance], [predicted_scores], k=k)  # NDCG計算
        scores.append(score)
        start = end
    # 全体の平均NDCGを計算
    mean_ndcg = sum(scores) / len(scores)
    return mean_ndcg

def lgb(df, config, reg=False):
    logger.info(f'model: {config}')
    train_x, train_y, train_query, valid_x, valid_y, valid_query = prepare_train_valid_dataset(df, config)

    if reg:
        model = lightgbm_model(config, train_x, train_y, valid_x, valid_y)
    else:
        model = lightgbm_model(config, train_x, train_y, valid_x, valid_y, train_query, valid_query)

    logger.info(pd.Series(model.feature_importance(importance_type='gain'), index=model.feature_name()).sort_values(ascending=False)[:50])
    logger.info('Predicting on validation set...')
    pred_valid_x = model.predict(valid_x, num_iteration=model.best_iteration)

    if reg:
        rmse = np.sqrt(mean_squared_error(valid_y, pred_valid_x))
        logger.info(f'Validation RMSE: {rmse}')
    else:
        ndcg3 = nddcg_at(3, pred_valid_x, valid_y, valid_query)
        logger.info(f'Validation NDCG@3: {ndcg3}')
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
    train_x, train_y, _, valid_x, valid_y, _ = prepare_train_valid_dataset(df, config)
    model_file = Path(f'models/{config.file}')

    if model_file.exists():
        model = load_model(model_file)
    else:
        model = train_model(model_class, train_x, train_y, config.params, scaler)
        save_model(model, model_file)

    logger.info('Predicting on validation set...')
    pred_valid_x = model.predict(valid_x)
    rmse = np.sqrt(mean_squared_error(valid_y, pred_valid_x))
    logger.info(f'Validation RMSE: {rmse:.4f}')
    return model, pred_valid_x

def knn_regression(model_class, df, config, scaler=None):
    logger.info(f'model: {config}')
    train_x, train_y, _, valid_x, valid_y, _ = prepare_train_valid_dataset(df, config)
    model_file = Path(f'models/{config.file}')

    if model_file.exists():
        model = UsearchKNeighborsRegressor()
        model.load(model_file)
    else:
        model = train_model(model_class, train_x, train_y, config.params, scaler)
        model.save(model_file)

    logger.info('Predicting on validation set...')
    pred_valid_x = model.predict(valid_x)
    rmse = np.sqrt(mean_squared_error(valid_y, pred_valid_x))
    logger.info(f'Validation RMSE: {rmse:.4f}')
    return model, pred_valid_x

def classification_as_regression(model_class, df, config, scaler=None):
    logger.info(f'model: {config}')
    train_x, train_y, _, valid_x, valid_y, _ = prepare_train_valid_dataset(df, config)
    model_file = Path(f'models/{config.file}')

    if model_file.exists():
        model, class_labels = load_model(model_file)
    else:
        model = train_model(model_class, train_x, train_y, config.params, scaler)
        class_labels = np.sort(np.unique(train_y))
        save_model((model, class_labels), model_file)

    logger.info('Predicting on validation set...')
    pred_valid_x = classification_to_regression(model, class_labels, valid_x)
    rmse = np.sqrt(mean_squared_error(valid_y, pred_valid_x))
    logger.info(f'Validation RMSE: {rmse:.4f}')
    return model, class_labels, pred_valid_x

def classification_to_regression(model, class_labels, x):
    pred_x_proba = model.predict_proba(x)
    pred_x = np.dot(pred_x_proba, class_labels)
    return pred_x

def standard_scaler():
    train_x, train_y, _, valid_x, _, _ = prepare_train_valid_dataset(df_feat, config.l1_sgd_regression)
    scaler_file = Path(f'models/{config.scaler_file}')
    if scaler_file.exists():
        scaler = load_model(scaler_file)
    else:
        scaler = StandardScaler()
        scaler.fit(train_x)
        save_model(scaler, scaler_file)
    return scaler

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

    # Scaler
    logger.info(f'--- Scaler ---')
    scaler = standard_scaler()
    scaled_l2_valid_x = scaler.transform(l2_valid_x)

    # Layer 1: LightGBM LambdaRank Prize1
    logger.info(f'--- Layer 1: LightGBM LambdaRank Prize1 ---')
    lgb_rank_prize1, l1_pred = lgb(df=df_feat, config=config.l1_lgb_rank_prize1)
    l2_pred = lgb_rank_prize1.predict(l2_valid_x, num_iteration=lgb_rank_prize1.best_iteration)
    l1_preds_i.append(l1_pred)
    l2_preds_i.append(l2_pred)

    # Layer 1: LightGBM LambdaRank Prize3
    logger.info(f'--- Layer 1: LightGBM LambdaRank Prize3 ---')
    lgb_rank_prize3, l1_pred = lgb(df=df_feat, config=config.l1_lgb_rank_prize3)
    l2_pred = lgb_rank_prize3.predict(l2_valid_x, num_iteration=lgb_rank_prize3.best_iteration)
    l1_preds_i.append(l1_pred)
    l2_preds_i.append(l2_pred)

    # Layer 1: LightGBM LambdaRank Prize5
    logger.info(f'--- Layer 1: LightGBM LambdaRank Prize5 ---')
    lgb_rank_prize5, l1_pred = lgb(df=df_feat, config=config.l1_lgb_rank_prize5)
    l2_pred = lgb_rank_prize5.predict(l2_valid_x, num_iteration=lgb_rank_prize5.best_iteration)
    l1_preds_i.append(l1_pred)
    l2_preds_i.append(l2_pred)

    # Layer 1: LightGBM LambdaRank Score1
    logger.info(f'--- Layer 1: LightGBM LambdaRank Score1 ---')
    lgb_rank_score1, l1_pred = lgb(df=df_feat, config=config.l1_lgb_rank_score1)
    l2_pred = lgb_rank_score1.predict(l2_valid_x, num_iteration=lgb_rank_score1.best_iteration)
    l1_preds_i.append(l1_pred)
    l2_preds_i.append(l2_pred)

    # Layer 1: LightGBM LambdaRank Score3
    logger.info(f'--- Layer 1: LightGBM LambdaRank Score3 ---')
    lgb_rank_score3, l1_pred = lgb(df=df_feat, config=config.l1_lgb_rank_score3)
    l2_pred = lgb_rank_score3.predict(l2_valid_x, num_iteration=lgb_rank_score3.best_iteration)
    l1_preds_i.append(l1_pred)
    l2_preds_i.append(l2_pred)

    # Layer 1: LightGBM LambdaRank Score5
    logger.info(f'--- Layer 1: LightGBM LambdaRank Score5 ---')
    lgb_rank_score5, l1_pred = lgb(df=df_feat, config=config.l1_lgb_rank_score5)
    l2_pred = lgb_rank_score5.predict(l2_valid_x, num_iteration=lgb_rank_score5.best_iteration)
    l1_preds_i.append(l1_pred)
    l2_preds_i.append(l2_pred)

    # Layer 1: LightGBM Regression Prize
    logger.info(f'--- Layer 1: LightGBM Regression Prize ---')
    lgb_regprize, l1_pred = lgb(df=df_feat, config=config.l1_lgb_regprize, reg=True)
    l2_pred = lgb_regprize.predict(l2_valid_x, num_iteration=lgb_regprize.best_iteration)
    l1_preds_i.append(l1_pred)
    l2_preds_i.append(l2_pred)

    # Layer 1: LightGBM Regression Score
    logger.info(f'--- Layer 1: LightGBM Regression Score ---')
    lgb_regscore, l1_pred = lgb(df=df_feat, config=config.l1_lgb_regscore, reg=True)
    l2_pred = lgb_regscore.predict(l2_valid_x, num_iteration=lgb_regscore.best_iteration)
    l1_preds_i.append(l1_pred)
    l2_preds_i.append(l2_pred)

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

    # Layer 1: ExtraTreesRegressor
    # logger.info(f'--- Layer 1: ExtraTreesRegressor ---')
    # extra_tree, l1_pred = regression(ExtraTreesRegressor, df=df_feat, config=config.l1_etr_regression)
    # l2_pred = extra_tree.predict(l2_valid_x)
    # l1_preds_i.append(l1_pred)
    # l2_preds_i.append(l2_pred)

    # Layer 1: ElasticNet
    logger.info(f'--- Layer 1: ElasticNet ---')
    elasticnet, l1_pred = regression(ElasticNet, df=df_feat, config=config.l1_en_regression, scaler=scaler)
    l2_pred = elasticnet.predict(scaled_l2_valid_x)
    l1_preds_i.append(l1_pred)
    l2_preds_i.append(l2_pred)

    # Layer 1: RandomForest Regression
    logger.info(f'--- Layer 1: RandomForest Regression ---')
    randomforest, l1_pred = regression(RandomForestRegressor, df=df_feat, config=config.l1_rf_regression)
    l2_pred = randomforest.predict(scaled_l2_valid_x)
    l1_preds_i.append(l1_pred)
    l2_preds_i.append(l2_pred)

    # Layer 1: KNeighbors Regression
    logger.info(f'--- Layer 1: KNeighbors Regression ---')
    kn_regression, l1_pred = knn_regression(UsearchKNeighborsRegressor, df=df_feat, config=config.l1_kn_regression)
    l2_pred = kn_regression.predict(l2_valid_x)
    l1_preds_i.append(l1_pred)
    l2_preds_i.append(l2_pred)

    # Layer 1: LogisticRegression Classification
    # logger.info(f'--- Layer 1: LogisticRegression Classification ---')
    # logi, class_labels, l1_pred = classification_as_regression(LogisticRegression, df=df_feat, config=config.l1_lr_classification, scaler=scaler)
    # l2_pred = classification_to_regression(logi, class_labels, scaled_l2_valid_x)
    # l1_preds_i.append(l1_pred)
    # l2_preds_i.append(l2_pred)

    # Layer 1: Gaussian Naive Bayes Classification
    # logger.info(f'--- Layer 1: Gaussian Naive Bayes Classification ---')
    # gnb, class_labels, l1_pred = classification_as_regression(GaussianNB, df=df_feat, config=config.l1_gnb_classification, scaler=scaler)
    # l2_pred = classification_to_regression(gnb, class_labels, scaled_l2_valid_x)
    # l1_preds_i.append(l1_pred)
    # l2_preds_i.append(l2_pred)

    predicted_l1_valid_xs.append(np.column_stack(l1_preds_i))
    predicted_l2_valid_xs.append(np.column_stack(l2_preds_i))
    l1_valid_query = config.l1_lgb_rank_prize1.valid
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

    columns = l2_valid_x.columns.tolist() + [
        "lgbrankprize1",
        "lgbrankprize3",
        "lgbrankprize5",
        "lgbrankscore1",
        "lgbrankscore3",
        "lgbrankscore5",
        "lgbregprize",
        "lgbregscore",
        # "sgd",
        "lasso",
        "ard",
        "huber",
        "bayesianridge",
        # "extratrees",
        "elasticnet",
        # "randomforest",
        # "logisticreg",
        # "gaussiannb",
    ]
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
    logger.info(pd.Series(m.feature_importance(importance_type='gain'), index=columns).sort_values(ascending=False)[:50])
    logger.info('Predicting on validation set...')
    pred_valid_x = m.predict(valid_x, num_iteration=m.best_iteration)
    ndcg3 = nddcg_at(3, pred_valid_x, valid_y, l2_valid_query)
    logger.info(f'Validation NDCG@3: {ndcg3}')
    with open(f'models/{model.file}', 'wb') as f:
        pickle.dump(m, f)
