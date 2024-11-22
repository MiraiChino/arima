from addict import Dict

# scraping
from_date = '2008-01'
to_date = '2024-10'
netkeiba_horse_file = f'temp/netkeiba{from_date}_{to_date}.horses.feather'
netkeiba_race_file = f'temp/netkeiba{from_date}_{to_date}.races.feather'
netkeiba_file = f'temp/netkeiba{from_date}_{to_date}.feather'

# feature extraction
encoder_file = f'encoder{from_date}_{to_date}.pickle'
avetime_file = f'avetime{from_date}_{to_date}.feather'
aversrize_file = f'aversrize{from_date}_{to_date}.feather'
rating_file = f'temp/rating{from_date}_{to_date}.feather'
feat_file = f'feature{from_date}_{to_date}.feather'
breakpoint_file = f'breakpoint{from_date}_{to_date}.pickle'
racefeat_file = f'racefeat{from_date}_{to_date}.feather'
pred_validd_file = f'pred_valid{from_date}_{to_date}.pickle'

NONEED_COLUMNS = [
    'id', 'result', 'time', 'margin', 'pop', 'odds', 'last3f', \
    'weight', 'weight_change', 'corner', 'corner1', 'corner2', 'corner3', 'corner4', \
    'year', 'hold_num', 'race_num', 'day_num', 'race_date', 'race_name', \
    'start_time', 'prize1', 'prize2', 'prize3', 'prize4', 'prize5', 'prize', 'tanshou', 'hukushou', \
    'tanno1', 'tanno2', 'hukuno1', 'hukuno2', 'hukuno3', 'tan1', 'tan2', \
    'huku1', 'huku2', 'huku3', 'wide1', 'wide2', 'wide3', 'ren', \
    'uma1', 'uma2', 'puku', 'san1', 'san2', \
    'last3frel', 'weightrel', 'penaltywgt', 'oddsrslt', \
    'running', 'running_style', 'corner1_group', 'corner2_group', 'corner3_group', 'corner4_group', \
    'horse_newr', 'jockey_newr', 'trainer_newr', 'score' 
]
RACEDATE_COLUMNS = ['year', 'place_code', 'hold_num', 'day_num', 'race_num']

##### Train

# dataset
dataset_query = Dict(
    layer1_train="'2008-01-01' <= race_date <= '2020-12-31'",
    layer1_valid="'2021-01-01' <= race_date <= '2022-12-31'",
    layer2_train="'2021-01-01' <= race_date <= '2022-12-31'",
    layer2_valid="'2023-01-01' <= race_date <= '2024-12-31'",
    train="'2008-01-01' <= race_date <= '2021-12-31'",
    valid="'2022-01-01' <= race_date <= '2024-12-31'",
)

scaler_file = f'scaler_{from_date}_{to_date}.pickle'

# Layer1: LightGBM LambdaRank Prize1
l1_lgb_rank_prize1 = Dict(
    train=dataset_query.layer1_train,
    valid=dataset_query.layer1_valid,
    file=f'lgbrankprize1_{from_date}_{to_date}.pickle',
    target='prizeper',
    params={
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'lambdarank_truncation_level': 10,
        'ndcg_eval_at': [1],
        'boosting_type': 'gbdt',
        'learning_rate': 0.01,
    }
)

# Layer1: LightGBM LambdaRank Prize3
l1_lgb_rank_prize3 = Dict(
    train=dataset_query.layer1_train,
    valid=dataset_query.layer1_valid,
    file=f'lgbrankprize3_{from_date}_{to_date}.pickle',
    target='prizeper',
    params={
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'lambdarank_truncation_level': 10,
        'ndcg_eval_at': [3],
        'boosting_type': 'gbdt',
        'learning_rate': 0.01,
    }
)

# Layer1: LightGBM LambdaRank Prize5
l1_lgb_rank_prize5 = Dict(
    train=dataset_query.layer1_train,
    valid=dataset_query.layer1_valid,
    file=f'lgbrankprize5_{from_date}_{to_date}.pickle',
    target='prizeper',
    params={
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'lambdarank_truncation_level': 10,
        'ndcg_eval_at': [5],
        'boosting_type': 'gbdt',
        'learning_rate': 0.01,
    }
)

# Layer1: LightGBM LambdaRank Score1
l1_lgb_rank_score1 = Dict(
    train=dataset_query.layer1_train,
    valid=dataset_query.layer1_valid,
    file=f'lgbrankscore1_{from_date}_{to_date}.pickle',
    target='score',
    params={
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'lambdarank_truncation_level': 10,
        'ndcg_eval_at': [1],
        'boosting_type': 'gbdt',
        'learning_rate': 0.01,
    }
)

# Layer1: LightGBM LambdaRank Score3
l1_lgb_rank_score3 = Dict(
    train=dataset_query.layer1_train,
    valid=dataset_query.layer1_valid,
    file=f'lgbrankscore3_{from_date}_{to_date}.pickle',
    target='score',
    params={
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'lambdarank_truncation_level': 10,
        'ndcg_eval_at': [3],
        'boosting_type': 'gbdt',
        'learning_rate': 0.01,
    }
)

# Layer1: LightGBM LambdaRank Score5
l1_lgb_rank_score5 = Dict(
    train=dataset_query.layer1_train,
    valid=dataset_query.layer1_valid,
    file=f'lgbrankscore5_{from_date}_{to_date}.pickle',
    target='score',
    params={
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'lambdarank_truncation_level': 10,
        'ndcg_eval_at': [5],
        'boosting_type': 'gbdt',
        'learning_rate': 0.01,
    }
)
# Layer1: LightGBM Regression Prize
l1_lgb_regprize = Dict(
    train=dataset_query.layer1_train,
    valid=dataset_query.layer1_valid,
    file=f'lgbregprize_{from_date}_{to_date}.pickle',
    target='prizeper',
    params={
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'metric': 'rmse',
        'learning_rate': 0.01
    }
)

# Layer1: LightGBM Regression Score
l1_lgb_regscore = Dict(
    train=dataset_query.layer1_train,
    valid=dataset_query.layer1_valid,
    file=f'lgbregscore_{from_date}_{to_date}.pickle',
    target='score',
    params={
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'metric': 'rmse',
        'learning_rate': 0.01
    }
)

# Layer1: SGDRegressor
l1_sgd_regression = Dict(
    train=dataset_query.layer1_train,
    valid=dataset_query.layer1_valid,
    file=f'sgdrprize_{from_date}_{to_date}.pickle',
    target='prizeper',
    params={
        'verbose': 0,
    }
)

# Layer1: Lasso Regression
l1_lasso_regression = Dict(
    train=dataset_query.layer1_train,
    valid=dataset_query.layer1_valid,
    file=f'lassoprize_{from_date}_{to_date}.pickle',
    target='prizeper',
    params={
        'n_jobs': -1,
        'verbose': 0,
    }
)

# Layer1: ARD Regression
l1_ard_regression = Dict(
    train=dataset_query.layer1_train,
    valid=dataset_query.layer1_valid,
    file=f'ardprize_{from_date}_{to_date}.pickle',
    target='prizeper',
    params={
        'verbose': False,
    }
)

# Layer1: Huber Regression
l1_huber_regression = Dict(
    train=dataset_query.layer1_train,
    valid=dataset_query.layer1_valid,
    file=f'huberprize_{from_date}_{to_date}.pickle',
    target='prizeper',
    params={
    }
)

# Layer1: BayesianRidge Regression
l1_br_regression = Dict(
    train=dataset_query.layer1_train,
    valid=dataset_query.layer1_valid,
    file=f'brprize_{from_date}_{to_date}.pickle',
    target='prizeper',
    params={
    }
)

# Layer1: ExtraTreeｓRegressor
l1_etr_regression = Dict(
    train=dataset_query.layer1_train,
    valid=dataset_query.layer1_valid,
    file=f'etrprize_{from_date}_{to_date}.pickle',
    target='prizeper',
    params={
        'n_estimators': 100,
        'max_depth': 32,
        'max_features': 'log2',
        'n_jobs': -1,
        'verbose': 0,
    }
)

# Layer1: ElasticNet
l1_en_regression = Dict(
    train=dataset_query.layer1_train,
    valid=dataset_query.layer1_valid,
    file=f'enprize_{from_date}_{to_date}.pickle',
    target='prizeper',
    params={
    }
)

# Layer1: RandomForest Regression
l1_rf_regression = Dict(
    train=dataset_query.layer1_train,
    valid=dataset_query.layer1_valid,
    file=f'rfregprize_{from_date}_{to_date}.pickle',
    target='prizeper',
    params={
        'n_estimators': 100,
        'max_depth': 32,
        'max_features': 'log2',
        'n_jobs': -1,
        'verbose': 0,
    }
)

# Layer1: KNeighbors Regression
l1_kn_regression = Dict(
    train=dataset_query.layer1_train,
    valid=dataset_query.layer1_valid,
    file=f'knprize_{from_date}_{to_date}.pickle',
    target='prizeper',
    params={
        'k': 5,
    }
)

# Layer1: LogisticRegression Classification
l1_lr_classification = Dict(
    train=dataset_query.layer1_train,
    valid=dataset_query.layer1_valid,
    file=f'lrclsscore_{from_date}_{to_date}.pickle',
    target='score',
    params={
        'verbose': 0,
    }
)

# Layer1: Gaussian Naive Bayes Classification
l1_gnb_classification = Dict(
    train=dataset_query.layer1_train,
    valid=dataset_query.layer1_valid,
    file=f'gnbclsscore_{from_date}_{to_date}.pickle',
    target='score',
    params={
    }

l1_models = [
    l1_lgb_rank_prize1,
    l1_lgb_rank_prize3,
    l1_lgb_rank_prize5,
    l1_lgb_rank_score1,
    l1_lgb_rank_score3,
    l1_lgb_rank_score5,
    l1_lgb_regprize,
    l1_lgb_regscore,
    l1_sgd_regscore,
    l1_lasso_regression,
    l1_ard_regression,
    l1_huber_regression,
    l1_br_regression,
    # l1_etr_regression,
    l1_en_regression,
    l1_rf_regression,
    l1_kn_regression,
    # l1_lr_classification,
    # l1_gnb_classification,
]

# Layer2: Stacking model: LightGBM LambdaRank
l2_stacking_lgb_rank = Dict(
    train=dataset_query.layer2_train,
    valid=dataset_query.layer2_valid,
    file=f'stacking_{from_date}_{to_date}.pickle',
    target='prizeper',
    params={
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'lambdarank_truncation_level': 10,
        'ndcg_eval_at': [3],
        'boosting_type': 'gbdt',
        'learning_rate': 0.01,
    },
)

# 馬券的中率を予測するモデル
bakenhit_lgb_reg = Dict(
    feature_importance_model=l1_lgb_regprize.file,
    feature_importance_len=100, # 300個x10binで68GBにてOOM
    bins=10,
    train=dataset_query.train,
    valid=dataset_query.valid,
    file=f'bakenhit_{from_date}_{to_date}.pickle',
    target='bakenhit',
    params={
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'metric': 'rmse',
        'learning_rate': 0.01
    }
)
