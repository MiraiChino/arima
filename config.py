from addict import Dict

# scraping
from_date = '2008-01'
to_date = '2024-08'
netkeiba_horse_file = f'temp/netkeiba{from_date}_{to_date}.horses.feather'
netkeiba_race_file = f'temp/netkeiba{from_date}_{to_date}.races.feather'
netkeiba_file = f'temp/netkeiba{from_date}_{to_date}.feather'

# feature extraction
encoder_file = f'encoder{from_date}_{to_date}.pickle'
avetime_file = f'avetime{from_date}_{to_date}.feather'
aversrize_file = f'aversrize{from_date}_{to_date}.feather'
rating_file = f'temp/rating{from_date}_{to_date}.feather'
feat_file = f'feature{from_date}_{to_date}.feather'
racefeat_file = f'racefeat{from_date}_{to_date}.feather'

NONEED_COLUMNS = [
    'id', 'result', 'time', 'margin', 'pop', 'odds', 'last3f', \
    'weight', 'weight_change', 'corner', 'corner1', 'corner2', 'corner3', 'corner4', \
    'year', 'hold_num', 'race_num', 'day_num', 'race_date', 'race_name', \
    'start_time', 'prize1', 'prize2', 'prize3', 'prize4', 'prize5', 'prize', 'tanshou', 'hukushou', \
    'tanno1', 'tanno2', 'hukuno1', 'hukuno2', 'hukuno3', 'tan1', 'tan2', \
    'huku1', 'huku2', 'huku3', 'wide1', 'wide2', 'wide3', 'ren', \
    'uma1', 'uma2', 'puku', 'san1', 'san2', \
    'last3frel', 'weightrel', 'penaltywgt', 'oddsrslt', \
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
)

# category features
cat_features = ['name', 'horse_id', 'sex', 'jockey', 'jockey_id', 'trainer','trainer_id',
                'race_id', 'field', 'turn', 'weather', 'field_condition', 'race_condition', 'place_code']

# Layer1: LightGBM LambdaRank Prize
l1_lgb_rank_prize = Dict(
    train=dataset_query.layer1_train,
    valid=dataset_query.layer1_valid,
    file=f'lgbrankprize_{from_date}_{to_date}.pickle',
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

# Layer1: LightGBM LambdaRank Score
l1_lgb_rank_score = Dict(
    train=dataset_query.layer1_train,
    valid=dataset_query.layer1_valid,
    file=f'lgbrankscore_{from_date}_{to_date}.pickle',
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

# Layer1: LightGBM Regression
l1_lgb_regression = Dict(
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

# Layer1: RandomForest Regression
l1_rf_regression = Dict(
    train=dataset_query.layer1_train,
    valid=dataset_query.layer1_valid,
    file=f'rfregprize_{from_date}_{to_date}.pickle',
    target='prizeper',
    params={
        'n_estimators': 100,
        'verbose': 2,
        'n_jobs': -1,
    }
)

# Layer1: SpportVectorRegression
l1_svr_regression = Dict(
    train=dataset_query.layer1_train,
    valid=dataset_query.layer1_valid,
    file=f'svrprize_{from_date}_{to_date}.pickle',
    target='prizeper',
    params={
        'kernel': 'rbf',
        'gamma': 1,
        'C': 1,
        'epsilon': 0.1,
        'cache_size': 1024,
        'verbose': True,
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
        'verbose': True,
    }
)

# Layer1: KNeighbors Regression
l1_kn_regression = Dict(
    train=dataset_query.layer1_train,
    valid=dataset_query.layer1_valid,
    file=f'knprize_{from_date}_{to_date}.pickle',
    target='prizeper',
    params={
        'n_neighbors': 5,
        'n_jobs': -1,
    }
)

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

# 予測確率を予測するモデル
prob_valid = "'2023-01-01' <= race_date <= '2024-12-31'"
prob_model = Dict(
    file=f'prob_{from_date}_{to_date}.pickle',
    target='result',
    params={
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'lambdarank_truncation_level': 10,
        'ndcg_eval_at': [1, 3, 5],
        'boosting_type': 'gbdt',
        'learning_rate': 0.01,
    },
)
