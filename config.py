from addict import Dict

# scraping
from_date = '2008-01'
to_date = '2022-12'
netkeiba_horse_file = f'temp/netkeiba{from_date}_{to_date}.horses.feather'
netkeiba_race_file = f'temp/netkeiba{from_date}_{to_date}.races.feather'
netkeiba_file = f'temp/netkeiba{from_date}_{to_date}.feather'

# feature extraction
encoder_file = f'encoder{from_date}_{to_date}.pickle'
avetime_file = f'avetime{from_date}_{to_date}.feather'
rating_file = f'temp/rating{from_date}_{to_date}.feather'
feat_file = f'feature{from_date}_{to_date}.feather'

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

# train
splits = [
    Dict(
        train="'2008-01-01' <= race_date <= '2008-12-31'",
        valid="'2009-01-01' <= race_date <= '2009-12-31'",
    ),
    Dict(
        train="'2009-01-01' <= race_date <= '2009-12-31'",
        valid="'2010-01-01' <= race_date <= '2010-12-31'",
    ),
    Dict(
        train="'2010-01-01' <= race_date <= '2010-12-31'",
        valid="'2011-01-01' <= race_date <= '2011-12-31'",
    ),
    Dict(
        train="'2011-01-01' <= race_date <= '2011-12-31'",
        valid="'2012-01-01' <= race_date <= '2012-12-31'",
    ),
    Dict(
        train="'2012-01-01' <= race_date <= '2012-12-31'",
        valid="'2013-01-01' <= race_date <= '2013-12-31'",
    ),
    Dict(
        train="'2013-01-01' <= race_date <= '2013-12-31'",
        valid="'2014-01-01' <= race_date <= '2014-12-31'",
    ),
    Dict(
        train="'2014-01-01' <= race_date <= '2014-12-31'",
        valid="'2015-01-01' <= race_date <= '2015-12-31'",
    ),
    Dict(
        train="'2015-01-01' <= race_date <= '2015-12-31'",
        valid="'2016-01-01' <= race_date <= '2016-12-31'",
    ),
    Dict(
        train="'2016-01-01' <= race_date <= '2016-12-31'",
        valid="'2017-01-01' <= race_date <= '2017-12-31'",
    ),
    Dict(
        train="'2017-01-01' <= race_date <= '2017-12-31'",
        valid="'2018-01-01' <= race_date <= '2018-12-31'",
    ),
    Dict(
        train="'2018-01-01' <= race_date <= '2018-12-31'",
        valid="'2019-01-01' <= race_date <= '2019-12-31'",
    ),
]

lgb_models = [
    Dict(
        file=f'lgbrankscore_model{from_date}_{to_date}.pickle',
        target='score',
        params={
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'lambdarank_truncation_level': 10,
            'ndcg_eval_at': [1, 3, 5],
            'boosting_type': 'gbdt',
            'learning_rate': 0.01,
        }
    ),
    Dict(
        file=f'lgbrankprize_model{from_date}_{to_date}.pickle',
        target='prizeper',
        params={
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'lambdarank_truncation_level': 10,
            'ndcg_eval_at': [1, 3, 5],
            'boosting_type': 'gbdt',
            'learning_rate': 0.01,
        }
    ),
    Dict(
        file=f'lgbregprize_model{from_date}_{to_date}.pickle',
        target='prize',
        params={
            'objective': 'regression',
            'boosting_type': 'gbdt',
            'metric': 'rmse',
            'learning_rate': 0.01
        }
    ),
]

# ['gate', 'horse_no', 'name', 'horse_id', 'sex', 'age', 'penalty', 'jockey', 'jockey_id',
# 'trainer', 'trainer_id', 'race_id', 'field', 'distance', 'turn', 'weather', 'field_condition',
# 'race_condition', 'place_code', 'cos_racedate', 'cos_starttime', 'last3frel', 'penaltyrel',
# 'weightrel', 'penaltywgt', 'oddsrslt', ...]
cat_features = [2, 3, 4, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18] # category変数をインデックス番号で指定
cat_models = [
    Dict(
        file=f'catrankprize_model{from_date}_{to_date}.pickle',
        target='prize',
        param={
            'loss_function':'YetiRank',
            'num_boost_round': 10000,
            'learning_rate': 0.01,
            'depth': 8,
            'custom_metric': ['NDCG:top=1', 'NDCG:top=3', 'NDCG:top=5'],
            'eval_metric': 'NDCG:top=5',
            'use_best_model': True,
            'early_stopping_rounds': 300,
            'has_time': True,
        }
    ),
]

stacking_valid = "'2020-01-01' <= race_date <= '2022-12-31'"
stacking_model = Dict(
    file=f'stacking_model{from_date}_{to_date}.pickle',
    target='score',
    params={
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'lambdarank_truncation_level': 10,
        'ndcg_eval_at': [1, 3, 5],
        'boosting_type': 'gbdt',
        'learning_rate': 0.01,
    },
)