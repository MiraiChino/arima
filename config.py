# scraping
from_date = "2008-01"
to_date = "2022-01"
netkeiba_db = "netkeiba.sqlite"

# feature extraction
encoder_file = "encoder.pickle"
params_file = "params.pickle"
feat_db = "feature.sqlite"

# train
train_query = "'2008-01-01' <= race_date <= '2017-12-31'"
valid_query = "'2018-01-01' <= race_date <= '2020-12-31'"
test_query = "'2021-01-01' <= race_date <= '2021-12-31'"

rank_file = "rank_model.pickle"
rank_target = "score"
rank_params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "lambdarank_truncation_level": 10,
    "ndcg_eval_at": [3, 5],
    "boosting_type": "gbdt",
    "learning_rate": 0.01,
}

reg_file = "reg_model.pickle"
reg_target = "prize"
reg_params = {
    'objective': 'regression',
    'boosting_type': 'gbdt',
    'metric': 'rmse',
    'learning_rate': 0.01
}