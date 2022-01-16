# scraping
from_date = "2008-01"
to_date = "2022-01"
netkeiba_db = "netkeiba.sqlite"

# feature extraction
encoder_file = "encoder.pickle"
feat_db = "feature.sqlite"

NONEED_COLUMNS = [
    "id", "index", "result", "time", "margin", "pop", "odds", "last3f", \
    "weight", "weight_change", "corner", "corner3", "corner4", \
    "year", "hold_num", "race_num", "day_num", "race_date", "race_name", \
    "start_time", "prize1", "prize2", "prize3", "prize4", "prize5", "prize", "score"
]
RACE_COLUMNS = ["year", "place_code", "hold_num", "day_num", "race_num"]

hist_pattern = [1, 2, 3, 4, 5, 10, 999999]

def feature_pattern(ave_time):
    from feature_extractor import ave, diff, interval, same_count, time
    return {
        "horse_interval": interval,
        "horse_place": same_count("place_code"),
        "horse_odds": ave("odds"),
        "horse_pop": ave("pop"),
        "horse_result": ave("result"),
        "horse_jockey": same_count("jockey"),
        "horse_penalty": ave("penalty"),
        "horse_distance": diff("distance"),
        "horse_weather": same_count("weather"),
        "horse_fc": same_count("field_condition"),
        "horse_time": time(ave_time),
        "horse_margin": ave("margin"),
        "horse_corner3": ave("corner3"),
        "horse_corner4": ave("corner4"),
        "horse_last3f": ave("last3f"),
        "horse_weight": ave("weight"),
        "horse_wc": ave("weight_change"),
        "horse_prize": ave("prize"),
    }

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
    "ndcg_eval_at": [1, 3, 5],
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