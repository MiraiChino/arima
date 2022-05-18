# scraping
from_date = "2008-01"
to_date = "2022-05"
netkeiba_db = "netkeiba20220508.sqlite"

# feature extraction
encoder_file = "encoder20220508.pickle"
feat_db = "feature20220508.sqlite"

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
        "name": {
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
        },
        "jockey": {
            "jockey_interval": interval,
            "jockey_place": same_count("place_code"),
            "jockey_odds": ave("odds"),
            "jockey_pop": ave("pop"),
            "jockey_result": ave("result"),
            "jockey_horse": same_count("name"),
            "jockey_penalty": ave("penalty"),
            "jockey_distance": diff("distance"),
            "jockey_weather": same_count("weather"),
            "jockey_fc": same_count("field_condition"),
            "jockey_time": time(ave_time),
            "jockey_margin": ave("margin"),
            "jockey_corner3": ave("corner3"),
            "jockey_corner4": ave("corner4"),
            "jockey_last3f": ave("last3f"),
            "jockey_weight": ave("weight"),
            "jockey_wc": ave("weight_change"),
            "jockey_prize": ave("prize"),
        },
        "barn": {
            "barn_interval": interval,
            "barn_place": same_count("place_code"),
            "barn_odds": ave("odds"),
            "barn_pop": ave("pop"),
            "barn_result": ave("result"),
            "barn_horse": same_count("name"),
            "barn_penalty": ave("penalty"),
            "barn_distance": diff("distance"),
            "barn_weather": same_count("weather"),
            "barn_fc": same_count("field_condition"),
            "barn_time": time(ave_time),
            "barn_margin": ave("margin"),
            "barn_corner3": ave("corner3"),
            "barn_corner4": ave("corner4"),
            "barn_last3f": ave("last3f"),
            "barn_weight": ave("weight"),
            "barn_wc": ave("weight_change"),
            "barn_prize": ave("prize"),
        },
    }

# train
splits = [
    dict(
        train="'2008-01-01' <= race_date <= '2012-12-31'",
        valid="'2013-01-01' <= race_date <= '2015-12-31'"
    ),
    dict(
        train="'2008-01-01' <= race_date <= '2015-12-31'",
        valid="'2016-01-01' <= race_date <= '2018-12-31'"
    ),
    dict(
        train="'2008-01-01' <= race_date <= '2018-12-31'",
        valid="'2019-01-01' <= race_date <= '2022-05-08'"
    ),
]

rank_file = "rank_model20220508.pickle"
rank_target = "score"
rank_params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "lambdarank_truncation_level": 10,
    "ndcg_eval_at": [1, 3, 5],
    "boosting_type": "gbdt",
    "learning_rate": 0.01,
}

reg_file = "reg_model20220508.pickle"
reg_target = "prize"
reg_params = {
    'objective': 'regression',
    'boosting_type': 'gbdt',
    'metric': 'rmse',
    'learning_rate': 0.01
}