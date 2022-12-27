# scraping
from_date = "2008-01"
to_date = "2022-12"
netkeiba_horse_file = f"netkeiba/netkeiba{from_date}_{to_date}.horses.feather"
netkeiba_race_file = f"netkeiba/netkeiba{from_date}_{to_date}.races.feather"

# feature extraction
encoder_file = f"encoder{from_date}_{to_date}.pickle"
avetime_file = f"avetime{from_date}_{to_date}.feather"
feat_file = f"feature{from_date}_{to_date}.feather"

NONEED_COLUMNS = [
    "id", "index", "result", "time", "margin", "pop", "odds", "last3f", \
    "weight", "weight_change", "corner", "corner3", "corner4", \
    "year", "hold_num", "race_num", "day_num", "race_date", "race_name", \
    "start_time", "prize1", "prize2", "prize3", "prize4", "prize5", "prize", "score", "tanshou", "hukushou"\
    "tanno1", "tanno2", "hukuno1", "hukuno2", "hukuno3", "tan1", "tan2", \
    "huku1", "huku2", "huku3", "wide1", "wide2", "wide3", "ren", \
    "uma1", "uma2", "puku", "san1", "san2"
]
RACEDATE_COLUMNS = ["year", "place_code", "hold_num", "day_num", "race_num"]

hist_pattern = [1, 2, 3, 5, 999999]

def feature_pattern(ave_time):
    from feature_extractor import (ave, distance_prize, interval, same_ave,
                                   same_count, same_drize, time)
    return {
        "name": {
            "horse_interval": interval,
            "horse_odds": ave("odds"),
            "horse_pop": ave("pop"),
            "horse_result": ave("result"),
            "horse_penalty": ave("penalty"),
            "horse_weather": same_count("weather"),
            "horse_time": time(ave_time),
            "horse_margin": ave("margin"),
            "horse_corner3": ave("corner3"),
            "horse_corner4": ave("corner4"),
            "horse_last3f": ave("last3f"),
            "horse_weight": ave("weight"),
            "horse_wc": ave("weight_change"),
            "horse_tan": ave("tanshou"),
            "horse_huku": ave("hukushou"),
            "horse_score": ave("score"),
            "horse_prize": ave("prize"),
            "horse_pprize": same_ave("place_code", target="prize"),
            "horse_dprize": same_ave("distance", target="prize"),
            "horse_fprize": same_ave("field", target="prize"),
            "horse_cprize": same_ave("field_condition", target="prize"),
            "horse_tprize": same_ave("turn", target="prize"),
            "horse_jprize": same_ave("jockey", target="prize"),
            "horse_ftprize": same_ave("field", "turn", target="prize"),
            "horse_fdprize": same_ave("field", "distance", target="prize"),
            "horse_fcprize": same_ave("field", "field_condition", target="prize"),
            "horse_pfprize": same_ave("place_code", "field", target="prize"),
            "horse_pdprize": same_ave("place_code", "distance", target="prize"),
            "horse_pftprize": same_ave("place_code", "field", "turn", target="prize"),
            "horse_pfdprize": same_ave("place_code", "field", "distance", target="prize"),
            "horse_pfcprize": same_ave("place_code", "field", "field_condition", target="prize"),
            "horse_dfcprize": same_ave("distance", "field", "field_condition", target="prize"),
            "horse_pfdcprize": same_ave("place_code", "field", "distance", "field_condition", target="prize"),
            "horse_pfdtprize": same_ave("place_code", "field", "distance", "turn", target="prize"),
            "horse_drize": distance_prize(),
            "horse_pdrize": same_drize("place_code"),
            "horse_fdrize": same_drize("field"),
            "horse_cdrize": same_drize("field_condition"),
            "horse_tdrize": same_drize("turn"),
            "horse_jdrize": same_drize("jockey"),
            "horse_ftdrize": same_drize("field", "turn"),
            "horse_fcdrize": same_drize("field", "field_condition"),
            "horse_pfdrize": same_drize("place_code", "field"),
            "horse_pftdrize": same_drize("place_code", "field", "turn"),
            "horse_pfcdrize": same_drize("place_code", "field", "field_condition"),
            "horse_pfctdrize": same_drize("place_code", "field", "field_condition", "turn"),
        },
        "jockey": {
            "jockey_interval": interval,
            "jockey_odds": ave("odds"),
            "jockey_pop": ave("pop"),
            "jockey_result": ave("result"),
            "jockey_weather": same_count("weather"),
            "jockey_time": time(ave_time),
            "jockey_margin": ave("margin"),
            "jockey_corner3": ave("corner3"),
            "jockey_corner4": ave("corner4"),
            "jockey_last3f": ave("last3f"),
            "jockey_tan": ave("tanshou"),
            "jockey_huku": ave("hukushou"),
            "jockey_score": ave("score"),
            "jockey_prize": ave("prize"),
            "jockey_pprize": same_ave("place_code", target="prize"),
            "jockey_dprize": same_ave("distance", target="prize"),
            "jockey_fprize": same_ave("field", target="prize"),
            "jockey_cprize": same_ave("field_condition", target="prize"),
            "jockey_tprize": same_ave("turn", target="prize"),
            "jockey_ftprize": same_ave("field", "turn", target="prize"),
            "jockey_fdprize": same_ave("field", "distance", target="prize"),
            "jockey_fcprize": same_ave("field", "field_condition", target="prize"),
            "jockey_pfprize": same_ave("place_code", "field", target="prize"),
            "jockey_pdprize": same_ave("place_code", "distance", target="prize"),
            "jockey_pftprize": same_ave("place_code", "field", "turn", target="prize"),
            "jockey_pfdprize": same_ave("place_code", "field", "distance", target="prize"),
            "jockey_pfcprize": same_ave("place_code", "field", "field_condition", target="prize"),
            "jockey_dfcprize": same_ave("distance", "field", "field_condition", target="prize"),
            "jockey_pfdcprize": same_ave("place_code", "field", "distance", "field_condition", target="prize"),
            "jockey_pfdtprize": same_ave("place_code", "field", "distance", "turn", target="prize"),
        },
        "trainer": {
            "trainer_interval": interval,
            "trainer_odds": ave("odds"),
            "trainer_pop": ave("pop"),
            "trainer_result": ave("result"),
            "trainer_time": time(ave_time),
            "trainer_margin": ave("margin"),
            "trainer_corner3": ave("corner3"),
            "trainer_corner4": ave("corner4"),
            "trainer_last3f": ave("last3f"),
            "trainer_tan": ave("tanshou"),
            "trainer_huku": ave("hukushou"),
            "trainer_score": ave("score"),
            "trainer_prize": ave("prize"),
            "trainer_pprize": same_ave("place_code", target="prize"),
            "trainer_dprize": same_ave("distance", target="prize"),
            "trainer_fprize": same_ave("field", target="prize"),
            "trainer_cprize": same_ave("field_condition", target="prize"),
            "trainer_fdprize": same_ave("field", "distance", target="prize"),
            "trainer_fcprize": same_ave("field", "field_condition", target="prize"),
            "trainer_pfprize": same_ave("place_code", "field", target="prize"),
            "trainer_pdprize": same_ave("place_code", "distance", target="prize"),
            "trainer_pfdprize": same_ave("place_code", "field", "distance", target="prize"),
            "trainer_pfcprize": same_ave("place_code", "field", "field_condition", target="prize"),
            "trainer_dfcprize": same_ave("distance", "field", "field_condition", target="prize"),
            "trainer_pfdcprize": same_ave("place_code", "field", "distance", "field_condition", target="prize"),
        },
    }

# train
splits = [
    dict(
        train="'2008-01-01' <= race_date <= '2011-12-31'",
        valid="'2012-01-01' <= race_date <= '2013-12-31'"
    ),
    dict(
        train="'2008-01-01' <= race_date <= '2013-12-31'",
        valid="'2014-01-01' <= race_date <= '2016-12-31'"
    ),
    dict(
        train="'2008-01-01' <= race_date <= '2016-12-31'",
        valid="'2017-01-01' <= race_date <= '2019-12-31'"
    ),
    dict(
        train="'2008-01-01' <= race_date <= '2019-12-31'",
        valid="'2020-01-01' <= race_date <= '2022-12-31'"
    ),
]

rank_file = f"rank_model{from_date}_{to_date}.pickle"
rank_target = "score"
rank_params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "lambdarank_truncation_level": 10,
    "ndcg_eval_at": [1, 3, 5],
    "boosting_type": "gbdt",
    "learning_rate": 0.01,
}

reg_file = f"reg_model{from_date}_{to_date}.pickle"
reg_target = "prize"
reg_params = {
    'objective': 'regression',
    'boosting_type': 'gbdt',
    'metric': 'rmse',
    'learning_rate': 0.01
}