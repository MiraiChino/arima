from dataclasses import dataclass


NONEED_COLUMNS = [
    "index", "result", "time", "margin", "pop", "odds", "last3f", \
    "weight", "weight_change", "corner", "corner3", "corner4", \
    "year", "hold_num", "race_num", "day_num", "race_date", "race_name", \
    "start_time", "prize1", "prize2", "prize3", "prize4", "prize5", "prize", "score"
]
RACE_CULMNS = ["year", "place_code", "hold_num", "day_num", "race_num"]

@dataclass
class Params:
    ave_time: dict
    hist_pattern: list
    feat_pattern: dict
