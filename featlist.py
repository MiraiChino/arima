import numpy as np

def ave(column):
    def wrapper(history, now, index):
        try:
            return (history[:, index(column)]).mean()
        except:
            import pdb; pdb.set_trace()
    return wrapper

def interval(history, now, index):
    return (now[index("race_date")] - history[:, index("race_date")]).astype("timedelta64[D]").mean() / np.timedelta64(1, 'D')

def same_count(column):
    def wrapper(history, now, index):
        return (history[:, index(column)] == now[index(column)]).sum()
    return wrapper

def diff(col1, col2):
    def wrapper(history, now, index):
        return ((history[:, index(col1)] - history[:, index(col2)]) / history[:, index(col2)]).mean()
    return wrapper

def same_ave(*same_columns, target="prize"):
    def wrapper(history, now, index):
        i1 = index(target)
        same_conditions = [(history[:, index(c)] == now[index(c)]) for c in same_columns]
        same_hist = history[np.logical_and.reduce(same_conditions)]
        if same_hist.any():
            prizes = same_hist[:, i1]
            return prizes.mean()
        else:
            return 0
    return wrapper

def interval_prize(history, now, index):
    interval = (now[index("race_date")] - history[:, index("race_date")]).astype("timedelta64[D]").mean() / np.timedelta64(1, 'D')
    prize = history[:, index("prize")].mean()
    return prize / interval

def drize(distance, prize, now_distance):
    return (1 - abs(distance - now_distance) / now_distance) * prize

def distance_prize():
    def wrapper(history, now, index):
        i1 = index("distance")
        distances = history[:, i1]
        prizes = history[:, index("prize")]
        dprizes = np.vectorize(drize)(distances, prizes, now[i1])
        return np.mean(dprizes)
    return wrapper

def same_drize(*same_columns):
    def wrapper(history, now, index):
        i1 = index("distance")
        same_conditions = [(history[:, index(c)] == now[index(c)]) for c in same_columns]
        same_hist = history[np.logical_and.reduce(same_conditions)]
        if same_hist.any():
            distances = same_hist[:, i1]
            prizes = same_hist[:, index("prize")]
            dprizes = np.vectorize(drize)(distances, prizes, now[i1])
            return np.mean(dprizes)
        else:
            return 0
    return wrapper

hist_pattern = [1, 2, 3, 6, 12, 999999] # months

feature_pattern = {
    'horse_id': {
        'by_race': {
            'horse_interval': interval,
        },
        'by_month': {
            'horse_result': ave('result'),
            'horse_odds': ave('odds'),
            'horse_pop': ave('pop'),
            'horse_penalty': ave('penalty'),
            'horse_weather':same_count('weather'),
            'horse_time': diff('time', 'avetime'),
            'horse_margin': ave('margin'),
            'horse_corner3': ave('corner3'),
            'horse_corner4': ave('corner4'),
            'horse_last3f': ave('last3f'),
            'horse_weight': ave('weight'),
            'horse_wc': ave('weight_change'),
            'horse_last3frel': ave('last3frel'),
            'horse_penaltyrel': ave('penaltyrel'),
            'horse_weightrel': ave('weightrel'),
            'horse_penaltywgt': ave('penaltywgt'),
            'horse_oddsrslt': ave('oddsrslt'),
            'horse_r': ave('horse_oldr'),
            'horse_tan': ave('tanshou'),
            'horse_huku': ave('hukushou'),
            'horse_score': ave('score'),
            'horse_prize': ave('prize'),
            'horse_iprize': interval_prize,
            'horse_pprize': same_ave('place_code', target='prize'),
            'horse_dprize': same_ave('distance', target='prize'),
            'horse_fprize': same_ave('field', target='prize'),
            'horse_cprize': same_ave('field_condition', target='prize'),
            'horse_tprize': same_ave('turn', target='prize'),
            'horse_jprize': same_ave('jockey', target='prize'),
            'horse_ftprize': same_ave('field', 'turn', target='prize'),
            'horse_fdprize': same_ave('field', 'distance', target='prize'),
            'horse_fcprize': same_ave('field', 'field_condition', target='prize'),
            'horse_pfprize': same_ave('place_code', 'field', target='prize'),
            'horse_pdprize': same_ave('place_code', 'distance', target='prize'),
            'horse_pftprize': same_ave('place_code', 'field', 'turn', target='prize'),
            'horse_pfdprize': same_ave('place_code', 'field', 'distance', target='prize'),
            'horse_pfcprize': same_ave('place_code', 'field', 'field_condition', target='prize'),
            'horse_dfcprize': same_ave('distance', 'field', 'field_condition', target='prize'),
            'horse_pfdcprize': same_ave('place_code', 'field', 'distance', 'field_condition', target='prize'),
            'horse_pfdtprize': same_ave('place_code', 'field', 'distance', 'turn', target='prize'),
            'horse_drize': distance_prize(),
            'horse_pdrize': same_drize('place_code'),
            'horse_fdrize': same_drize('field'),
            'horse_cdrize': same_drize('field_condition'),
            'horse_tdrize': same_drize('turn'),
            'horse_jdrize': same_drize('jockey'),
            'horse_ftdrize': same_drize('field', 'turn'),
            'horse_fcdrize': same_drize('field', 'field_condition'),
            'horse_pfdrize': same_drize('place_code', 'field'),
            'horse_pftdrize': same_drize('place_code', 'field', 'turn'),
            'horse_pfcdrize': same_drize('place_code', 'field', 'field_condition'),
            'horse_pfctdrize': same_drize('place_code', 'field', 'field_condition', 'turn'),
        },
    },
    'jockey_id': {
        'by_race': {
            'jockey_interval': interval,
        },
        'by_month': {
            "jockey_odds": ave("odds"),
            "jockey_pop": ave("pop"),
            "jockey_result": ave("result"),
            "jockey_weather": same_count("weather"),
            "jockey_time": diff('time', 'avetime'),
            "jockey_margin": ave("margin"),
            "jockey_corner3": ave("corner3"),
            "jockey_corner4": ave("corner4"),
            "jockey_last3f": ave("last3f"),
            "jockey_last3frel": ave("last3frel"),
            "jockey_penaltyrel": ave("penaltyrel"),
            "jockey_penaltywgt": ave("penaltywgt"),
            "jockey_oddsrslt": ave("oddsrslt"),
            "jockey_r": ave("jockey_oldr"),
            "jockey_tan": ave("tanshou"),
            "jockey_huku": ave("hukushou"),
            "jockey_score": ave("score"),
            "jockey_prize": ave("prize"),
            "jockey_iprize": interval_prize,
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
    },
    'trainer_id': {
        'by_race': {
            'trainer_interval': interval,
        },
        'by_month': {
            "trainer_odds": ave("odds"),
            "trainer_pop": ave("pop"),
            "trainer_result": ave("result"),
            "trainer_time": diff('time', 'avetime'),
            "trainer_margin": ave("margin"),
            "trainer_corner3": ave("corner3"),
            "trainer_corner4": ave("corner4"),
            "trainer_last3f": ave("last3f"),
            "trainer_last3frel": ave("last3frel"),
            "trainer_penaltyrel": ave("penaltyrel"),
            "trainer_weightrel": ave("weightrel"),
            "trainer_penaltywgt": ave("penaltywgt"),
            "trainer_oddsrslt": ave("oddsrslt"),
            "trainer_r": ave("trainer_oldr"),
            "trainer_tan": ave("tanshou"),
            "trainer_huku": ave("hukushou"),
            "trainer_score": ave("score"),
            "trainer_prize": ave("prize"),
            "trainer_iprize": interval_prize,
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
    },
}