import argparse
import itertools
import math
import sqlite3
from dataclasses import dataclass, field

import dill as pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

import chrome
import config
import feature_extractor
import netkeiba
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raceid", dest="race_id", required=True, type=str,
                        help="Example 202206010111.")
    return parser.parse_args()

@dataclass
class Baken:
    nums: list = field(default_factory=list)
    prob: dict = field(default_factory=dict)
    odds: dict = field(default_factory=dict)
    df: pd.DataFrame = pd.DataFrame()
    df2: pd.DataFrame = pd.DataFrame()

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    return exp_x / np.sum(exp_x)

def probability(x):
    return softmax(scale(x))

def p1(no1_index, probs):
    return probs[no1_index]

def p12(no1_index, no2_index, probs, g=0.81):
    p2 = probs[no2_index]**g / sum(x**g for x in probs.values())
    return p1(no1_index, probs), p2, 

def p123(no1_index, no2_index, no3_index, probs, d=0.65):
    p3 = probs[no3_index]**d / sum(x**d for x in probs.values())
    return *p12(no1_index, no2_index, probs), p3

def p_sanrentan(a, b, c):
    return a * b/(1-a) * c/(1-a-b)

def p_sanrenpuku(a, b, c):
    return p_sanrentan(a,b,c) + p_sanrentan(a,c,b) + p_sanrentan(b,a,c) \
            + p_sanrentan(b,c,a) + p_sanrentan(c,a,b) + p_sanrentan(c,b,a)

def p_umatan(a, b):
    return a * b/(1-a)

def p_umaren(a, b):
    return p_umatan(a,b) + p_umatan(b,a)

def p_wide(index1, index2, sanrentan_probs):
    wide = 0
    for (no1, no2, no3), prob in sanrentan_probs.items():
        if (index1 == no1 and (index2 == no2 or index2 == no3)) or \
            (index1 == no2 and (index2 == no1 or index2 == no3)) or \
            (index1 == no3 and (index2 == no1 or index2 == no2)):
            wide += prob
    return wide

def p_hukushou(index1, sanrentan_probs):
    hukushou = 0
    for (no1, no2, no3), prob in sanrentan_probs.items():
        if index1 == no1 or index1 == no2 or index1 == no3:
            hukushou += prob
    return hukushou

def synthetic_odds(odds):
    return 1 / sum(1/o for o in odds)

def cumulative_odds(odds):
    return [synthetic_odds(odds[:i]) for i in range(1, len(odds)+1)]

def cumulative_prob(prob):
    return [sum(prob[:i]) for i in range(1, len(prob)+1)]

def tuples1_in_tuples2(tuples1, tuples2):
    if not tuples1:
        return False
    for t1 in tuples1:
        if t1 not in tuples2:
            return False
    return True

def bet_equally(odds, expected_return):
    return [math.ceil(expected_return/odd/100)*100 for odd in odds]

def min_bet(odds, max_return=100000):
    for expected_return in range(100, max_return+100, 100):
        bets = bet_equally(odds, expected_return)
        invest = sum(bets)
        if invest < expected_return:
            return bets
    return []

def result_prob(df, task_logs=[]):
    task_logs.append(f"loading {config.encoder_file}")
    with open(config.encoder_file, "rb") as f:
        netkeiba_encoder = pickle.load(f)
    df_format = netkeiba_encoder.format(df)
    df_encoded = netkeiba_encoder.transform(df_format)
    df_feat = pd.DataFrame()
    task_logs.append(f"connecting {config.feat_db}")
    with sqlite3.connect(config.feat_db) as conn:
        df_avetime = pd.read_sql_query(f"SELECT * FROM ave_time", conn)
        ave_time = {(f, d, fc): t for f, d, fc, t in df_avetime.to_dict(orient="split")["data"]}
        hist_pattern = config.hist_pattern
        feat_pattern = config.feature_pattern(ave_time)
        race_date = df_encoded["race_date"][0]
        condition = " OR ".join(f"{column}=={row[column]}" for _, row in df_encoded.iterrows() for column in feat_pattern.keys())
        c_size = 10000
        reader = pd.read_sql_query(f"SELECT * FROM horse WHERE race_date < '{race_date}' AND ({condition})", conn, chunksize=c_size)
        chunks = []
        total_loaded = 0
        for df_chunk in reader:
            df_chunk_reduced = utils.reduce_mem_usage(df_chunk, verbose=True)
            h, w = df_chunk_reduced.shape
            total_loaded += h
            task_logs.append(f"loaded {total_loaded} rows from {config.feat_db}")
            chunks.append(df_chunk_reduced)
        hist = pd.concat(chunks, ignore_index=True)
        hist["race_date"] = pd.to_datetime(hist["race_date"])
        hist = hist.sort_values("race_date")
        hist = hist.loc[:, :'score']
        for index, row in df_encoded.iterrows():
            df_agg = feature_extractor.search_history(row, hist_pattern, feat_pattern, hist)
            df_feat = pd.concat([df_feat, df_agg])
    df_feat = df_feat.drop(columns=config.NONEED_COLUMNS)

    probs = []
    model_files = [f"{i}_{file}" for file in (config.rank_file, config.reg_file) for i in range(len(config.splits))]
    for model_file in model_files:
        with open(model_file, "rb") as f:
            task_logs.append(f"predicting: with {model_file}")
            model = pickle.load(f)
            pred = model.predict(df_feat.values, num_iteration=model.best_iteration)
            pred_prob = probability(pred)
            probs.append(pred_prob)
    prob = np.array(probs).mean(axis=0)
    return {i: p for i, p in zip(df_feat["horse_no"].to_list(), prob)}

def baken_prob(prob, names):
    baken = {
        "??????": Baken(),
        "??????": Baken(),
        "?????????": Baken(),
        "??????": Baken(),
        "??????": Baken(),
        "?????????": Baken(),
        "?????????": Baken(),
    }
    baken["??????"].nums = list(prob.keys())
    baken["??????"].nums = list(prob.keys())
    baken["??????"].nums = list(itertools.permutations(baken["??????"].nums, 2))
    baken["??????"].nums = list(itertools.combinations(baken["??????"].nums, 2))
    baken["?????????"].nums = list(itertools.combinations(baken["??????"].nums, 2))
    baken["?????????"].nums = list(itertools.permutations(baken["??????"].nums, 3))
    baken["?????????"].nums = list(itertools.combinations(baken["??????"].nums, 3))
    baken["??????"].prob = {no1: p1(no1, prob) for no1 in baken["??????"].nums}
    baken["??????"].prob = {(no1, no2): p_umatan(*p12(no1, no2, prob)) for no1, no2 in baken["??????"].nums}
    baken["??????"].prob = {tuple(sorted([no1, no2])): p_umaren(*p12(no1, no2, prob)) for no1, no2 in baken["??????"].nums}
    baken["?????????"].prob = {(no1, no2, no3): p_sanrentan(*p123(no1, no2, no3, prob)) for no1, no2, no3 in baken["?????????"].nums}
    baken["?????????"].prob = {tuple(sorted([no1, no2, no3])): p_sanrenpuku(*p123(no1, no2, no3, prob)) for no1, no2, no3 in baken["?????????"].nums}
    baken["??????"].prob = {no1: p_hukushou(no1, baken["?????????"].prob) for no1 in baken["??????"].nums}
    baken["?????????"].prob = {(no1, no2): p_wide(no1, no2, baken["?????????"].prob) for no1, no2 in baken["?????????"].nums}
    for b_type, b in baken.items():
        high_probs = sorted(b.prob.items(), key=lambda x: x[1], reverse=True)
        baken[b_type].nums = [i for i, _ in high_probs]
        baken[b_type].prob = dict(high_probs)

    baken["??????"].df = pd.DataFrame({
        "??????": [str(no1) for no1 in baken["??????"].nums],
        "??????": [names[no] for no in baken["??????"].nums],
        "??????": [p1 for p1 in baken["??????"].prob.values()],
    })
    baken["??????"].df = pd.DataFrame({
        "??????": [str(no1) for no1 in baken["??????"].nums],
        "??????": [names[no] for no in baken["??????"].nums],
        "??????": [p1 for p1 in baken["??????"].prob.values()],
    })
    baken["??????"].df = pd.DataFrame({
        "??????": [str(no1) for no1, no2 in baken["??????"].nums],
        "??????": [str(no2) for no1, no2 in baken["??????"].nums],
        "??????": [p for p in baken["??????"].prob.values()],
    })
    baken["??????"].df = pd.DataFrame({
        "??????": [str(no1) for no1, no2 in baken["??????"].nums],
        "??????": [str(no2) for no1, no2 in baken["??????"].nums],
        "??????": [p for p in baken["??????"].prob.values()],
    })
    baken["?????????"].df = pd.DataFrame({
        "??????": [str(no1) for no1, no2 in baken["?????????"].nums],
        "??????": [str(no2) for no1, no2 in baken["?????????"].nums],
        "??????": [p for p in baken["?????????"].prob.values()],
    })
    baken["?????????"].df = pd.DataFrame({
        "??????": [str(no1) for no1, no2, no3 in baken["?????????"].nums],
        "??????": [str(no2) for no1, no2, no3 in baken["?????????"].nums],
        "??????": [str(no3) for no1, no2, no3 in baken["?????????"].nums],
        "??????": [p for p in baken["?????????"].prob.values()],
    })
    baken["?????????"].df = pd.DataFrame({
        "??????": [str(no1) for no1, no2, no3 in baken["?????????"].nums],
        "??????": [str(no2) for no1, no2, no3 in baken["?????????"].nums],
        "??????": [str(no3) for no1, no2, no3 in baken["?????????"].nums],
        "??????": [p for p in baken["?????????"].prob.values()],
    })
    baken["?????????"].df.style.set_properties(subset=['text'], **{'width': '300px'})
    return baken

def calc_odds(baken, race_id, top=100, task_logs=[]):
    with chrome.driver() as driver:
        task_logs.append(f"scraping: https://race.netkeiba.com/odds/index.html?race_id={race_id}")
        baken["??????"].odds, baken["??????"].odds = netkeiba.scrape_tanhuku(driver, race_id)
        task_logs.append(f"scraping: https://race.netkeiba.com/odds/index.html?type=b6&race_id={race_id}")
        baken["??????"].odds = netkeiba.scrape_umatan(driver, race_id)
        task_logs.append(f"scraping: https://race.netkeiba.com/odds/index.html?type=b4&race_id={race_id}")
        baken["??????"].odds = netkeiba.scrape_umaren(driver, race_id)
        task_logs.append(f"scraping: https://race.netkeiba.com/odds/index.html?type=b5&race_id={race_id}")
        baken["?????????"].odds = netkeiba.scrape_wide(driver, race_id)
        sanrentan_odds_gen = netkeiba.scrape_sanrentan_generator(driver, race_id)
        sanrenpuku_odds_gen = netkeiba.scrape_sanrenpuku_generator(driver, race_id)
        task_logs.append(f"scraping: https://race.netkeiba.com/odds/index.html?type=b8&race_id={race_id}")
        while not tuples1_in_tuples2(baken["?????????"].nums[:top], list(baken["?????????"].odds.keys())):
            try:
                baken["?????????"].odds |= next(sanrentan_odds_gen)
            except StopIteration:
                pass
        task_logs.append(f"scraping: https://race.netkeiba.com/odds/index.html?type=b7&race_id={race_id}")
        while not tuples1_in_tuples2(baken["?????????"].nums[:top], list(baken["?????????"].odds.keys())):
            try:
                baken["?????????"].odds |= next(sanrenpuku_odds_gen)
            except StopIteration:
                pass

    for b_type, b in baken.items():
        b.df = b.df.head(top)
        probs = list(b.prob.values())[:top]
        b.nums = b.nums[:top]
        cum_probs = cumulative_prob(probs)
        b.df["??????"] = pd.Series([f"{p*100:.2f}%" for p in b.df["??????"].values])
        b.df["?????????"] = pd.Series([b.odds[nums] for nums in b.nums])
        b.df["?????????"] = b.df["?????????"] * pd.Series(probs)
        b.df["?????????"] = pd.Series([round(p, 2) for p in b.df["?????????"].values])
        b.df["????????????"] = pd.Series([f"{p*100:.2f}%" for p in cum_probs])
        b.df["???????????????"] = pd.Series([round(p, 2) for p in cumulative_odds(b.df["?????????"].values)])
        b.df["???????????????"] = b.df["???????????????"] * pd.Series(cum_probs)
        b.df["???????????????"] = pd.Series([round(p, 2) for p in b.df["???????????????"].values])
    return baken

def good_baken(baken, odd_th=2.0):
    for b_type, b in baken.items():
        index_synodd2 = len([odd for odd in b.df["???????????????"] if odd_th <= odd])
        top_odds = b.df["?????????"][:index_synodd2]
        bets = min_bet(top_odds)
        if bets:
            bets_str = [f"{bet}???" for bet in bets]
            returns = [int(round(odd*bet, -1)) for odd, bet in zip(top_odds, bets)]
            returns_str = [f"{ret}???" for ret in returns]
            invest = sum(bets)
            min_ret, max_ret = min(returns), max(returns)
            bets_str.append(f"???: {invest}???")
            returns_str.append(f"{min_ret}???~{max_ret}???")
        else:
            bets_str = ['']
            returns_str = ['']
            invest = 0
            returns = False
            min_ret, max_ret = 0, 0
        b.df2 = pd.DataFrame()
        b.df2["????????????"] = pd.Series(bets_str)
        b.df2["??????"] = pd.Series(returns_str)
        b.df.index += 1
    return baken

if __name__ == "__main__":
    args = parse_args()
    horses = [horse for horse in netkeiba.scrape_shutuba(args.race_id)]
    df_original = pd.DataFrame(horses, columns=netkeiba.COLUMNS)
    print(df_original)
    p = result_prob(df_original)
    print(p)
    names = {no: name for no, name in zip(df_original["horse_no"].to_list(), df_original["name"].to_list())}
    baken = baken_prob(p, names)
    baken = calc_odds(baken, args.race_id, top=100)
    baken = good_baken(baken, odds_th=2.0)
    for b_type, b in baken.items():
        print(b_type)
        print(b.df)
    import pdb; pdb.set_trace()
