import argparse
import itertools
import math
from dataclasses import dataclass, field

import dill as pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

import chrome
import config
import feature_extractor
import netkeiba


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
    task_logs.append(f"encoding")
    df_format = netkeiba_encoder.format(df)
    df_encoded = netkeiba_encoder.transform(df_format)
    race_date = df_encoded["race_date"][0]

    task_logs.append(f"loading {config.avetime_file}")
    df_avetime = pd.read_feather(config.avetime_file)
    ave_time = {(f, d, fc): t for f, d, fc, t in df_avetime.to_dict(orient="split")["data"]}
    hist_pattern = config.hist_pattern
    feat_pattern = config.feature_pattern(ave_time)

    task_logs.append(f"loading {config.feat_file}")
    history = pd.read_feather(config.feat_file)
    task_logs.append(f"searching race history")

    condition = " or ".join(f"{column}=={df_encoded[column].tolist()}" for column in feat_pattern.keys())
    hist = history.query(f"race_date < '{race_date}' and ({condition})")
    hist["race_date"] = pd.to_datetime(hist["race_date"])
    hist = hist.sort_values("race_date")
    hist = hist.loc[:, :'score']
    df_feat = pd.DataFrame()
    for index, row in df_encoded.iterrows():
        df_agg = feature_extractor.search_history(row, hist_pattern, feat_pattern, hist)
        df_feat = pd.concat([df_feat, df_agg])
    noneed_columns = [c for c in config.NONEED_COLUMNS if c not in netkeiba.RACE_PAY_COLUMNS]
    df_feat = df_feat.drop(columns=noneed_columns)
    probs = []
    model_files = [f"{i}_{file}" for file in (config.rank_file, config.reg_file) for i in range(len(config.splits))]
    task_logs.append(f"predict")
    for model_file in model_files:
        with open(model_file, "rb") as f:
            model = pickle.load(f)
            pred = model.predict(df_feat.values, num_iteration=model.best_iteration)
            pred_prob = probability(pred)
            probs.append(pred_prob)
            task_logs.append(f"{model_file}:")
            task_logs.append(f"{[f'{p*100:.2f}%' for p in pred_prob]}")
    prob = np.array(probs).mean(axis=0)
    return {i: p for i, p in zip(df_feat["horse_no"].to_list(), prob)}

def baken_prob(prob, names):
    baken = {
        "単勝": Baken(),
        "複勝": Baken(),
        "ワイド": Baken(),
        "馬連": Baken(),
        "馬単": Baken(),
        "三連複": Baken(),
        "三連単": Baken(),
    }
    baken["単勝"].nums = list(prob.keys())
    baken["複勝"].nums = list(prob.keys())
    baken["馬単"].nums = list(itertools.permutations(baken["単勝"].nums, 2))
    baken["馬連"].nums = list(itertools.combinations(baken["単勝"].nums, 2))
    baken["ワイド"].nums = list(itertools.combinations(baken["単勝"].nums, 2))
    baken["三連単"].nums = list(itertools.permutations(baken["単勝"].nums, 3))
    baken["三連複"].nums = list(itertools.combinations(baken["単勝"].nums, 3))
    baken["単勝"].prob = {no1: p1(no1, prob) for no1 in baken["単勝"].nums}
    baken["馬単"].prob = {(no1, no2): p_umatan(*p12(no1, no2, prob)) for no1, no2 in baken["馬単"].nums}
    baken["馬連"].prob = {tuple(sorted([no1, no2])): p_umaren(*p12(no1, no2, prob)) for no1, no2 in baken["馬連"].nums}
    baken["三連単"].prob = {(no1, no2, no3): p_sanrentan(*p123(no1, no2, no3, prob)) for no1, no2, no3 in baken["三連単"].nums}
    baken["三連複"].prob = {tuple(sorted([no1, no2, no3])): p_sanrenpuku(*p123(no1, no2, no3, prob)) for no1, no2, no3 in baken["三連複"].nums}
    baken["複勝"].prob = {no1: p_hukushou(no1, baken["三連単"].prob) for no1 in baken["複勝"].nums}
    baken["ワイド"].prob = {(no1, no2): p_wide(no1, no2, baken["三連単"].prob) for no1, no2 in baken["ワイド"].nums}
    for b_type, b in baken.items():
        high_probs = sorted(b.prob.items(), key=lambda x: x[1], reverse=True)
        baken[b_type].nums = [i for i, _ in high_probs]
        baken[b_type].prob = dict(high_probs)

    baken["単勝"].df = pd.DataFrame({
        "馬番": [str(no1) for no1 in baken["単勝"].nums],
        "馬名": [names[no] for no in baken["単勝"].nums],
        "確率": [p1 for p1 in baken["単勝"].prob.values()],
    })
    baken["複勝"].df = pd.DataFrame({
        "馬番": [str(no1) for no1 in baken["複勝"].nums],
        "馬名": [names[no] for no in baken["複勝"].nums],
        "確率": [p1 for p1 in baken["複勝"].prob.values()],
    })
    baken["馬単"].df = pd.DataFrame({
        "一位": [str(no1) for no1, no2 in baken["馬単"].nums],
        "二位": [str(no2) for no1, no2 in baken["馬単"].nums],
        "確率": [p for p in baken["馬単"].prob.values()],
    })
    baken["馬連"].df = pd.DataFrame({
        "一位": [str(no1) for no1, no2 in baken["馬連"].nums],
        "二位": [str(no2) for no1, no2 in baken["馬連"].nums],
        "確率": [p for p in baken["馬連"].prob.values()],
    })
    baken["ワイド"].df = pd.DataFrame({
        "一位": [str(no1) for no1, no2 in baken["ワイド"].nums],
        "二位": [str(no2) for no1, no2 in baken["ワイド"].nums],
        "確率": [p for p in baken["ワイド"].prob.values()],
    })
    baken["三連単"].df = pd.DataFrame({
        "一位": [str(no1) for no1, no2, no3 in baken["三連単"].nums],
        "二位": [str(no2) for no1, no2, no3 in baken["三連単"].nums],
        "三位": [str(no3) for no1, no2, no3 in baken["三連単"].nums],
        "確率": [p for p in baken["三連単"].prob.values()],
    })
    baken["三連複"].df = pd.DataFrame({
        "一位": [str(no1) for no1, no2, no3 in baken["三連複"].nums],
        "二位": [str(no2) for no1, no2, no3 in baken["三連複"].nums],
        "三位": [str(no3) for no1, no2, no3 in baken["三連複"].nums],
        "確率": [p for p in baken["三連複"].prob.values()],
    })
    baken["三連複"].df.style.set_properties(subset=['text'], **{'width': '300px'})
    return baken

def pretty_prob(baken, top=100):
    for b_type, b in baken.items():
        b.df = b.df.head(top)
        probs = list(b.prob.values())[:top]
        b.nums = b.nums[:top]
        cum_probs = cumulative_prob(probs)
        b.df["確率"] = pd.Series([f"{p*100:.2f}%" for p in b.df["確率"].values])
        b.df["累積確率"] = pd.Series([f"{p*100:.2f}%" for p in cum_probs])
    return baken

def pretty_baken(baken, top=100):
    for b_type, b in baken.items():
        b.df = b.df.head(top)
        probs = list(b.prob.values())[:top]
        b.nums = b.nums[:top]
        cum_probs = cumulative_prob(probs)
        b.df["確率"] = pd.Series([f"{p*100:.2f}%" for p in b.df["確率"].values])
        b.df["オッズ"] = pd.Series([b.odds[nums] for nums in b.nums])
        b.df["期待値"] = b.df["オッズ"] * pd.Series(probs)
        b.df["期待値"] = pd.Series([round(p, 2) for p in b.df["期待値"].values])
        b.df["累積確率"] = pd.Series([f"{p*100:.2f}%" for p in cum_probs])
        b.df["合成オッズ"] = pd.Series([round(p, 2) for p in cumulative_odds(b.df["オッズ"].values)])
        b.df["合成期待値"] = b.df["合成オッズ"] * pd.Series(cum_probs)
        b.df["合成期待値"] = pd.Series([round(p, 2) for p in b.df["合成期待値"].values])
    return baken

def calc_odds(baken, race_id, top=100, task_logs=[]):
    with chrome.driver() as driver:
        task_logs.append(f"scraping: https://race.netkeiba.com/odds/index.html?race_id={race_id}")
        baken["単勝"].odds, baken["複勝"].odds = netkeiba.scrape_tanhuku(driver, race_id)
        task_logs.append(f"scraping: https://race.netkeiba.com/odds/index.html?type=b6&race_id={race_id}")
        baken["馬単"].odds = netkeiba.scrape_umatan(driver, race_id)
        task_logs.append(f"scraping: https://race.netkeiba.com/odds/index.html?type=b4&race_id={race_id}")
        baken["馬連"].odds = netkeiba.scrape_umaren(driver, race_id)
        task_logs.append(f"scraping: https://race.netkeiba.com/odds/index.html?type=b5&race_id={race_id}")
        baken["ワイド"].odds = netkeiba.scrape_wide(driver, race_id)
        sanrentan_odds_gen = netkeiba.scrape_sanrentan_generator(driver, race_id)
        sanrenpuku_odds_gen = netkeiba.scrape_sanrenpuku_generator(driver, race_id)
        task_logs.append(f"scraping: https://race.netkeiba.com/odds/index.html?type=b8&race_id={race_id}")
        while not tuples1_in_tuples2(baken["三連単"].nums[:top], list(baken["三連単"].odds.keys())):
            try:
                baken["三連単"].odds |= next(sanrentan_odds_gen)
            except StopIteration:
                pass
        task_logs.append(f"scraping: https://race.netkeiba.com/odds/index.html?type=b7&race_id={race_id}")
        while not tuples1_in_tuples2(baken["三連複"].nums[:top], list(baken["三連複"].odds.keys())):
            try:
                baken["三連複"].odds |= next(sanrenpuku_odds_gen)
            except StopIteration:
                pass
    return baken

def good_baken(baken, odd_th=2.0):
    for b_type, b in baken.items():
        index_synodd2 = len([odd for odd in b.df["合成オッズ"] if odd_th <= odd])
        top_odds = b.df["オッズ"][:index_synodd2]
        bets = min_bet(top_odds)
        if bets:
            bets_str = [f"{bet}円" for bet in bets]
            returns = [int(round(odd*bet, -1)) for odd, bet in zip(top_odds, bets)]
            returns_str = [f"{ret}円" for ret in returns]
            invest = sum(bets)
            min_ret, max_ret = min(returns), max(returns)
            bets_str.append(f"計: {invest}円")
            returns_str.append(f"{min_ret}円~{max_ret}円")
        else:
            bets_str = ['']
            returns_str = ['']
            invest = 0
            returns = False
            min_ret, max_ret = 0, 0
        b.df2 = pd.DataFrame()
        b.df2["均等買い"] = pd.Series(bets_str)
        b.df2["払戻"] = pd.Series(returns_str)
        b.df.index += 1
    return baken

if __name__ == "__main__":
    args = parse_args()
    race_data, horses = netkeiba.scrape_shutuba(args.race_id)
    df_horses = pd.DataFrame(horses, columns=netkeiba.HORSE_COLUMNS)
    df_races = pd.DataFrame([race_data], columns=netkeiba.RACE_PRE_COLUMNS)
    df_original = pd.merge(df_horses, df_races, on='race_id', how='left')
    print(df_original)
    p = result_prob(df_original)
    print(p)
    names = {no: name for no, name in zip(df_original["horse_no"].to_list(), df_original["name"].to_list())}
    baken = baken_prob(p, names)
    baken = calc_odds(baken, args.race_id, top=100)
    baken = good_baken(baken, odd_th=2.0)
    for b_type, b in baken.items():
        print(b_type)
        print(b.df)
    import pdb; pdb.set_trace()
