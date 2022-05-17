import argparse
import itertools
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

def result_prob(df):
    with open(config.encoder_file, "rb") as f:
        netkeiba_encoder = pickle.load(f)
    df_format = netkeiba_encoder.format(df)
    df_encoded = netkeiba_encoder.transform(df_format)
    df_feat = pd.DataFrame()
    with sqlite3.connect(config.feat_db) as conn:
        df_avetime = pd.read_sql_query(f"SELECT * FROM ave_time", conn)
        ave_time = {(f, d, fc): t for f, d, fc, t in df_avetime.to_dict(orient="split")["data"]}
        hist_pattern = config.hist_pattern
        feat_pattern = config.feature_pattern(ave_time)
        race_date = df_encoded["race_date"][0]
        condition = " OR ".join(f"{column}=={row[column]}" for _, row in df_encoded.iterrows() for column in feat_pattern.keys())
        reader = pd.read_sql_query(f"SELECT * FROM horse WHERE race_date < '{race_date}' AND ({condition})", conn, chunksize=10000)
        chunks = []
        for df_chunk in reader:
            df_chunk_reduced = utils.reduce_mem_usage(df_chunk, verbose=True)
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
    for model_file in (config.rank_file, config.reg_file):
        with open(model_file, "rb") as f:
            model = pickle.load(f)
            pred = model.predict(df_feat.values, num_iteration=model.best_iteration)
            pred_prob = probability(pred)
            probs.append(pred_prob)
    prob = np.array(probs).mean(axis=0)
    return {i: p for i, p in zip(df_feat["horse_no"].to_list(), prob)}

def baken_prob(prob, names, race_id, top=30):
    baken = {
        "単勝": Baken(),
        "複勝": Baken(),
        "馬単": Baken(),
        "馬連": Baken(),
        "ワイド": Baken(),
        "三連単": Baken(),
        "三連複": Baken()
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
        high_probs = sorted(b.prob.items(), key=lambda x: x[1], reverse=True)[:top]
        baken[b_type].nums = [i for i, _ in high_probs]
        baken[b_type].prob = dict(high_probs)

    with chrome.driver() as driver:
        baken["単勝"].odds = netkeiba.scrape_tanshou(driver, race_id)
        baken["複勝"].odds = netkeiba.scrape_hukushou(driver, race_id)
        baken["馬単"].odds = netkeiba.scrape_umatan(driver, race_id)
        baken["馬連"].odds = netkeiba.scrape_umaren(driver, race_id)
        baken["ワイド"].odds = netkeiba.scrape_wide(driver, race_id)
        sanrentan_odds_gen = netkeiba.scrape_sanrentan_generator(driver, race_id)
        sanrenpuku_odds_gen = netkeiba.scrape_sanrenpuku_generator(driver, race_id)
        while not tuples1_in_tuples2(baken["三連単"].nums[:top], list(baken["三連単"].odds.keys())):
            try:
                baken["三連単"].odds |= next(sanrentan_odds_gen)
            except StopIteration:
                pass
        while not tuples1_in_tuples2(baken["三連複"].nums[:top], list(baken["三連複"].odds.keys())):
            try:
                baken["三連複"].odds |= next(sanrenpuku_odds_gen)
            except StopIteration:
                pass

    baken["単勝"].df = pd.DataFrame({
        "馬番": [str(no1) for no1 in baken["単勝"].nums],
        "馬名": [names[no-1] for no in baken["単勝"].nums],
        "オッズ(予想)": [0.8/p1 for p1 in baken["単勝"].prob.values()],
        "オッズ(今)": [baken["単勝"].odds[no1] for no1 in baken["単勝"].nums],
        "確率": [p1 for p1 in baken["単勝"].prob.values()]
    })
    baken["複勝"].df = pd.DataFrame({
        "馬番": [str(no1) for no1 in baken["複勝"].nums],
        "馬名": [names[no-1] for no in baken["複勝"].nums],
        "オッズ(予想)": [0.8/p1 for p1 in baken["複勝"].prob.values()],
        "オッズ(今)": [baken["複勝"].odds[no1] for no1 in baken["複勝"].nums],
        "確率": [p1 for p1 in baken["複勝"].prob.values()]
    })
    baken["馬単"].df = pd.DataFrame({
        "一位": [str(no1) for no1, no2 in baken["馬単"].nums],
        "二位": [str(no2) for no1, no2 in baken["馬単"].nums],
        "オッズ(予想)": [0.75/p for p in baken["馬単"].prob.values()],
        "オッズ(今)": [baken["馬単"].odds[(no1, no2)] for no1, no2 in baken["馬単"].nums],
        "確率": [p for p in baken["馬単"].prob.values()]
    })
    baken["馬連"].df = pd.DataFrame({
        "一位": [str(no1) for no1, no2 in baken["馬連"].nums],
        "二位": [str(no2) for no1, no2 in baken["馬連"].nums],
        "オッズ(予想)": [0.775/p for p in baken["馬連"].prob.values()],
        "オッズ(今)": [baken["馬連"].odds[(no1, no2)] for no1, no2 in baken["馬連"].nums],
        "確率": [p for p in baken["馬連"].prob.values()]
    })
    baken["ワイド"].df = pd.DataFrame({
        "一位": [str(no1) for no1, no2 in baken["ワイド"].nums],
        "二位": [str(no2) for no1, no2 in baken["ワイド"].nums],
        "オッズ(予想)": [0.775/p for p in baken["ワイド"].prob.values()],
        "オッズ(今)": [baken["ワイド"].odds[(no1, no2)] for no1, no2 in baken["ワイド"].nums],
        "確率": [p for p in baken["ワイド"].prob.values()]
    })
    baken["三連単"].df = pd.DataFrame({
        "一位": [str(no1) for no1, no2, no3 in baken["三連単"].nums],
        "二位": [str(no2) for no1, no2, no3 in baken["三連単"].nums],
        "三位": [str(no3) for no1, no2, no3 in baken["三連単"].nums],
        "オッズ(予想)": [0.725/p for p in baken["三連単"].prob.values()],
        "オッズ(今)": [baken["三連単"].odds[(no1, no2, no3)] for no1, no2, no3 in baken["三連単"].nums],
        "確率": [p for p in baken["三連単"].prob.values()]
    })
    baken["三連複"].df = pd.DataFrame({
        "一位": [str(no1) for no1, no2, no3 in baken["三連複"].nums],
        "二位": [str(no2) for no1, no2, no3 in baken["三連複"].nums],
        "三位": [str(no3) for no1, no2, no3 in baken["三連複"].nums],
        "オッズ(予想)": [0.75/p for p in baken["三連複"].prob.values()],
        "オッズ(今)": [baken["三連複"].odds[(no1, no2, no3)] for no1, no2, no3 in baken["三連複"].nums],
        "確率": [p for p in baken["三連複"].prob.values()]
    })
    baken["三連複"].df.style.set_properties(subset=['text'], **{'width': '300px'})

    for b_type, b in baken.items():
        b.df["期待値"] = b.df["オッズ(今)"] * b.df["確率"]
        b.df["期待値"] = pd.Series([round(p, 2) for p in b.df["期待値"].values])
        b.df["確率"] = pd.Series([f"{p*100:.2f}%" for p in b.df["確率"].values])
        b.df["オッズ(予想)"] = pd.Series([round(p, 1) for p in b.df["オッズ(予想)"].values])
        b.df["合成オッズ"] = pd.Series(cumulative_odds(b.df["オッズ(今)"].values))
        b.df["合成オッズ"] = pd.Series([round(p, 2) for p in b.df["合成オッズ"].values])
        b.df["累積確率"] = pd.Series([f"{p*100:.2f}%" for p in cumulative_prob(list(b.prob.values()))])
        b.df["合成期待値"] = b.df["合成オッズ"] * cumulative_prob(list(b.prob.values()))
        b.df["合成期待値"] = pd.Series([round(p, 2) for p in b.df["合成期待値"].values])
        b.df.index += 1
    return baken

if __name__ == "__main__":
    args = parse_args()
    horses = [horse for horse in netkeiba.scrape_shutuba(args.race_id)]
    df_original = pd.DataFrame(horses, columns=netkeiba.COLUMNS)
    print(df_original)
    p = result_prob(df_original)
    print(p)
    baken = baken_prob(p, df_original["name"].to_list(), args.race_id)
    for b_type, b in baken.items():
        print(b_type)
        print(b.df)
