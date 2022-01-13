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
import feature_params
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
    return a*b*c / ((1-a)*(1-b-c))

def p_sanrenpuku(a, b, c):
    return p_sanrentan(a,b,c) + p_sanrentan(a,c,b) + p_sanrentan(b,a,c) \
            + p_sanrentan(b,c,a) + p_sanrentan(c,a,b) + p_sanrentan(c,b,a)

def p_umatan(a, b):
    return a*b / (1-b)

def p_umaren(a, b):
    return p_umatan(a,b) + p_umatan(b,a)

def tuples1_in_tuples2(tuples1, tuples2):
    if not tuples1:
        return False
    for t1 in tuples1:
        if t1 not in tuples2:
            return False
    return True

def predict_result_prob(df):
    with open(config.encoder_file, "rb") as f:
        netkeiba_encoder = pickle.load(f)
    with open(config.params_file, "rb") as f:
        params = pickle.load(f)
    df_format = netkeiba_encoder.format(df)
    df_encoded = netkeiba_encoder.transform(df_format)
        
    df_feat = pd.DataFrame()
    with sqlite3.connect(config.feat_db) as conn:
        for name in df_encoded["name"].unique():
            df_agg = feature_extractor.search_history(name, df_encoded, params.hist_pattern, params.feat_pattern, conn)
            df_feat = pd.concat([df_feat, df_agg])
    df_feat = df_feat.drop(columns=feature_params.NONEED_COLUMNS)

    probs = []
    for model_file in (config.rank_file, config.reg_file):
        with open(model_file, "rb") as f:
            model = pickle.load(f)
            pred = model.predict(df_feat.values, num_iteration=model.best_iteration)
            pred_prob = probability(pred)
            probs.append(pred_prob)
    prob = np.array(probs).mean(axis=0)
    return {i: p for i, p in zip(df_feat["horse_no"].to_list(), prob)}

def predict_baken_prob(prob, race_id, top=30):
    baken = {
        "単勝": Baken(),
        "馬単": Baken(),
        "馬連": Baken(),
        "三連単": Baken(),
        "三連複": Baken()
    }
    baken["単勝"].nums = list(prob.keys())
    baken["馬単"].nums = list(itertools.permutations(baken["単勝"].nums, 2))
    baken["馬連"].nums = list(itertools.combinations(baken["単勝"].nums, 2))
    baken["三連単"].nums = list(itertools.permutations(baken["単勝"].nums, 3))
    baken["三連複"].nums = list(itertools.combinations(baken["単勝"].nums, 3))
    baken["単勝"].prob = {no1: p1(no1, prob) for no1 in baken["単勝"].nums}
    baken["馬単"].prob = {(no1, no2): p_umatan(*p12(no1, no2, prob)) for no1, no2 in baken["馬単"].nums}
    baken["馬連"].prob = {tuple(sorted([no1, no2])): p_umaren(*p12(no1, no2, prob)) for no1, no2 in baken["馬連"].nums}
    baken["三連単"].prob = {(no1, no2, no3): p_sanrentan(*p123(no1, no2, no3, prob)) for no1, no2, no3 in baken["三連単"].nums}
    baken["三連複"].prob = {tuple(sorted([no1, no2, no3])): p_sanrenpuku(*p123(no1, no2, no3, prob)) for no1, no2, no3 in baken["三連複"].nums}
    
    for b_type, b in baken.items():
        high_probs = sorted(b.prob.items(), key=lambda x: x[1], reverse=True)[:top]
        baken[b_type].nums = [i for i, _ in high_probs]
        baken[b_type].prob = dict(high_probs)

    with chrome.driver() as driver:
        baken["単勝"].odds = netkeiba.scrape_tanshou(driver, race_id)
        baken["馬単"].odds = netkeiba.scrape_umatan(driver, race_id)
        baken["馬連"].odds = netkeiba.scrape_umaren(driver, race_id)
        sanrentan_odds_gen = netkeiba.scrape_sanrentan_generator(driver, race_id)
        sanrenpuku_odds_gen = netkeiba.scrape_sanrenpuku_generator(driver, race_id)
        while not tuples1_in_tuples2(baken["三連単"].nums[:top], list(baken["三連単"].odds.keys())):
            baken["三連単"].odds |= next(sanrentan_odds_gen)
        while not tuples1_in_tuples2(baken["三連複"].nums[:top], list(baken["三連複"].odds.keys())):
            baken["三連複"].odds |= next(sanrenpuku_odds_gen)

    baken["単勝"].df = pd.DataFrame({
        "一位": [str(no1) for no1 in baken["単勝"].nums],
        "単勝オッズ(予想)": [0.8/p1 for p1 in baken["単勝"].prob.values()],
        "単勝オッズ(今)": [baken["単勝"].odds[no1] for no1 in baken["単勝"].nums],
        "単勝確率": [p1 for p1 in baken["単勝"].prob.values()]
    })
    baken["馬単"].df = pd.DataFrame({
        "一位": [str(no1) for no1, no2 in baken["馬単"].nums],
        "二位": [str(no2) for no1, no2 in baken["馬単"].nums],
        "馬単オッズ(予想)": [0.75/p for p in baken["馬単"].prob.values()],
        "馬単オッズ(今)": [baken["馬単"].odds[(no1, no2)] for no1, no2 in baken["馬単"].nums],
        "馬単確率": [p for p in baken["馬単"].prob.values()]
    })
    baken["馬連"].df = pd.DataFrame({
        "一位": [str(no1) for no1, no2 in baken["馬連"].nums],
        "二位": [str(no2) for no1, no2 in baken["馬連"].nums],
        "馬連オッズ(予想)": [0.775/p for p in baken["馬連"].prob.values()],
        "馬連オッズ(今)": [baken["馬連"].odds[(no1, no2)] for no1, no2 in baken["馬連"].nums],
        "馬連確率": [p for p in baken["馬連"].prob.values()]
    })
    baken["三連単"].df = pd.DataFrame({
        "一位": [str(no1) for no1, no2, no3 in baken["三連単"].nums],
        "二位": [str(no2) for no1, no2, no3 in baken["三連単"].nums],
        "三位": [str(no3) for no1, no2, no3 in baken["三連単"].nums],
        "三連単オッズ(予想)": [0.725/p for p in baken["三連単"].prob.values()],
        "三連単オッズ(今)": [baken["三連単"].odds[(no1, no2, no3)] for no1, no2, no3 in baken["三連単"].nums],
        "三連単確率": [p for p in baken["三連単"].prob.values()]
    })
    baken["三連複"].df = pd.DataFrame({
        "一位": [str(no1) for no1, no2, no3 in baken["三連複"].nums],
        "二位": [str(no2) for no1, no2, no3 in baken["三連複"].nums],
        "三位": [str(no3) for no1, no2, no3 in baken["三連複"].nums],
        "三連複オッズ(予想)": [0.75/p for p in baken["三連複"].prob.values()],
        "三連複オッズ(今)": [baken["三連複"].odds[(no1, no2, no3)] for no1, no2, no3 in baken["三連複"].nums],
        "三連複確率": [p for p in baken["三連複"].prob.values()]
    })

    for b_type, b in baken.items():
        b.df["期待値"] = b.df[f"{b_type}オッズ(今)"] * b.df[f"{b_type}確率"]
        b.df["期待値"] = pd.Series([round(p, 2) for p in b.df["期待値"].values])
        b.df[f"{b_type}確率"] = pd.Series([f"{p*100:.2f}%" for p in b.df[f"{b_type}確率"].values])
        b.df[f"{b_type}オッズ(予想)"] = pd.Series([round(p, 1) for p in b.df[f"{b_type}オッズ(予想)"].values])
    return baken

if __name__ == "__main__":
    args = parse_args()
    horses = [horse for horse in netkeiba.scrape_shutuba(args.race_id)]
    df_original = pd.DataFrame(horses, columns=netkeiba.COLUMNS)
    result_prob = predict_result_prob(df_original)
    baken = predict_baken_prob(result_prob, args.race_id)
