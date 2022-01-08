import argparse
import itertools
import sqlite3

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
    parser.add_argument("--raceid", dest="race_id", required=True, type=int,
                        help="Example 202206010111.")
    return parser.parse_args()

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
    return p1(no1_index), p2, 

def p123(no1_index, no2_index, no3_index, probs, d=0.65):
    p3 = probs[no3_index]**d / sum(x**d for x in probs.values())
    return p12(no1_index, no2_index, probs), p3

def sanrentan(a, b, c):
    return a*b*c / ((1-a)*(1-b-c))

def sanrenpuku(a, b, c):
    return sanrentan(a,b,c) + sanrentan(a,c,b) + sanrentan(b,a,c) \
            + sanrentan(b,c,a) + sanrentan(c,a,b) + sanrentan(c,b,a)

def umatan(a, b):
    return a*b / (1-b)

def umaren(a, b):
    return umatan(a,b) + umatan(b,a)

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
    return prob

def predict_baken_prob(prob):
    df_prob = pd.DataFrame({"馬番": df_original["horse_no"].values, "馬名": df_original["name"].values, "単勝確率": prob})
    tanshou_order = df_prob.sort_values("単勝確率", ascending=False)["馬番"].values.tolist()

    with chrome.driver() as driver:
        odds_tanshou = netkeiba.scrape_tanshou(driver)
        odds_umatan = netkeiba.scrape_umatan(driver)
        odds_umaren = netkeiba.scrape_umaren(driver)
        odds_sanrentan = netkeiba.scrape_sanrentan(driver)
        odds_sanrenpuku = netkeiba.scrape_sanrenpuku(driver)

    baken = {}
    baken["単勝"] = pd.DataFrame({
        "一位": [str(no1) for no1 in tanshou_order],
        "単勝オッズ(予想)": [0.8/p1(no1) for no1 in tanshou_order],
        "単勝オッズ(今)": [odds_tanshou.get(no1) for no1 in tanshou_order],
        "単勝確率": [p1(no1) for no1 in tanshou_order]
    })

    perm2 = list(itertools.permutations(tanshou_order, 2))
    baken["馬単"] = pd.DataFrame({
        "一位": [str(no1) for no1, no2 in perm2],
        "二位": [str(no2) for no1, no2 in perm2],
        "馬単オッズ(予想)": [0.775/umatan(*p12(no1, no2)) for no1, no2 in perm2],
        "馬単オッズ(今)": [odds_umatan.get((no1, no2)) for no1, no2 in perm2],
        "馬単確率": [umatan(*p12(no1, no2)) for no1, no2 in perm2]
    })

    comb2 = list(itertools.combinations(tanshou_order, 2))
    baken["馬連"] = pd.DataFrame({
        "一位": [str(no1) for no1, no2 in comb2],
        "二位": [str(no2) for no1, no2 in comb2],
        "馬連オッズ(予想)": [0.775/umaren(*p12(no1, no2)) for no1, no2 in comb2],
        "馬連オッズ(今)": [odds_umaren.get(tuple(sorted([no1, no2]))) for no1, no2 in comb2],
        "馬連確率": [umaren(*p12(no1, no2)) for no1, no2 in comb2]
    })

    perm3 = list(itertools.permutations(tanshou_order, 3))
    baken["三連単"] = pd.DataFrame({
        "一位": [str(no1) for no1, no2, no3 in perm3],
        "二位": [str(no2) for no1, no2, no3 in perm3],
        "三位": [str(no3) for no1, no2, no3 in perm3],
        "三連単オッズ(予想)": [0.775/sanrentan(*p123(no1, no2, no3)) for no1, no2, no3 in perm3],
        "三連単オッズ(今)": [odds_sanrentan.get((no1, no2, no3)) for no1, no2, no3 in perm3],
        "三連単確率": [sanrentan(*p123(no1, no2, no3)) for no1, no2, no3 in perm3]
    })

    comb3 = list(itertools.combinations(tanshou_order, 3))
    baken["三連複"] = pd.DataFrame({
        "一位": [str(no1) for no1, no2, no3 in comb3],
        "二位": [str(no2) for no1, no2, no3 in comb3],
        "三位": [str(no3) for no1, no2, no3 in comb3],
        "三連複オッズ(予想)": [0.775/sanrenpuku(*p123(no1, no2, no3)) for no1, no2, no3 in comb3],
        "三連複オッズ(今)": [odds_sanrenpuku.get(tuple(sorted([no1, no2, no3]))) for no1, no2, no3 in comb3],
        "三連複確率": [sanrenpuku(*p123(no1, no2, no3)) for no1, no2, no3 in comb3]
    })

    for baken_name, df_baken in baken.items():
        df_baken.sort_values(f"{baken_name}確率").reset_index(drop=True, inplace=True)
        df_baken["期待値"] = df_baken["{baken_name}オッズ(今)"] * df_baken[f"{baken_name}確率"]
        df_baken[f"{baken_name}確率"] = pd.Series([f"{p*100:.2f}%" for p in df_baken[f"{baken_name}確率"].values])
        df_baken[f"{baken_name}オッズ(予想)"] = pd.Series([round(p, 1) for p in df_baken[f"{baken_name}オッズ(予想)"].values])
        df_baken["期待値"] = pd.Series([round(p, 3) for p in df_baken["期待値"].values])
    return baken

if __name__ == "__main__":
    args = parse_args()
    race_id = str(args.race_id)
    
    horses = [horse for horse in netkeiba.scrape_shutuba(race_id)]
    df_original = pd.DataFrame(horses, columns=netkeiba.COLUMNS)
    result_prob = predict_result_prob(df_original)
    baken = predict_baken_prob(result_prob)
