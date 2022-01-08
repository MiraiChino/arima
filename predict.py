import argparse
import sqlite3

import dill as pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

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

def prob(x):
    return softmax(scale(x))

if __name__ == "__main__":
    args = parse_args()
    race_id = str(args.race_id)

    with open(config.encoder_file, "rb") as f:
        netkeiba_encoder = pickle.load(f)
    with open(config.params_file, "rb") as f:
        params = pickle.load(f)

    horses = [horse for horse in netkeiba.scrape_shutuba(race_id)]
    df_original = pd.DataFrame(horses, columns=netkeiba.COLUMNS)
    df_format = netkeiba_encoder.format(df_original)
    df_encoded = netkeiba_encoder.transform(df_format)

    df_feat = pd.DataFrame()
    with sqlite3.connect(config.feat_db) as conn:
        for name in df_encoded["name"].unique():
            df_agg = feature_extractor.search_history(name, df_encoded, params.hist_pattern, params.feat_pattern, conn)
            df_feat = pd.concat([df_feat, df_agg])
    df_feat = df_feat.drop(columns=feature_params.NONEED_COLUMNS)
    with open(config.rank_file, "rb") as f:
        rank_model = pickle.load(f)
        rank_pred = rank_model.predict(df_feat.values, num_iteration=rank_model.best_iteration)
        rank_prob = prob(rank_pred)
    with open(config.reg_file, "rb") as f:
        reg_model = pickle.load(f)
        reg_pred = reg_model.predict(df_feat.values, num_iteration=reg_model.best_iteration)
        reg_prob = prob(reg_pred)
    prob = np.array([reg_prob, rank_prob]).mean(axis=0)

