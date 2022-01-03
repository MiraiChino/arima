import argparse
import pickle
import sqlite3

import pandas as pd

import encoder
import feature_extractor
import netkeiba
import train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raceid", dest="race_id", required=True, type=int,
                        help="Example 202206010111.")

if __name__ == "__main__":
    args = parse_args()
    race_id = str(args.race_id)

    with open("encoder.pickle", "rb") as f:
        netkeiba_encoder = pickle.load(f)
    with open("params.pickle", "rb") as f:
        params = pickle.load(f)

    horses = [horse for horse in netkeiba.scrape_shutuba(race_id)]
    df_original = pd.DataFrame(horses, columns=netkeiba.COLUMNS)
    df_format = encoder.format(df_original)
    df_encoded = netkeiba_encoder.transform(df_format)

    df_feat = pd.DataFrame()
    with sqlite3.connect("feature.sqlite") as conn:
        for name in df_encoded["name"].unique():
            df_agg = feature_extractor.search_history(name, df_encoded, params.hist_pattern, params.feat_pattern, conn)
            df_feat = pd.concat([df_feat, df_agg])
    df_feat = df_feat.drop(columns=train.NONEED_COLUMNS)

    with open("rank_model.pickle", "rb") as f:
        rank_model = pickle.load(f)
        rank_pred = rank_model.predict(df_feat.values, num_iteration=rank_model.best_iteration)
    with open("reg_model.pickle", "rb") as f:
        reg_model = pickle.load(f)
        reg_pred = reg_model.predict(df_feat.values, num_iteration=reg_model.best_iteration)

