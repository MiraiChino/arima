import argparse
import itertools
import math
import re
from dataclasses import dataclass, field

import dill as pickle
import numpy as np
import pandas as pd
import polars as pl

import chrome
import config
import featlist
import feature
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
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    df_return: pd.DataFrame = field(default_factory=pd.DataFrame)

def normalize(probs, max_value):
    """確率を指定された最大値に正規化する関数"""
    total_prob = sum(probs.values())
    if total_prob == 0:
        return {k: 0 for k in probs}  # 確率がすべて0の場合は0で返す
    scaling_factor = max_value / total_prob
    return {k: v * scaling_factor for k, v in probs.items()}

def standardize(x):
    """標準化関数：平均0、標準偏差1に変換"""
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / std

def softmax(x):
    """ソフトマックス関数"""
    c = np.max(x)
    exp_x = np.exp(x - c)
    return exp_x / np.sum(exp_x)

def probability(x):
    """標準化された入力に対してソフトマックスを計算"""
    return softmax(standardize(x))

def p1(no1_index, probs):
    return probs[no1_index]

def p12(no1_index, no2_index, probs, g=0.81):
    p2 = probs[no2_index]**g / sum(x**g for x in probs.values())
    return p1(no1_index, probs), p2

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
        horse_encoder = pickle.load(f)

    task_logs.append(f"encoding")
    df = df.with_columns(
        pl.lit(None).alias('tanno1'),
        pl.lit(None).alias('tanno2'),
        pl.lit(None).alias('hukuno1'),
        pl.lit(None).alias('hukuno2'),
        pl.lit(None).alias('hukuno3'),
        pl.lit(None).alias('tan1'),
        pl.lit(None).alias('tan2'),
        pl.lit(None).alias('huku1'),
        pl.lit(None).alias('huku2'),
        pl.lit(None).alias('huku3'),
        pl.lit(None).alias('corner1_group'),
        pl.lit(None).alias('corner2_group'),
        pl.lit(None).alias('corner3_group'),
        pl.lit(None).alias('corner4_group'),
        pl.lit(None).alias('running'),
        pl.lit(None).alias('running_style').cast(pl.Int8),
        pl.col("race_id").count().over("race_id").alias("num_horses"),
        pl.col('corner').cast(str),
        pl.col('last3f').cast(float),
        pl.col('weight').cast(float),
        pl.col('time').cast(str),
    )
    df_formatted = horse_encoder.format(df)
    df_encoded = horse_encoder.transform(df_formatted)
    df_encoded = df_encoded.with_columns([
        pl.col('race_date').cast(pl.Datetime),
        pl.col('field').cast(pl.Int8),
        pl.col('distance').cast(pl.Int16),
        pl.col('field_condition').cast(pl.Int8),
        pl.col('place_code').cast(pl.Int8),
        pl.col('gate').cast(pl.Int8),
        pl.col('turn').cast(pl.Int8),
    ])

    task_logs.append(f"loading {config.feat_file}")
    history = pl.read_ipc(config.feat_file)
    hist_pattern = featlist.hist_pattern
    feat_pattern = featlist.feature_pattern

    task_logs.append(f"searching race history")
    condition = [pl.col(column).is_in(df_encoded[column].to_list()) for column in feat_pattern.keys()]
    race_date = df_encoded["race_date"][0]
    hist = history.filter((pl.col('race_date') < race_date) & pl.any_horizontal(condition))
    hist = hist.select([
        *df_encoded.columns,
        'avetime', 'aversrize',
        'horse_oldr', 'jockey_oldr', 'trainer_oldr',
        'horse_newr', 'jockey_newr', 'trainer_newr',
    ])

    # avetime
    race_condition = ['field', 'distance', 'field_condition']
    avetime = hist.select([*race_condition, 'avetime']).unique(subset=[*race_condition, 'avetime'])
    df_encoded = df_encoded.join(avetime, on=race_condition, how='left')

    # aversrize
    race_condition = ['running_style', 'field', 'distance', 'field_condition', 'place_code', 'gate', 'turn']
    aversrize = hist.select([*race_condition, 'aversrize']).unique(subset=[*race_condition, 'aversrize'])
    df_encoded = df_encoded.join(aversrize, on=race_condition, how='left')
    
    df_feat = pd.DataFrame()
    for row in df_encoded.rows():
        df_agg = feature.search_history(row, hist_pattern, feat_pattern, hist)
        df_feat = (df_feat.copy() if df_agg.empty else df_agg.copy() if df_feat.empty
            else pd.concat([df_feat.astype(df_agg.dtypes), df_agg.astype(df_feat.dtypes)])
        )
    noneed_columns = [c for c in config.NONEED_COLUMNS if c in df_feat.columns]
    df_feat = df_feat.drop(columns=noneed_columns) # (16, 964)
    task_logs.append(f"predict")
    preds = []
    re_modelfile = re.compile(r"^models/(.*)_\d+.*_\d+.*$")
    for model in config.lgb_models:
        preds_lgb = []
        for i in range(len(config.splits)):
            model_file = f"models/{i}_{model.file}"
            model_name = re_modelfile.match(model_file).groups()[0]
            with open(model_file, "rb") as f:
                m = pickle.load(f)
                pred = m.predict(df_feat.values, num_iteration=m.best_iteration)
                preds_lgb.append(pred)
                task_logs.append(f"{model_name}: {[f'{p*100:.1f}%' for p in probability(pred)]}")
        preds.append(preds_lgb)
    for model in config.cat_models:
        preds_cat = []
        for i in range(len(config.splits)):
            model_file = f"models/{i}_{model.file}"
            model_name = re_modelfile.match(model_file).groups()[0]
            with open(model_file, "rb") as f:
                m = pickle.load(f)
                for col in config.cat_features:
                    df_feat[col] = df_feat[col].astype('int64')
                pred = m.predict(df_feat.values, prediction_type='RawFormulaVal')
                preds_cat.append(pred)
                task_logs.append(f"{model_name}: {[f'{p*100:.1f}%' for p in probability(pred)]}")
        preds.append(preds_cat)
    stacked_feat = np.array(preds).mean(axis=1).T # (16, 4)
    x = np.hstack([df_feat, stacked_feat]) # (16, 756)
    model = config.stacking_model
    model_file = f"models/{model.file}"
    model_name = re_modelfile.match(model_file).groups()[0]
    with open(model_file, "rb") as f:
        m = pickle.load(f)
        pred = m.predict(x, num_iteration=m.best_iteration)
        prob = probability(pred)
        task_logs.append(f"{model_name}: {[f'{p*100:.1f}%' for p in probability(pred)]}")
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

    # 各馬券の確率を正規化
    baken["単勝"].prob = normalize(baken["単勝"].prob, 1.0)
    baken["馬単"].prob = normalize(baken["馬単"].prob, 1.0)
    baken["馬連"].prob = normalize(baken["馬連"].prob, 1.0)
    baken["三連単"].prob = normalize(baken["三連単"].prob, 1.0)
    baken["三連複"].prob = normalize(baken["三連複"].prob, 1.0)
    baken["複勝"].prob = normalize(baken["複勝"].prob, 3.0)
    baken["ワイド"].prob = normalize(baken["ワイド"].prob, 3.0)

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
        odds = []
        for nums in b.nums:
            try:
                odds.append(b.odds[nums])
            except Exception as e:
                print(f"{e}: Not found {nums}")
        b.df["確率"] = pd.Series([f"{p*100:.2f}%" for p in b.df["確率"].values])
        b.df["オッズ"] = pd.Series(odds)
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
                break
        task_logs.append(f"scraping: https://race.netkeiba.com/odds/index.html?type=b7&race_id={race_id}")
        while not tuples1_in_tuples2(baken["三連複"].nums[:top], list(baken["三連複"].odds.keys())):
            try:
                baken["三連複"].odds |= next(sanrenpuku_odds_gen)
            except StopIteration:
                break
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
        b.df_return = pd.DataFrame()
        b.df_return["均等買い"] = pd.Series(bets_str)
        b.df_return["払戻"] = pd.Series(returns_str)
        b.df.index += 1
    return baken

if __name__ == "__main__":
    args = parse_args()
    race_data, horses = netkeiba.scrape_shutuba(args.race_id)
    df_horses = pl.DataFrame(horses, schema=netkeiba.HORSE_COLUMNS, orient="row")
    df_races = pl.DataFrame([race_data], schema=netkeiba.RACE_PRE_COLUMNS, orient="row")
    df_original = df_horses.join(df_races, on='race_id', how='left')
    print(df_original)
    p = result_prob(df_original)
    print(p)
    names = {no: name for no, name in zip(df_original["horse_no"].to_list(), df_original["name"].to_list())}
    baken = baken_prob(p, names)
    baken = calc_odds(baken, args.race_id, top=100)
    baken = pretty_baken(baken, top=100)
    baken = good_baken(baken, odd_th=2.0)
    for b_type, b in baken.items():
        print(b_type)
        print(b.df)
    import pdb; pdb.set_trace()
