import argparse
import itertools
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import dill as pickle
import numpy as np
import pandas as pd
import polars as pl

import chrome
import config
import featlist
import feature
import netkeiba
from knn import UsearchKNeighborsRegressor


scaler = None
l1_models = {}
l2_modelname, l2_model = None, None
re_modelfile = re.compile(r"^models/(.*)_\d+.*_\d+.*$")
horse_encoder = None
history = None
hist_pattern = featlist.hist_pattern
feat_pattern = featlist.feature_pattern

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
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)  # NaNや無限大を処理
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return np.zeros_like(x)  # 全て0にする
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

def load_models_and_configs(task_logs=[]):
    global scaler, l1_models, l2_modelname, l2_model, horse_encoder, history, hist_pattern, feat_pattern

    task_logs.append(f"loading {config.encoder_file}")
    if horse_encoder is None:
        with open(config.encoder_file, "rb") as f:
            horse_encoder = pickle.load(f)

    task_logs.append(f"loading {config.feat_file}")
    if history is None:
        history = pl.read_ipc(config.feat_file)

    task_logs.append(f"loading scaler")
    with open(f"models/{config.scaler_file}", "rb") as f:
        scaler = pickle.load(f)

    task_logs.append(f"loading models")
    for i, query in enumerate(config.layer1_splits):
        for model_config in config.l1_models:
            model_file = f"models/{i}_{model_config.file}"
            model_name = re_modelfile.match(model_file).groups()[0]
            task_logs.append(f"loading {model_name}")
            if model_name not in l1_models:
                if "kn" in model_name:
                    model = UsearchKNeighborsRegressor()
                    model.load(model_file)
                else:
                    with open(model_file, "rb") as f:
                        model = pickle.load(f)
                l1_models[model_name] = model

    if not l2_model:
        model_file = f"models/{config.l2_stacking_lgb_rank.file}"
        l2_modelname = re_modelfile.match(model_file).groups()[0]
        with open(model_file, "rb") as f:
            l2_model = pickle.load(f)
            task_logs.append(f"loaded {l2_modelname}")

def search_df_feat(df, task_logs=[]):
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
    global horse_encoder, history
    if horse_encoder is None:
        with open(config.encoder_file, "rb") as f:
            horse_encoder = pickle.load(f)
    if history is None:
        history = pl.read_ipc(config.feat_file)
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

    task_logs.append(f"filtering race history")
    condition = [pl.col(column).is_in(df_encoded[column].to_list()) for column in feat_pattern.keys()]
    race_date = df_encoded["race_date"][0]
    hist = history.filter((pl.col('race_date') < race_date) & pl.any_horizontal(condition))
    
    hist = hist.select([
        *df_encoded.columns,
        'avetime', 'aversrize',
        'horse_oldr', 'jockey_oldr', 'trainer_oldr',
        'horse_newr', 'jockey_newr', 'trainer_newr',
    ])

    # 過去の最頻runningをrunning_styleとして定義
    task_logs.append(f"calculating running style")
    horse_ids = df_encoded.get_column("horse_id").cast(pl.Int32).to_list()
    running_styles = []
    for horse_id in horse_ids:
        horse_hist = hist.filter(pl.col("horse_id") == horse_id)
        last_race = horse_hist[-1]
        if last_race.is_empty():
            running_style = -1
        else:
            running_style = last_race.get_column("running_style").to_list()[0]
        running_styles.append(running_style)
    df_running_style = pl.DataFrame({
        "horse_id": horse_ids,
        "new_running_style": running_styles,
    }).with_columns(pl.col("horse_id").cast(pl.Float64), pl.col("new_running_style").cast(pl.Int8))
    df_encoded = df_encoded.join(df_running_style, on="horse_id", how="left")
    df_encoded = df_encoded.with_columns(
        pl.col("new_running_style").fill_null(pl.col("running_style")).alias("running_style")
    )
    df_encoded = df_encoded.drop("new_running_style")

    # avetime
    task_logs.append(f"calculating avetime")
    race_condition = ['field', 'distance', 'field_condition']
    avetime = hist.select([*race_condition, 'avetime']).unique(subset=[*race_condition, 'avetime'])
    df_encoded = df_encoded.join(avetime, on=race_condition, how='left')
    # aversrize
    task_logs.append(f"calculating aversrize")
    race_condition = ['running_style', 'field', 'distance', 'field_condition', 'place_code', 'gate', 'turn']
    aversrize = hist.select([*race_condition, 'aversrize']).unique(subset=[*race_condition, 'aversrize'])
    df_encoded = df_encoded.join(aversrize, on=race_condition, how='left')

    task_logs.append(f"searching race history")
    df_feat = pd.DataFrame()
    for row in df_encoded.rows():
        df_agg = feature.search_history(row, hist_pattern, feat_pattern, hist)
        if df_feat.empty:
            df_feat = df_agg
        else:
            df_feat = pd.concat([df_feat, df_agg])
    noneed_columns = [c for c in config.NONEED_COLUMNS if c in df_feat.columns]
    df_feat = df_feat.drop(columns=noneed_columns) # (16, 964)
    return df_feat

def classification_to_regression(model, class_labels, x):
    pred_x_proba = model.predict_proba(x)
    pred_x = np.dot(pred_x_proba, class_labels)
    return pred_x

def result_prob(df_feat, task_logs=[]):
    task_logs.append(f"predict")
    preds = []

    # Layer 1 predictions
    for i, query in enumerate(config.layer1_splits):
        preds_l1 = []
        for model_config in config.l1_models:
            model_file = f"models/{i}_{model_config.file}"
            model_name = re_modelfile.match(model_file).groups()[0]
            if model_name not in l1_models.keys():
                continue
            try:
                model = l1_models[model_name]
                if "lgb" in model_name:
                    values = df_feat[model.feature_name()].values
                    pred = model.predict(values, num_iteration=model.best_iteration)
                elif "sgd" in model_name:
                    df_feat = df_feat.fillna(0)[scaler.feature_names_in_]
                    pred = model.predict(scaler.transform(df_feat))
                elif "ard" in model_name:
                    df_feat = df_feat.fillna(0)[scaler.feature_names_in_]
                    pred = model.predict(scaler.transform(df_feat))
                elif "huber" in model_name:
                    df_feat = df_feat.fillna(0)[scaler.feature_names_in_]
                    pred = model.predict(scaler.transform(df_feat))
                elif "br" in model_name:
                    df_feat = df_feat.fillna(0)[scaler.feature_names_in_]
                    pred = model.predict(scaler.transform(df_feat))
                elif "etr" in model_name:
                    df_feat = df_feat[model.feature_names_in_]
                    pred = model.predict(df_feat)
                elif "en" in model_name:
                    df_feat = df_feat.fillna(0)[scaler.feature_names_in_]
                    pred = model.predict(scaler.transform(df_feat))
                elif "rf" in model_name:
                    pred = model.predict(df_feat[model.feature_names_in_])
                elif "kn" in model_name:
                    pred = model.predict(df_feat)
                elif "lr" in model_name:
                    model, class_labels = model
                    pred = classification_to_regression(model, class_labels, df_feat[scaler.feature_names_in_])
                elif "gnb" in model_name:
                    model, class_labels = model
                    pred = classification_to_regression(model, class_labels, df_feat[scaler.feature_names_in_])
                
            except:
                import traceback; print(traceback.format_exc())
                import pdb; pdb.set_trace()
            task_logs.append(f"{model_name}: {[f'{p*100:.1f}%' for p in probability(pred)]}")
            preds_l1.append(pred)
        preds.append(np.column_stack(preds_l1))

    # Layer 2 prediction
    stacked_feat = np.hstack(preds)
    x = np.hstack([df_feat, stacked_feat])
    pred = l2_model.predict(x, num_iteration=l2_model.best_iteration)
    prob = probability(pred)
    task_logs.append(f"{l2_modelname}: {[f'{p*100:.1f}%' for p in probability(pred)]}")
    return {i: p for i, p in zip(df_feat["horse_no"].to_list(), prob)}

def bin_race_dict(df_past, breakpoints, bin_count, task_logs=[]):
    race_dict = {}
    df_past = pl.DataFrame(df_past)

    task_logs.append("calculating bins")
    # 各カラムのビンを生成し、カウントを更新
    for col, breaks in breakpoints.items():
        if col in ["race_id", "race_date"] + config.NONEED_COLUMNS:
            continue
        
        # ラベルを初期化
        labels = [str(i) for i in range(bin_count)]
        for i in labels:
            race_dict[f"{col}_bin{i}"] = 0

        # データをビンにカット
        bins = df_past[col].cut(breaks, labels=labels)
        for i in bins:
            key = f"{col}_bin{i}"
            if key in race_dict.keys():
                race_dict[key] += 1
    return race_dict

def result_bakenhit(df_past, task_logs=[]):
    task_logs.append(f'loading {config.breakpoint_file}')
    with open(config.breakpoint_file, "rb") as f:
        breakpoints = pickle.load(f)
   
    important_columns = list(breakpoints.keys())
    noneed = ["race_id", "race_date"] + config.NONEED_COLUMNS
    df_past = df_past[important_columns].drop(columns=noneed, errors="ignore")
    race_dict = bin_race_dict(df_past, breakpoints, config.bakenhit_lgb_reg.bins, task_logs)
    race_features_df = pl.DataFrame([race_dict])

    task_logs.append(f'loading {config.bakenhit_lgb_reg.file}')
    try:
        with open(f"{config.bakenhit_lgb_reg.file}", "rb") as f:
            model, pred_valid_x = pickle.load(f)
        values = race_features_df[model.feature_name()].to_pandas().values
        pred = model.predict(values, num_iteration=model.best_iteration)
        task_logs.append(f'bakenhit prob {pred}')
    except:
        task_logs.append(f'Could not predict bakenhit prob')
        return 0
    return pred.tolist()[0]

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
                odds.append(b.odds.get(nums, None))
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
        task_logs.append(f"scraping 単勝,複勝: https://race.netkeiba.com/odds/index.html?race_id={race_id}")
        baken["単勝"].odds, baken["複勝"].odds = netkeiba.scrape_tanhuku(driver, race_id)
        task_logs.append(f"scraping 馬単: https://race.netkeiba.com/odds/index.html?type=b6&race_id={race_id}")
        baken["馬単"].odds = netkeiba.scrape_umatan(driver, race_id)
        task_logs.append(f"scraping 馬連: https://race.netkeiba.com/odds/index.html?type=b4&race_id={race_id}")
        baken["馬連"].odds = netkeiba.scrape_umaren(driver, race_id)
        task_logs.append(f"scraping ワイド: https://race.netkeiba.com/odds/index.html?type=b5&race_id={race_id}")
        baken["ワイド"].odds = netkeiba.scrape_wide(driver, race_id)
        sanrentan_odds_gen = netkeiba.scrape_sanrentan_generator(driver, race_id)
        sanrenpuku_odds_gen = netkeiba.scrape_sanrenpuku_generator(driver, race_id)
        task_logs.append(f"scraping 三連単: https://race.netkeiba.com/odds/index.html?type=b8&race_id={race_id}")
        while not tuples1_in_tuples2(baken["三連単"].nums[:top], list(baken["三連単"].odds.keys())):
            try:
                len_scraped = len(baken["三連単"].odds)
                result = next(sanrentan_odds_gen)
                task_logs.append(f"scraping 三連単: 人気 {len_scraped+1}~{len_scraped+len(result)}")
                baken["三連単"].odds |= result
            except StopIteration:
                break
        task_logs.append(f"scraping 三連複: https://race.netkeiba.com/odds/index.html?type=b7&race_id={race_id}")
        while not tuples1_in_tuples2(baken["三連複"].nums[:top], list(baken["三連複"].odds.keys())):
            try:
                len_scraped = len(baken["三連複"].odds)
                result = next(sanrenpuku_odds_gen)
                task_logs.append(f"scraping 三連複: 人気 {len_scraped+1}~{len_scraped+len(result)}")
                baken["三連複"].odds |= result
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
    load_models_and_configs()  # モデルを事前にロード
    df_feat = search_df_feat(df_original)
    p = result_prob(df_feat)
    print(p)
    bakenhit_prob = result_bakenhit(df_feat)
    print(bakenhit_prob)
    names = {no: name for no, name in zip(df_original["horse_no"].to_list(), df_original["name"].to_list())}
    baken = baken_prob(p, names)
    baken = calc_odds(baken, args.race_id, top=100)
    baken = pretty_baken(baken, top=100)
    baken = good_baken(baken, odd_th=2.0)
    for b_type, b in baken.items():
        print(b_type)
        print(b.df)
    import pdb; pdb.set_trace()
