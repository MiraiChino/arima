import argparse
import functools
import traceback
from datetime import timedelta
from pathlib import Path

import dill as pickle
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
from trueskill import TrueSkill

import config
import featlist
from encoder import HorseEncoder


def parse_args():
    """
    コマンドライン引数を解析します。

    Returns:
        Namespace: 解析された引数を含むオブジェクト。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--out', action='store_true')
    return parser.parse_args()

def search_history(target_row, hist_pattern, feat_pattern, df):
    """
    特定の行に基づいて履歴データを検索し、特徴量を生成します。

    Args:
        target_row (list): 対象の行データ。
        hist_pattern (dict): 履歴パターンの辞書。
        feat_pattern (dict): 特徴量パターンの辞書。
        df (DataFrame): 検索対象のデータフレーム。

    Returns:
        DataFrame: 生成された特徴量を含むデータフレーム。
    """
    columns = df.columns
    index = lambda x: columns.index(x)
    condition = [(pl.col(column) == target_row[index(column)]) for column in feat_pattern.keys()]
    hist = df.filter(pl.any(condition))
    def latest_newr(player):
        id = target_row[index(f'{player}_id')]
        player_newr = df.filter(pl.col(f'{player}_id') == id).select(f'{player}_newr')
        if 0 < player_newr.height:
            latest_player_newr = player_newr.row(-1)[0]
        else:
            latest_player_newr = 0.0
        return latest_player_newr
    
    players = ['horse', 'jockey', 'trainer']
    oldrs = [latest_newr(p) for p in players]
    feats = np.array([*target_row, *oldrs]) # 66 + 3
    remove = ['horse_newr', 'jockey_newr', 'trainer_newr']
    feat_columns = [c for c in df.columns if c not in remove] # 72 - 3
    for column in feat_pattern.keys():
        hist_target = hist.filter(pl.col(column) == target_row[index(column)])
        f_pattern = feat_pattern[column]
        funcs = list(f_pattern['by_race'].keys()) + list(f_pattern['by_month'].keys())
        past_columns = [f"{col}_{x}" for col in funcs for x in hist_pattern]
        feat_columns += past_columns
        if hist_target.height == 0:
            feat = np.empty(len(past_columns))
            feat[:] = np.nan
        else:
            row = target_row + hist_target.select('^.*newr$').row(-1)
            nphist_target = hist_target.select(pl.exclude('^.*newr$')).to_numpy()
            nphist_target = np.append(nphist_target, [row], axis=0)
            feat = agg_history_i(
                i=len(nphist_target)-1,
                f_byrace=list(f_pattern['by_race'].values()),
                f_bymonth=list(f_pattern['by_month'].values()),
                hist_pattern=hist_pattern,
                history=nphist_target,
                index=index,
            )
        feats = np.append(feats, feat)
    df_result = pd.DataFrame([feats], columns=feat_columns)
    return df_result

def rate_all(df):
    """
    競馬データの評価を行い、プレイヤーのレーティングを更新します。

    Args:
        df (DataFrame): 評価対象のデータフレーム。

    Returns:
        DataFrame: 更新されたレーティングを含むデータフレーム。
    """
    env = TrueSkill(draw_probability=0.000001)
    df_use = (
        df.select([
            pl.col('race_id'),
            pl.col('result').fill_null(18),
            pl.col(f'horse_id').fill_null(-1),
            pl.col(f'jockey_id').fill_null(-1),
            pl.col(f'trainer_id').fill_null(-1),
        ])
    )
    df_aggraces = (
        df_use.with_column(
            pl.when(pl.col('result').is_null()).then(0).otherwise(1).alias('weight')
        )
        .groupby('race_id', maintain_order=True)
        .agg_list()
    )

    def unique(c):
        return df_use.select(c).unique().to_series().to_list()
    
    players = ['horse', 'jockey', 'trainer']
    ratings = {
        player: {id: env.create_rating() for id in unique(f'{player}_id')}
        for player in players
    }
    data = {
        **{f'{player}_oldr': [] for player in players},
        **{f'{player}_newr': [] for player in players},
    }

    def update_ratings(player, player_ids, results, weights):
        weights = [(w, ) for w in weights]
        player_ratings = ratings[player]
        old_ratings= [(player_ratings[id], ) for id in player_ids]
        new_ratings= env.rate(old_ratings, ranks=results, weights=weights)
        for id, (old_r, ), (new_r, ) in zip(player_ids, old_ratings, new_ratings):
            ratings[player][id] = new_r
            if id == -1:
                data[f'{player}_oldr'].append(None)
                data[f'{player}_newr'].append(None)
            else:
                data[f'{player}_oldr'].append(env.expose(old_r))
                data[f'{player}_newr'].append(env.expose(new_r))
    for race_id, results, h_ids, j_ids, t_ids, weights in tqdm(df_aggraces.rows()):
        update_ratings('horse', h_ids, results, weights)
        update_ratings('jockey', j_ids, results, weights)
        update_ratings('trainer', t_ids, results, weights)
    return pl.DataFrame(data)


def calc(f, past_j, row, index, nan):
    """
    過去のデータに基づいて計算を行います。

    Args:
        f (function): 計算に使用する関数。
        past_j (array): 過去のデータ。
        row (array): 現在の行データ。
        index (function): 列名をインデックスに変換する関数。
        nan (float): NaN値。

    Returns:
        float: 計算結果またはNaN。
    """
    if past_j.any():
        return f(past_j, row, index)
    else:
        return nan

def agg_history_i(i, f_byrace, f_bymonth, hist_pattern, history, index, mo=timedelta(days=30)):
    """
    特定のインデックスに基づいて履歴データを集約します。

    Args:
        i (int): 集約対象のインデックス。
        f_byrace (list): レースごとの集約関数のリスト。
        f_bymonth (list): 月ごとの集約関数のリスト。
        hist_pattern (dict): 履歴パターンの辞書。
        history (array): 履歴データ。
        index (function): 列名をインデックスに変換する関数。
        mo (timedelta): 月の期間（デフォルトは30日）。

    Returns:
        array: 集約された履歴データ。
    """
    no_hist = np.empty(((len(f_byrace) + len(f_bymonth)) * len(hist_pattern),))
    no_hist[:] = np.nan
    row = history[i, :]
    hist_race = history[:, index('race_date')]
    now_race = row[index('race_date')]
    past_days = now_race - hist_race
    past_hist = history[np.where(hist_race < now_race)][::-1]
    zero_m = timedelta(days=0)
    if past_hist.any():
        try:
            last_jrace_fresult = [
                f(past_hist[:1+j, :], row, index)
                for f in f_byrace
                for j in hist_pattern
            ]
            last_jmonth_fresult = [
                calc(
                    f,
                    past_hist[np.where((zero_m < past_days) & (past_days <= j*mo))],
                    row,
                    index,
                    np.nan,
                )
                for f in f_bymonth
                for j in hist_pattern
            ]
            return np.array(last_jrace_fresult + last_jmonth_fresult)
        except Exception as e:
            print(traceback.format_exc())
            import pdb; pdb.set_trace()
            return no_hist
    else:
        return no_hist

def agg_history(f_pattern, hist_pattern, history, index):
    """
    履歴データを集約し、特徴量を計算します。

    Args:
        f_pattern (dict): 特徴量パターンの辞書。
        hist_pattern (dict): 履歴パターンの辞書。
        history (array): 履歴データ。
        index (function): 列名をインデックスに変換する関数。

    Returns:
        array: 集約された履歴データの配列。
    """
    result = []
    for i in range(0, len(history)):
        history_i = agg_history_i(
            i=i,
            f_byrace=list(f_pattern['by_race'].values()),
            f_bymonth=list(f_pattern['by_month'].values()),
            hist_pattern=hist_pattern,
            history=history,
            index=index,
        )
        result.append(history_i)
    return np.array(result)

def save_feat(player, feat_pattern, hist_pattern, df):
    """
    プレイヤーごとの特徴量をファイルに保存します。

    Args:
        player (str): プレイヤーの名前。
        feat_pattern (dict): 特徴量パターンの辞書。
        hist_pattern (dict): 履歴パターンの辞書。
        df (DataFrame): 特徴量を生成するためのデータフレーム。
    """
    name = df[0, player]
    try:
        name = int(name)
    except:
        return

    f_pattern = feat_pattern[player]
    df = df.sort('race_date')
    hist = df.to_numpy()
    columns = df.columns
    index = lambda x: columns.index(x)
    a_agghist = agg_history(f_pattern, hist_pattern, hist, index)

    all_hist = np.column_stack((hist, a_agghist))
    funcs = list(f_pattern['by_race'].keys()) + list(f_pattern['by_month'].keys())
    past_columns = [f"{col}_{x}" for col in funcs for x in hist_pattern]
    df_feat = pd.DataFrame(all_hist, columns=columns+past_columns)
    df_feat = (
        pl.from_pandas(df_feat)
        .with_column(
            pl.col([pl.Int8, pl.Int16, pl.Int32, pl.Int64]).cast(pl.Float64),
        )
    )
    df_feat.write_ipc(f'feat/{player}_{name}.feather')

def prepare():
    """
    データを読み込み、エンコードし、評価を行います。
    """
    print(f'loading {config.netkeiba_file}')
    df_original = pl.read_ipc(config.netkeiba_file)

    print('encoding')
    horse_encoder = HorseEncoder()
    df_encoded = horse_encoder.format_fit_transform(df_original)
    with open(config.encoder_file, 'wb') as f:
        pickle.dump(horse_encoder, f)

    df_avetime = (
        df_encoded.groupby(['field', 'distance', 'field_condition'])
        .agg(pl.mean('time').alias('avetime'))
    )
    df_avetime.write_ipc(config.avetime_file)
    df_encoded = df_encoded.join(
        df_avetime,
        on=['field', 'distance', 'field_condition'],
        how='left',
    )

    print('rating')
    df_ratings = rate_all(df_encoded)
    df_ratings.write_ipc(config.rating_file)
    df_encoded = pl.concat([df_encoded, df_ratings], how='horizontal')
    hist_pattern = featlist.hist_pattern
    feat_pattern = featlist.feature_pattern
    players = list(feat_pattern.keys())

    for player in players:
        n_players = pl.n_unique(df_encoded[player])
        save_player_feat = functools.partial(save_feat, player, feat_pattern, hist_pattern)
        print(player)
        for df in tqdm(df_encoded.groupby(player), total=n_players):
            save_player_feat(df)

def out():
    """
    特徴量を結合し、最終的なデータセットを保存します。
    """
    feat_pattern = featlist.feature_pattern
    cols = list(feat_pattern.keys())

    dfs = {}
    for column in cols:
        print(column)
        dfs[column] = (
            pl.scan_ipc(f'feat/{column}_*.feather')
            .fill_null(strategy="zero")
            .sort(['race_date', 'race_id', 'horse_no'])
            .collect()
        )
        print(f'{column} concatted')
        print(dfs[column].select(['year', 'race_date', pl.col('race_id').cast(pl.Int64), 'horse_no']))
    df = dfs[cols[0]]
    for column in cols[1:]:
        join_ids = ['race_date', 'race_id', 'horse_no']
        cols_to_use = list(set(dfs[column].columns) - set(df.columns)) + join_ids
        df = df.join(dfs[column].select(cols_to_use), on=join_ids, how='left')

    print("downcasting")
    pd_feat = df.with_column(pl.col('start_time').dt.strftime("%H:%M")).to_pandas()
    df_fillna = pd_feat.fillna(-1, downcast='infer')
    fcols = df_fillna.select_dtypes('float').columns
    icols = df_fillna.select_dtypes('integer').columns
    df_fillna[fcols] = df_fillna[fcols].apply(pd.to_numeric, downcast='float')
    df_fillna[icols] = df_fillna[icols].apply(pd.to_numeric, downcast='integer')
    print(df_fillna.dtypes)
    print(f"save to {config.feat_file}")
    df = (
        pl.from_pandas(df_fillna)
        .with_column(
            pl.col('start_time').cast(str).str.strptime(pl.Datetime, "%H:%M")
        )
    )
    df.write_ipc(config.feat_file)

if __name__ == '__main__':
    args = parse_args()
    if args.prepare:
        prepare()
    if args.out:
        out()