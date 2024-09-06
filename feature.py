import argparse
import functools
import pdb
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import dill as pickle
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

import config
import featlist

def parse_args():
    """
    コマンドライン引数を解析します。

    Returns:
        Namespace: 解析された引数を含むオブジェクト。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--out', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Run the script without making any file changes')
    return parser.parse_args()

@lru_cache(maxsize=None)
def get_player_newr(df, player, id):
    """
    プレイヤーの最新のレーティングを取得します（キャッシュを使用）。

    Args:
        df (DataFrame): データフレーム。
        player (str): プレイヤーの種類（例：'horse', 'jockey', 'trainer'）。
        id (int): プレイヤーのID。

    Returns:
        float: 最新のレーティング。
    """
    player_newr = df.filter(pl.col(f'{player}_id') == id).select(f'{player}_newr')
    if player_newr.height == 0:
        return 0.0
    try:
        return player_newr.row(-1)[0]
    except IndexError as e:
        print(f"Error in get_player_newr: {e}")
        pdb.set_trace()

def latest_newr(player, target_row, index, df):
    """
    ターゲット行のプレイヤーの最新のレーティングを取得します。

    Args:
        player (str): プレイヤーの種類（例：'horse', 'jockey', 'trainer'）。
        target_row (array): ターゲット行データ。
        index (function): 列名をインデックスに変換する関数。
        df (DataFrame): データフレーム。

    Returns:
        float: 最新のレーティング。
    """
    try:
        id = target_row[index(f'{player}_id')]
    except IndexError as e:
        print(f"Error in latest_newr: {e}")
        pdb.set_trace()
    return get_player_newr(df, player, id)

def pre_aggregate_history(history, index, f_byrace, f_bymonth, hist_pattern, mo=timedelta(days=30)):
    """
    履歴データを事前に集約します。

    Args:
        history (array): 履歴データ。
        index (function): 列名をインデックスに変換する関数。
        f_byrace (list): レースごとの関数リスト。
        f_bymonth (list): 月ごとの関数リスト。
        hist_pattern (list): 履歴パターン。
        mo (timedelta): 月の期間。

    Returns:
        dict: 事前に集約されたデータ。
    """
    aggregated_history = {}
    hist_race = history[:, index('race_date')]
    
    if len(history) == 0:
        return {}
    
    for i, row in enumerate(history):
        now_race = row[index('race_date')]
        past_hist = history[:i+1][::-1]
        
        last_jrace_fresult = np.concatenate([
            [f(past_hist[:1+j, :], row, index) for j in hist_pattern]
            for f in f_byrace
        ])
        
        try:
            last_jmonth_fresult = np.concatenate([
                [calc(
                    f,
                    past_hist[(timedelta(days=0) < (now_race - hist_race[:i+1])) & ((now_race - hist_race[:i+1]) <= j*mo)],
                    row,
                    index,
                    np.nan,
                ) for j in hist_pattern]
                for f in f_bymonth
            ])
        except Exception as e:
            print(f"Error in pre_aggregate_history: {e}")
            pdb.set_trace()
        
        aggregated_history[i] = np.concatenate([last_jrace_fresult, last_jmonth_fresult])
    
    return aggregated_history

def efficient_agg_history_i(i, f_byrace, f_bymonth, hist_pattern, history, index, mo=timedelta(days=30), aggregated_history=None):
    """
    履歴データを効率的に集約します。

    Args:
        i (int): インデックス。
        f_byrace (list): レースごとの関数リスト。
        f_bymonth (list): 月ごとの関数リスト。
        hist_pattern (list): 履歴パターン。
        history (array): 履歴データ。
        index (function): 列名をインデックスに変換する関数。
        mo (timedelta): 月の期間。
        aggregated_history (dict): 事前に集約されたデータ。

    Returns:
        array: 集約結果。
    """
    if aggregated_history is None:
        aggregated_history = pre_aggregate_history(history, index, f_byrace, f_bymonth, hist_pattern, mo)
    
    return aggregated_history[i]

def agg_history(f_pattern, hist_pattern, history, index):
    """
    履歴データを集約します。

    Args:
        f_pattern (dict): 特徴量パターン。
        hist_pattern (list): 履歴パターン。
        history (array): 履歴データ。
        index (function): 列名をインデックスに変換する関数。

    Returns:
        array: 集約結果。
    """
    f_byrace = list(f_pattern['by_race'].values())
    f_bymonth = list(f_pattern['by_month'].values())
    aggregated_history = pre_aggregate_history(history, index, f_byrace, f_bymonth, hist_pattern)
    
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(
            lambda i: efficient_agg_history_i(
                i=i,
                f_byrace=f_byrace,
                f_bymonth=f_bymonth,
                hist_pattern=hist_pattern,
                history=history,
                index=index,
                aggregated_history=aggregated_history
            ),
            range(len(history))
        ))
    
    return np.array(results)

def save_feat(player, feat_pattern, hist_pattern, df, dry_run=False):
    """
    特徴量を保存します。

    Args:
        player (str): プレイヤーの種類（例：'horse', 'jockey', 'trainer'）。
        feat_pattern (dict): 特徴量パターン。
        hist_pattern (list): 履歴パターン。
        df (DataFrame): データフレーム。
        dry_run (bool): ファイルの追加・削除を実行しないモード。
    """
    name = df[0, player]

    if name is None:
        return
    elif isinstance(name, float):
        name = int(name)

    f_pattern = feat_pattern[player]
    try:
        df = df.sort('race_date')
    except Exception as e:
        print(f"Error sorting DataFrame: {e}")
        pdb.set_trace()
    try:
        hist = df.to_numpy()
    except Exception as e:
        print(f"Error converting DataFrame to numpy array: {e}")
        pdb.set_trace()
    columns = df.columns
    index = lambda x: columns.index(x)
    a_agghist = agg_history(f_pattern, hist_pattern, hist, index)

    all_hist = np.column_stack((hist, a_agghist))
    funcs = list(f_pattern['by_race'].keys()) + list(f_pattern['by_month'].keys())
    past_columns = [f"{col}_{x}" for col in funcs for x in hist_pattern]
    df_feat = pd.DataFrame(all_hist, columns=columns+past_columns)
    df_feat = (
        pl.from_pandas(df_feat)
        .with_columns(
            pl.col([pl.Int8, pl.Int16, pl.Int32, pl.Int64]).cast(pl.Float64),
        )
    )
    if not dry_run:
        df_feat.write_ipc(f'feat/{player}_{name}.feather')

def search_history(target_row, hist_pattern, feat_pattern, df):
    """
    履歴データを検索し、特徴量を生成します。

    Args:
        target_row (array): ターゲット行データ。
        hist_pattern (list): 履歴パターン。
        feat_pattern (dict): 特徴量パターン。
        df (DataFrame): データフレーム。

    Returns:
        DataFrame: 特徴量を含むデータフレーム。
    """
    columns = df.columns
    index = lambda x: columns.index(x)
    try:
        condition = [(pl.col(column) == target_row[index(column)]) for column in feat_pattern.keys()]
    except IndexError as e:
        print(f"Error in search_history: {e}")
        pdb.set_trace()
    hist = df.filter(pl.any(condition))
    
    players = ['horse', 'jockey', 'trainer']
    oldrs = [latest_newr(p, target_row, index, df) for p in players]
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
            feat = efficient_agg_history_i(
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
    from trueskill import TrueSkill
    env = TrueSkill(draw_probability=0.000001)
    try:
        df_use = (
            df.select([
                pl.col('race_id'),
                pl.col('result').fill_null(18),
                pl.col(f'horse_id').fill_null(-1),
                pl.col(f'jockey_id').fill_null(-1),
                pl.col(f'trainer_id').fill_null(-1),
            ])
        )
    except Exception as e:
        print(f"Error selecting columns: {e}")
        pdb.set_trace()
    df_aggraces = (
        df_use.with_columns(
            pl.when(pl.col('result').is_null()).then(0).otherwise(1).alias('weight')
        )
        .group_by('race_id', maintain_order=True).all()
    )

    def unique(c):
        return df_use.select(c).unique().to_series().to_list()
    
    players = ['horse', 'jockey', 'trainer']
    unique_ids = {player: unique(f'{player}_id') for player in players}
    ratings = {
        player: {id: env.create_rating() for id in unique_ids[player]}
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
        try:
            new_ratings = env.rate(old_ratings, ranks=results, weights=weights)
        except Exception as e:
            print(f"Error updating ratings: {e}")
            pdb.set_trace()
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
    try:
        if past_j.any():
            return f(past_j, row, index)
    except Exception as e:
        print(f"Error in calc function: {e}")
        pdb.set_trace()
    return nan

def prepare(dry_run=False):
    """
    データを読み込み、エンコードし、評価を行います。

    Args:
        dry_run (bool): ファイルの追加・削除を実行しないモード。
    """
    print(f'loading {config.netkeiba_file}')
    try:
        df_original = pl.read_ipc(config.netkeiba_file)
    except Exception as e:
        print(f"Error loading file: {e}")
        pdb.set_trace()

    print('encoding')
    from encoder import HorseEncoder
    horse_encoder = HorseEncoder()
    df_encoded = horse_encoder.format_fit_transform(df_original)
    if not dry_run:
        try:
            with open(config.encoder_file, 'wb') as f:
                pickle.dump(horse_encoder, f)
        except Exception as e:
            print(f"Error saving encoder: {e}")
            pdb.set_trace()
    else:
        print(f"dry-run: would save encoder to {config.encoder_file}")

    df_avetime = (
        df_encoded.group_by(['field', 'distance', 'field_condition'])
        .agg(pl.mean('time').alias('avetime'))
    )
    if not dry_run:
        df_avetime.write_ipc(config.avetime_file)
    else:
        print(f"dry-run: would save avetime to {config.avetime_file}")

    df_encoded = df_encoded.join(
        df_avetime,
        on=['field', 'distance', 'field_condition'],
        how='left',
    )

    # rating_fileが存在するか確認
    print('rating')
    if Path(config.rating_file).exists():
        print(f"Skip rating. Already exists {config.rating_file}.")
        df_ratings = pl.read_ipc(config.rating_file)
    else:
        df_ratings = rate_all(df_encoded)
        if not dry_run:
            df_ratings.write_ipc(config.rating_file)
        else:
            print(f"dry-run: would save ratings to {config.rating_file}")
    
    df_encoded = pl.concat([df_encoded, df_ratings], how='horizontal')
    hist_pattern = featlist.hist_pattern
    feat_pattern = featlist.feature_pattern
    players = list(feat_pattern.keys())

    # 過去のレース日を追跡
    existing_horse_latest_race = {}
    for player in players:
        existing_horse_latest_race[player] = {}
        feat_files = list(Path('feat').glob(f'{player}_*.feather'))
        for feat_file in tqdm(feat_files, desc=f'Loading {player}', total=len(feat_files)):
            existing_feats = pl.read_ipc(feat_file)
            for row in existing_feats.iter_rows(named=True):
                horse_id = row['horse_id']
                race_date = row['race_date']
                if horse_id not in existing_horse_latest_race[player]:
                    existing_horse_latest_race[player][horse_id] = race_date
                else:
                    existing_horse_latest_race[player][horse_id] = max(existing_horse_latest_race[player][horse_id], race_date)

    # 最新のレースに出た馬、騎手、調教師の特徴量を計算
    try:
        for player in players:
            n_players = df_encoded.n_unique(player)
            save_player_feat = functools.partial(save_feat, player, feat_pattern, hist_pattern, dry_run=dry_run)

            for name, df in tqdm(df_encoded.group_by(player), desc=f'feat {player}', total=n_players):
                new_horses = []
                for row in df.iter_rows(named=True):
                    horse_id = row['horse_id']
                    race_date = row['race_date']
                    if horse_id not in existing_horse_latest_race[player] or (race_date and race_date > existing_horse_latest_race[player].get(horse_id)):
                        new_horses.append(horse_id)
                if new_horses:
                    save_player_feat(df.filter(pl.col('horse_id').is_in(new_horses)))
    except Exception as e:
        print(f"Error in prepare: {traceback.format_exc()}")
        pdb.set_trace()

def out(dry_run=False):
    """
    特徴量を結合し、最終的なデータセットを保存します。

    Args:
        dry_run (bool): ファイルの追加・削除を実行しないモード。
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
    try:
        df = dfs[cols[0]]
        for column in cols[1:]:
            join_ids = ['race_date', 'race_id', 'horse_no']
            cols_to_use = list(set(dfs[column].columns) - set(df.columns)) + join_ids
            df = df.join(dfs[column].select(cols_to_use), on=join_ids, how='left')
    except Exception as e:
        print(f"Error joining DataFrames: {e}")
        pdb.set_trace()

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
    try:
        if not dry_run:
            df.write_ipc(config.feat_file)
        else:
            print(f"dry-run: would save final dataframe to {config.feat_file}")
    except Exception as e:
        print(f"Error saving file: {e}")
        pdb.set_trace()

if __name__ == '__main__':
    args = parse_args()
    if args.prepare:
        prepare(dry_run=args.dry_run)
    if args.out:
        out(dry_run=args.dry_run)