import argparse
import traceback
from datetime import timedelta
from pathlib import Path

import dill as pickle
import polars as pl
from tqdm import tqdm
from trueskill import TrueSkill

import config
import utils
from encoder import HorseEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--update', action='store_true')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--out', action='store_true')
    return parser.parse_args()

# def ave(column):
#     def wrapper(df_past=None, now=None, i=None):
#         return pl.col(column).mean()
#     return wrapper

# def same_count(column):
#     def wrapper(df_past=None, now=None, i=None):
#         return (pl.col(column) == now[column]).sum()
#     return wrapper

# def diff(col1, col2):
#     def wrapper(df_past=None, now=None, i=None):
#         return ((pl.col(col1) - pl.col(col2)) / pl.col(col2)).mean()
#     return wrapper

# def div(col1, col2):
#     def wrapper(df_past=None, now=None, i=None):
#         return (pl.col(col1) / pl.col(col2)).mean()
#     return wrapper

# def same_ave(*columns, target='prize'):
#     def wrapper(df_past=None, now=None, i=None):
#         df_temp = df_past.select(pl.col(target).mean().over(columns).suffix('_ave'))
#         return df_temp.get_column(f'{target}_ave').take(i)
#     return wrapper


# def drize(df_past=None, now=None, i=None):
#     now = df[i, 'distance']
#     return diff_multi('distance', 'prize', now).mean()

# def same_drize(*columns):
#     def wrapper(df_past=None, now=None, i=None):
#         now = df[i, 'distance']
#         df_temp = df.select(diff_multi('distance', 'prize', now).mean().over(columns).alias('drize_ave'))
#         return df_temp.get_column('drize_ave').take(i)
#     return wrapper

def search_history(target_row, hist_pattern, feat_pattern, df):
    row_df = pd.DataFrame([target_row]).reset_index()
    if 'id' not in row_df.columns:
        row_df['id'] = None
    if 'index' not in row_df.columns:
        row_df['index'] = None
    hist_df = row_df.copy(deep=True)
    condition = ' or '.join(f'{column}=={target_row[column]}' for column in feat_pattern.keys())
    condition_df = df.query(condition)

    for column in feat_pattern.keys():
        funcs = list(feat_pattern[column].values())
        past_columns = [f'{col}_{x}' for col in list(feat_pattern[column].keys()) for x in hist_pattern]
        hist = condition_df.query(f'{column}=={target_row[column]}')
        hist = pd.concat([hist, row_df])
        a, columns, index = df2np(hist)
        targetrow_agg = agg_history_i(len(a)-1, funcs, hist_pattern, a, index)
        row_df_column = pd.DataFrame([targetrow_agg], columns=past_columns)
        hist_df = pd.concat([hist_df, row_df_column], axis='columns')
        # TODO: ratingも追加
        sorted_rating = hist.sort_values('race_id', ascending=False)
        import pdb; pdb.set_trace()
        latest_rating = sorted_rating[f'{column}_rnew'][0]
    return hist_df

def rate_all(df):
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

def ave(column):
    def wrapper(df=None, i=None):
        return pl.col(column).mean()
    return wrapper

def same_count(column):
    def wrapper(df=None, i=None):
        return (pl.col(column) == df[i, column]).sum()
    return wrapper

def diff(col1, col2):
    def wrapper(df=None, i=None):
        return ((pl.col(col1) - pl.col(col2)) / pl.col(col2)).mean()
    return wrapper

def div(col1, col2):
    def wrapper(df=None, i=None):
        return (pl.col(col1) / pl.col(col2)).mean()
    return wrapper
    
def same_ave(*columns, target='prize'):
    def wrapper(df=None, i=None):
        same_condition = pl.fold(
            acc=pl.lit(True),
            f=lambda a, b: a & b,
            exprs=[(pl.col(c) == df[i, c]) for c in columns]
        )
        return (same_condition * pl.col(target)).mean()
    return wrapper

def diff_multi(column, target, now):
    return (1 - abs(pl.col(column) - now) / now) * pl.col(target)

def drize(target='prize'):
    def wrapper(df=None, i=None):
        return diff_multi('distance', target, df[i, 'distance']).mean()
    return wrapper

def same_drize(*columns, target='prize'):
    def wrapper(df=None, i=None):
        same_condition = pl.fold(
            acc=pl.lit(True),
            f=lambda a, b: a & b,
            exprs=[(pl.col(c) == df[i, c]) for c in columns]
        )
        return (same_condition * diff_multi('distance', target, df[i, 'distance'])).mean()
    return wrapper

def featcols_past_nmonth(df, past_n, f_pattern, mo=timedelta(days=30)):
    feats = []
    for i in range(df.height):
        try:
            feat = (
                df.lazy()
                .with_column((pl.col('past_race') - pl.col('race_date')).alias('past'))
                .head(i)    # 過去レースをスライス
                .filter(pl.col('past') <= past_n * mo)
                .select([
                    expr(df, i).alias(f'{col}_{past_n}mo')
                    for col, expr in f_pattern.items()
                    # ave('result').alias(f'horse_result_{past_n}mo'),
                    # same_count(df, i, 'weather').alias(f'horse_weather_{past_n}mo'),
                    # diff('time', 'avetime').alias(f'horse_time_{past_n}mo'),
                    # div('prize', 'interval').alias(f'horse_iprize_{past_n}mo'),
                    # same_ave(df, i, 'place_code', target='prize').alias(f'horse_pprize_{past_n}mo'),
                    # drize(df, i, target='prize').alias(f'horse_drize_{past_n}mo'),
                    # same_drize(df, i, 'place_code', target='prize').alias(f'horse_pdrize_{past_n}mo'),
                ])
            )
        except:
            import traceback
            print(traceback.format_exc())
            import pdb; pdb.set_trace()
        feats.append(feat)
    return pl.concat(feats).collect()

def featcols_past_nrace(df, past_n, f_pattern):
    return (
        df.reverse()
        .with_column(pl.col('race_date').cumcount().cast(pl.Int64).alias('idx'))
        .groupby_dynamic('idx', every='1i', period=f'{past_n}i')
        .agg([expr().alias(f'{col_name}_{past_n}') for col_name, expr in f_pattern.items()])
        .reverse()
        .select([f'{col_name}_{past_n}' for col_name in f_pattern.keys()])
    )

def prepare():
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
    df_ratings = pl.read_ipc(config.rating_file)
    df_ratings = rate_all(df_encoded)
    df_ratings.write_ipc(config.rating_file)
    df_encoded = pl.concat([df_encoded, df_ratings], how='horizontal')
    hist_pattern = config.hist_pattern
    feat_pattern = config.feature_pattern
    players = list(feat_pattern.keys())
    
    for player in players:
        print(player)
        n_players = pl.n_unique(df_encoded[player])
        for df_player in tqdm(df_encoded.groupby(player, maintain_order=True), total=n_players):
            f_pattern = feat_pattern[player]
            df_with_interval = df_player.with_columns([
                pl.col('race_date').diff().dt.days().fill_null(0).alias('interval'),
                pl.col('race_date').shift().alias('past_race'),
            ])
            df_feat_byrace = [featcols_past_nrace(df_with_interval, n, f_pattern['by_race']) for n in hist_pattern]
            df_feat_bymonth = [featcols_past_nmonth(df_with_interval, n, f_pattern['by_month']) for n in hist_pattern]
            df_feat = pl.concat(
                [
                    df_with_interval,
                    *df_feat_byrace,
                    *df_feat_bymonth,
                ],
                how='horizontal'
            )
            name = df_player[0, player]
            df_feat.write_ipc(f'feat/{player}_{int(name)}.feather')

def out():
    feat_pattern = config.feature_pattern
    cols = list(feat_pattern.keys())

    df_feats = {}
    all_feat_files = [p for p in sorted(Path('feat').iterdir(), key=lambda p: p.stat().st_mtime) if p.suffix == '.feather']
    for column in cols:
        feat_files = [str(p) for p in all_feat_files if column in p.name]
        dfs = []
        print(column)
        for file in tqdm(feat_files):
            # with open(file, 'rb') as f:
            #     df_chunk = pickle.load(f)
            df_chunk = pd.read_feather(file)
            dfs.append(df_chunk)
        df_feats[column] = pd.concat(dfs)
        print(f'{column} concatted')
        df_feats[column] = pd.concat([df.sort_values('horse_no') for _, df in df_feats[column].groupby(['race_date', 'race_id'])])
        df_feats[column] = df_feats[column].reset_index(drop=True)
        df_feats[column]['id'] = df_feats[column].index
        print(df_feats[column][['id', 'year', 'race_date', 'race_id', 'horse_no']])
    df_feat = df_feats[cols[0]]
    for column in cols[1:]:
        cols_to_use = df_feats[column].columns.difference(df_feat.columns).tolist() + ['id']
        df_feat = pd.merge(df_feat, df_feats[column][cols_to_use], on='id')
    df_feat = utils.reduce_mem_usage(df_feat)
    df_feat.to_feather(config.feat_file)

def update():
    if not Path('netkeiba.log').exists():
        print(f'no {config.encoder_file}')
        return
    with open('netkeiba.log', 'r') as f:
        filenames = f.readlines()

    dfs = []
    for filename in filenames:
        df_races = pd.read_feather(f'{filename}.races.feather')
        df_horses = pd.read_feather(f'{filename}.horses.feather')
        df = pd.merge(df_horses, df_races, on='race_id', how='left')
        dfs.append(df)
    df_original = pd.concat(dfs)
    if 'index' in df_original.columns:
        df_original = df_original.drop(columns='index')

    if not Path(config.encoder_file).exists():
        print(f'no {config.encoder_file}')
        return
    with open(config.encoder_file, 'rb') as f:
        horse_encoder = pickle.load(f)
    df_format = horse_encoder.format(df_original)
    df_encoded = horse_encoder.transform(df_format)

    # TODO: avetimeを再計算する
    df_avetime = pd.read_feather(config.avetime_file)
    ave_time = {(f, d, fc): t for f, d, fc, t in df_avetime.to_dict(orient='split')['data']}
    feat_pattern = config.feature_pattern
    cols = list(feat_pattern.keys())
    for column in cols:
        print(column)
        past_columns = [f'{col}_{x}' for col in list(feat_pattern[column].keys()) for x in config.hist_pattern]
        funcs = list(feat_pattern[column].values())

        for name in tqdm(df_encoded[column].unique()):
            try:
                print(f'{column}_{int(name)}')
                # TODO: なかったら新しく作る
                df_feat = pd.read_feather(f'feat/{column}_{int(name)}.feather')
                pre_hist = df_feat.drop(columns=past_columns)
                # TODO: 必ず追加するようになっているので、同じ行があったら削除する
                # TODO: 同じ馬を何回も計算してる
                # TODO: 馬１匹のfeatを１つ減らしてためす
                rows = df_encoded.query(f'{column}=={name}')
                hist = pd.concat([pre_hist, rows])
                hist_nodup = hist.drop_duplicates(subset=['result', 'gate', 'horse_no', 'name', 'race_date', 'prize'])
                print(len(hist_nodup), len(pre_hist))
                if len(hist_nodup) <= len(pre_hist):
                    continue
                print(len(hist_nodup), len(pre_hist))
                a, columns, index = df2np(hist_nodup)
                print(df_feat[['result', 'horse_no', 'name', 'prize', 'horse_prize_999999']])
                print(hist_nodup[['result', 'horse_no', 'name', 'prize']])
                import pdb; pdb.set_trace()
                # ここからは未確認
                for i in range(len(pre_hist), len(hist_nodup)):
                    targetrow_agg = agg_history_i(i, funcs, config.hist_pattern, a, index)
                    row_df_column = pd.DataFrame([targetrow_agg], columns=past_columns)
                    import pdb; pdb.set_trace()
                    # hist_df = pd.concat([hist_df, row_df_column], axis='columns')
                    # df_agg = search_history(row, config.hist_pattern, feat_pattern, df_hist)
                    # new_feat = pd.concat([df_feat, df_agg]).reset_index(drop=True)
            except:
                import traceback
                print(traceback.format_exc())
                print(f"didn't update feat/{column}_{name}.feather")
            else:
                pass
                # new_feat.to_feather(f'feat/{column}_{name}.feather')

if __name__ == '__main__':
    args = parse_args()
    if args.update:
        update()
    if args.prepare:
        prepare()
    if args.out:
        out()