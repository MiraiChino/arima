import pdb
import polars as pl
from itertools import combinations
from pathlib import Path
from tqdm import tqdm
import pickle

import config
import predict
import train

PROCESSED_RACES_FILE = "temp/processed_races.pickle"
SAVE_INTERVAL = 1000  # 1000レースごとに保存

# 進捗状況を保存
def save_progress(race_features_df, processed_races):
    race_features_df.write_ipc(config.racefeat_file)  # config.racefeat_fileに保存
    with open(PROCESSED_RACES_FILE, 'wb') as f:
        pickle.dump(processed_races, f)  # processed_racesをpickleファイルに保存

# 進捗状況を読み込み
def load_progress():
    race_features_df = pl.DataFrame()  # 進捗がない場合の初期値
    processed_races = set()  # 初期のprocessed_races

    if Path(config.racefeat_file).exists() and Path(PROCESSED_RACES_FILE).exists():
        race_features_df = pl.read_ipc(config.racefeat_file)
        with open(PROCESSED_RACES_FILE, 'rb') as f:
            processed_races = pickle.load(f)

    return race_features_df, processed_races

def generate_baken(order):
    # 着順が3未満の場合のエラーチェック
    if len(order) < 3:
        raise ValueError("着順のリストは少なくとも3つの要素が必要です。")

    # 単勝: 1着馬
    tansho = [order[0]]

    # 複勝: 順不同の上位3頭（着順の順序は問わない）
    fukusho = sorted(order[:3])

    # ワイド: 上位3頭の組み合わせ（順不同）
    wide = [tuple(sorted(pair)) for pair in combinations(order[:3], 2)]

    # 馬連: 上位2頭の組み合わせ（順不同）
    umaren = [sorted(order[:2])]

    # 馬単: 上位2頭の順序を含む組み合わせ（着順通り）
    umatan = [order[:2]]

    # 3連複: 上位3頭の順不同組み合わせ
    sanrenpuku = [sorted(order[:3])]

    # 3連単: 上位3頭の順序（着順通り）
    sanrentan = [order[:3]]

    return {
        "単勝": tansho,
        "複勝": fukusho,
        "ワイド": wide,
        "馬連": umaren,
        "馬単": umatan,
        "3連複": sanrenpuku,
        "3連単": sanrentan,
    }

def baken_hit(predicted, actual):
    hit = {
        '単勝': int(predicted['単勝'] == actual['単勝']),
        '複勝': len(set(predicted['複勝']) & set(actual['複勝'])),
        'ワイド': len(set(map(tuple, map(sorted, predicted['ワイド']))) & set(map(tuple, map(sorted, actual['ワイド'])))),
        '馬連': int(set(map(tuple, map(sorted, predicted['馬連']))) == set(map(tuple, map(sorted, actual['馬連'])))),
        '馬単': int(set(map(tuple, predicted['馬単'])) == set(map(tuple, actual['馬単']))),
        '3連複': int(set(map(tuple, map(sorted, predicted['3連複']))) == set(map(tuple, map(sorted, actual['3連複'])))),
        '3連単': int(set(map(tuple, predicted['3連単'])) == set(map(tuple, actual['3連単'])))
    }
    return sum(hit.values())

def process_race(df, df_shutsuba, breakpoints, bin_count=10):
    race_dict = {}
    race_id = str(df.get_column("race_id")[0])

    # 各カラムのビンを生成し、カウントを更新
    for col, breaks in breakpoints.items():
        # ラベルを初期化
        labels = [str(i) for i in range(bin_count)]
        for i in labels:
            race_dict[f"{col}_bin{i}"] = 0

        # データをビンにカット
        bins = df[col].cut(breaks, labels=labels)
        for i in bins:
            key = f"{col}_bin{i}"
            race_dict[key] += 1

    # 予測結果
    result_prob = predict.result_prob(df_shutsuba)

    # 予測結果と実際の結果のデータフレームを作成
    s_prob = pl.Series([v for k, v in sorted(result_prob.items())])
    df_results = df.select("horse_no", "result").with_columns(s_prob.alias("prob"))

    # 馬券の予測
    df_sorted = df_results.sort(by='prob', descending=True)
    predicted = df_sorted.get_column("result").to_list()
    actual = df_shutsuba.get_column("horse_no").to_list()
    baken_predicted = generate_baken(predicted)
    baken_actual = generate_baken(actual)
    
    # 馬券の的中数
    bakenhit = baken_hit(baken_predicted, baken_actual)

    return race_dict, bakenhit, race_id

def racefeat():
    print(f'loading {config.netkeiba_file}')
    try:
        df_netkeiba = pl.read_ipc(config.netkeiba_file)
    except Exception as e:
        print(f"Error loading file: {e}")
        pdb.set_trace()

    print(f'loading {config.feat_file}')
    df_feat = pl.read_ipc(config.feat_file)
    # 'start_time'を時間フォーマットに変換
    df_feat = df_feat.with_columns(pl.col('start_time').dt.strftime("%H:%M"))
    # 不要なカラムを除外して新しいデータフレームを作成
    df_x = df_feat.select(pl.exclude(config.NONEED_COLUMNS))

    # 進捗のロード
    race_features_df, processed_races = load_progress()

    # ビンの数を設定
    bin_count = 10
    breakpoints = {}

    # 各カラムのヒストグラムのブレークポイントを計算
    for col in tqdm(df_x.columns, desc="Calculating breakpoints for features"):
        breakpoints[col] = df_x[col].hist(bin_count=bin_count)["breakpoint"].to_list()[:-1]

    # 各レース日ごとにデータフレームをグループ化
    races = df_feat.group_by(config.RACEDATE_COLUMNS)

    # モデルなどをロード
    predict.load_models_and_configs()

    # 1レースずつ特徴量と馬券の的中数を計算
    # 並列化するとモデルのロードがボトルネックになるため遅くなった
    for racedate, df in tqdm(races, total=len(races.len()), desc="Processing races"):
        race_id = str(df.get_column("race_id")[0])

        # すでに処理済みのレースはスキップ
        if race_id in processed_races:
            continue

        # 対応する出馬データ
        df_shutsuba = df_netkeiba.filter(pl.col("race_id") == race_id)

        race_dict, bakenhit, race_id = process_race(df, df_shutsuba, breakpoints, bin_count)
        race_dict['bakenhit'] = bakenhit  # bakenhitをrace_dictに追加
        race_features_df = race_features_df.vstack(pl.DataFrame([race_dict]))
        processed_races.add(race_id)
        if len(processed_races) % SAVE_INTERVAL == 0:
            save_progress(race_features_df, processed_races)

    # 最後に全ての進捗を保存
    save_progress(race_features_df, processed_races)
    print(f"Final progress saved for all races.")

    return df_race

if __name__ == "__main__":
    df_race = racefeat()

    # 馬券の的中率を予測する
    bakenhit_model, _ = train.lgb(df=df_race, config=config.bakenhit_lgb_reg)
