import pdb
import polars as pl
from itertools import combinations
from pathlib import Path
from tqdm import tqdm
import pickle

import config
import predict


PROGRESS_FILE = "temp/racefeat_progress.pkl"  # 進捗を保存するファイル
SAVE_INTERVAL = 10  # 10レースごとに保存

# 進捗状況を保存
def save_progress(race_features, race_bakenhits, processed_races):
    with open(PROGRESS_FILE, "wb") as f:
        pickle.dump((race_features, race_bakenhits, processed_races), f)

# 進捗状況を読み込み
def load_progress():
    if Path(PROGRESS_FILE).exists():
        with open(PROGRESS_FILE, "rb") as f:
            return pickle.load(f)
    return [], [], set()  # 進捗がない場合の初期値

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
    hit = {}

    # 単勝: 一致するかをチェック
    hit['単勝'] = 1 if predicted['単勝'] == actual['単勝'] else 0

    # 複勝: 任意の値が一致すればOK
    hit['複勝'] = len(set(predicted['複勝']) & set(actual['複勝']))

    # ワイド: タプルのいずれかが一致すればOK
    predicted_wide = set([tuple(sorted(pair)) for pair in predicted['ワイド']])
    actual_wide = set([tuple(sorted(pair)) for pair in actual['ワイド']])
    hit['ワイド'] = len(predicted_wide & actual_wide)

    # 馬連: タプルが完全一致するか
    predicted_umaren = set([tuple(sorted(pair)) for pair in predicted['馬連']])
    actual_umaren = set([tuple(sorted(pair)) for pair in actual['馬連']])
    hit['馬連'] = 1 if predicted_umaren == actual_umaren else 0

    # 馬単: 順序が重要なのでそのまま比較
    predicted_umatan = set([tuple(pair) for pair in predicted['馬単']])
    actual_umatan = set([tuple(pair) for pair in actual['馬単']])
    hit['馬単'] = 1 if predicted_umatan == actual_umatan else 0

    # 3連複: 順序関係なく3つのセットが一致するか
    predicted_sanrenpuku = set([tuple(sorted(triple)) for triple in predicted['3連複']])
    actual_sanrenpuku = set([tuple(sorted(triple)) for triple in actual['3連複']])
    hit['3連複'] = 1 if predicted_sanrenpuku == actual_sanrenpuku else 0

    # 3連単: 順序も含めて完全一致するか
    predicted_sanrentan = set([tuple(triple) for triple in predicted['3連単']])
    actual_sanrentan = set([tuple(triple) for triple in actual['3連単']])
    hit['3連単'] = 1 if predicted_sanrentan == actual_sanrentan else 0

    return sum(x for x in hit.values())

if __name__ == "__main__":
    if Path(config.racefeat_file).exists():
        print(f"Skip racefeat. Already exists {config.racefeat_file}.")
        df_race = pl.read_ipc(config.racefeat_file)
    else:
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

        # ビンの数を設定
        bin_count = 10
        breakpoints = {}

        # 各カラムのヒストグラムのブレークポイントを計算
        for col in tqdm(df_x.columns, desc="Calculating breakpoints for features"):
            breakpoints[col] = df_x[col].hist(bin_count=bin_count)["breakpoint"].to_list()[:-1]

        # 進捗のロード
        race_features, race_bakenhits, processed_races = load_progress()
        processed_count = 0  # 処理されたレースのカウント

        # 各レース日ごとにデータフレームをグループ化
        for racedate, df in tqdm(df_feat.group_by(config.RACEDATE_COLUMNS), total=len(df_feat.group_by(config.RACEDATE_COLUMNS).len()), desc="Processing races"):
            race_id = str(df.get_column("race_id")[0])

            # すでに処理済みのレースはスキップ
            if race_id in processed_races:
                continue

            race_dict = {}

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
            race_features.append(race_dict)

            # 対応する出馬データ
            race_id = str(df.get_column("race_id")[0])
            df_shutsuba = df_netkeiba.filter(pl.col("race_id") == race_id)

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
            race_bakenhits.append(bakenhit)

            # 処理済みのレースIDを保存
            processed_races.add(race_id)
            processed_count += 1

            # 進捗状況の保存
            if processed_count % SAVE_INTERVAL == 0:
                save_progress(race_features, race_bakenhits, processed_races)

        df_race = pl.DataFrame(race_features).with_columns(pl.Series(race_bakenhits).alias("hit"))

        if not Path(config.racefeat_file).exists():
            df_race.write_ipc(config.racefeat_file)

        # 最終結果保存後に進捗ファイルを削除
        if Path(PROGRESS_FILE).exists():
            Path(PROGRESS_FILE).unlink()
