import itertools
import pdb
import pickle
import pandas as pd
import polars as pl
import re
from pathlib import Path
from tqdm import tqdm

import config
import predict


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

def generate_baken(order):
    # 着順が3未満の場合のエラーチェック
    if len(order) < 3:
        raise ValueError("着順のリストは少なくとも3つの要素が必要です。")

    # 単勝: 1着馬
    tansho = [order[0]]

    # 複勝: 順不同の上位3頭（着順の順序は問わない）
    fukusho = sorted(order[:3])

    # ワイド: 上位3頭の組み合わせ（順不同）
    wide = [tuple(sorted(pair)) for pair in itertools.combinations(order[:3], 2)]

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

def process_race(df, df_shutsuba, prob):
    df = pl.DataFrame(df)
    try:
        # 重複を削除
        if len(df) == len(prob):
            df_filtered = df
        else:
            df_filtered = df.unique(subset="name", keep="first").sort(by="horse_no")

        # 予測結果と実際の結果のデータフレームを作成
        s_prob = pl.Series([v for k, v in sorted(prob.items())])
        df_results = df_filtered.select("horse_no", "result").with_columns(s_prob.alias("prob"))
    except:
        # デバッグ用の出力
        print(f"df length: {len(df)}")
        print(f"s_prob length: {len(s_prob)}")
        import pdb; pdb.set_trace()

    # 馬券の予測
    df_sorted = df_results.sort(by='prob', descending=True)
    predicted = df_sorted.get_column("result").to_list()
    baken_predicted = generate_baken(predicted)
    
    actual = df_shutsuba.get_column("horse_no").to_list()
    baken_actual = generate_baken(actual)
    
    # 馬券の的中数
    bakenhit = baken_hit(baken_predicted, baken_actual)

    return bakenhit

def use_columns_from_importance():
    models_dir = Path("models/")
    pattern = re.compile(rf"(\d+)_{config.bakenhit_lgb_reg.feature_importance_model.file}")
    # 最大のiを探す
    max_i = -1
    max_file = models_dir / f"0_{config.bakenhit_lgb_reg.feature_importance_model.file}"

    for file in models_dir.iterdir():  # フォルダ内の全ファイルを探索
        if file.is_file():  # ファイルのみを対象
            match = pattern.match(file.name)
            if match:
                i = int(match.group(1))  # iを数値として取得
                if i > max_i:
                    max_i = i
                    max_file = models_dir / file.name

    print(f"loading {max_file}")
    with open(f"{max_file}", "rb") as f:
        m = pickle.load(f)
        importance = pl.DataFrame({
            "column": m.feature_name(),
            "importance": m.feature_importance(importance_type='gain')
        }).sort(by="importance", descending=True)
        importance_head = importance.head(config.bakenhit_lgb_reg.feature_importance_len)
        print(importance_head)
        use_columns = importance_head.get_column("column").to_list()
    return use_columns

def calc_breakpoints(df_x, bin_count):
    if Path(config.breakpoint_file).exists():
        print(f'loading {config.breakpoint_file}')
        with open(config.breakpoint_file, "rb") as f:
            breakpoints = pickle.load(f)
    else:
        print(f'calculating {config.breakpoint_file}')
        breakpoints = {}

        # 各カラムのヒストグラムのブレークポイントを計算
        for col in tqdm(df_x.columns, desc="Calculating breakpoints for features"):
            breakpoints[col] = df_x[col].hist(bin_count=bin_count)["breakpoint"].to_list()[:-1]
        
        with open(config.breakpoint_file, "wb") as f:
            pickle.dump(breakpoints, f)
        print(f'saved {config.breakpoint_file}')
    return breakpoints

def save_racefeat():
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

    use_columns = use_columns_from_importance()
    df_x = df_x[use_columns]

    # ビンの数を設定
    bin_count = config.bakenhit_lgb_reg.bins
    breakpoints = calc_breakpoints(df_x, bin_count)

    # ジェネレータを使用してレースを処理
    group = df_feat.group_by(config.RACEDATE_COLUMNS)
    total = len(group.len())

    # モデルをロード
    print("loading models")
    predict.load_models_and_configs()
    schema = None

    for racedate, df in tqdm(group, total=total, desc="Processing races"):
        race_id = str(df.get_column("race_id")[0])
        feather_file = f"racefeat/{race_id}.feather"

        if Path(feather_file).exists():
            continue

        df_shutsuba = df_netkeiba.filter(pl.col("race_id") == race_id)

        # 予測結果
        df_feat = predict.search_df_feat(df_shutsuba)
        result_prob = predict.result_prob(df_feat)

        race_dict = predict.bin_race_dict(df_feat, breakpoints, bin_count)

        race_dict['race_id'] = str(df.get_column("race_id")[0])
        race_dict['race_date'] = str(df.get_column("race_date")[0])
        bakenhit = process_race(df, df_shutsuba, result_prob)
        race_dict['bakenhit'] = bakenhit
        race_features_df = pl.DataFrame([race_dict])
        if schema is None:
            df_downcast = downcast(race_features_df)
            schema = df_downcast.schema
        else:
            df_downcast = race_features_df.with_columns([pl.col(n).cast(t) for n, t in schema.items()])
        df_downcast.write_ipc(feather_file)

def downcast(df):
    df_downcast = df.to_pandas().fillna(-1, downcast='infer')
    fcols = df_downcast.select_dtypes('float').columns
    icols = df_downcast.select_dtypes('integer').columns
    df_downcast[fcols] = df_downcast[fcols].apply(pd.to_numeric, downcast='float')
    df_downcast[icols] = df_downcast[icols].apply(pd.to_numeric, downcast='integer')
    return pl.from_pandas(df_downcast)

def plot_kde_with_peak(pred_valid_x):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    # グラフのスタイルとサイズ設定
    sns.set_theme(style="white", palette="pastel")
    plt.figure(figsize=(4, 3))

    # カーネル密度推定（KDE）のプロット
    kde = sns.kdeplot(pred_valid_x, bw_adjust=1.2, color='lightblue', label='KDE', alpha=0.7, fill=True)

    # KDEの極大値をプロットして表示
    x, y = kde.get_children()[0].get_paths()[0].vertices.T
    maxid = y.argmax()
    peak_x = x[maxid]

    # 極大値のラインをプロット
    y_position = plt.ylim()[1] * 1.01
    plt.axvline(peak_x, color='blue', linestyle='-', label=f'Mean: {peak_x:.2f}', linewidth=0.8)
    plt.text(peak_x, y_position, f'  {peak_x * 100:.2f}%', horizontalalignment='center', color='blue', fontsize=10)

    # y軸を非表示に設定
    plt.yticks([])
    plt.ylabel('')

    # 横軸をパーセント表示に設定
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1))

    # グラフの枠（スパイン）を消去
    for spine in ['top', 'right', 'left', 'bottom']:
        plt.gca().spines[spine].set_visible(False)

    # レイアウト調整
    plt.tight_layout()

    return plt, peak_x

def image_base64_kde_with_axvline(x):
    import pickle
    from io import BytesIO
    import base64
    import matplotlib
    matplotlib.use('Agg')

    print(f"loading {config.bakenhit_lgb_reg.file}")
    try:
        with open(config.bakenhit_lgb_reg.file, "rb") as f:
            model, pred_valid_x = pickle.load(f)
    except:
        return "data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw=="
    print(f"ploting")
    plt, peak_x = plot_kde_with_peak(pred_valid_x)

    y_position = plt.ylim()[1] * 0.90
    plt.axvline(x, color='crimson', linestyle='--', label=f'Mean: {x:.2f}', linewidth=0.8)
    if x < peak_x:
        plt.text(x, y_position, f'{x * 100:.2f}%', horizontalalignment='right', color='crimson', fontsize=10)
    else:
        plt.text(x, y_position, f'  {x * 100:.2f}%', horizontalalignment='left', color='crimson', fontsize=10)

    # 画像をバッファに保存して返す
    print(f"saving image to buffer")
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # バッファの内容をBase64にエンコード
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return image_base64

if __name__ == "__main__":
    if not Path(config.bakenhit_lgb_reg.file).exists():
        if Path(config.racefeat_file).exists():
            print(f"Already exists {config.racefeat_file}.")
            df_race = pl.read_ipc(config.racefeat_file)
        else:
            save_racefeat()
            
            # 全ての.featherファイルを読み込み、結合
            print("Loading racefeat/*.feather")
            df_race = pl.scan_ipc("racefeat/*.feather").collect()

            print(f"Saving race features to {config.racefeat_file}.")
            df_race.write_ipc(config.racefeat_file)
            print(f"Successfully saved race features to {config.racefeat_file}.")

        # 馬券の的中率を予測する
        import train
        from loguru import logger
        from datetime import datetime

        now = datetime.now().strftime('%Y%m%d_%H%M')
        logger.add(f"temp/bakenhit_{now}.log")

        df_race = df_race.to_pandas()
        bakenhit_config = config.bakenhit_lgb_reg
        noneed = ["race_id", "race_date"] + config.NONEED_COLUMNS
        train_x = df_race.query(bakenhit_config.train).drop(columns=noneed, errors="ignore")
        logger.info("len(train_x): ", len(train_x))
        train_y = train_x.pop(bakenhit_config.target)
        valid_x = df_race.query(bakenhit_config.valid).drop(columns=noneed, errors="ignore")
        valid_y = valid_x.pop(bakenhit_config.target)
        model = train.lightgbm_model(bakenhit_config, train_x, train_y, valid_x, valid_y)
        logger.info(pd.Series(model.feature_importance(importance_type='gain'), index=model.feature_name()).sort_values(ascending=False)[:50])

        import numpy as np
        from sklearn.metrics import mean_squared_error
        pred_valid_x = model.predict(valid_x, num_iteration=model.best_iteration)
        rmse = np.sqrt(mean_squared_error(valid_y, pred_valid_x))
        logger.info(f'Validation RMSE: {rmse:.4f}')

        with open(config.bakenhit_lgb_reg.file, 'wb') as f:
            pickle.dump((model, pred_valid_x), f)
    elif Path(config.bakenhit_lgb_reg.file).exists():
        with open(config.bakenhit_lgb_reg.file, "rb") as f:
            model, pred_valid_x = pickle.load(f)

        import matplotlib
        matplotlib.use('MacOSX')
        plt, peak_x = plot_kde_with_peak(pred_valid_x)
        y_position = plt.ylim()[1] * 0.90
        x = 0.925
        plt.axvline(x, color='crimson', linestyle='--', label=f'Mean: {x:.2f}', linewidth=0.8)
        if x < peak_x:
            plt.text(x, y_position, f'{x * 100:.2f}%', horizontalalignment='right', color='crimson', fontsize=10)
        else:
            plt.text(x, y_position, f'  {x * 100:.2f}%', horizontalalignment='left', color='crimson', fontsize=10)
        plt.show()
