# arima

## TODO
- [ ] テストデータでシミュレーションできるようにする
- [x] 馬券の買い方を表示する
- [x] 騎手、調教師も特徴を計算できるようにする
- [x] 過去の対戦馬の情報もいれたい→queryにすべての対戦馬の過去情報も入ってる
- [x] レーティング
- [x] ワイド
- [x] 複勝
- [x] 時系列データのクロスバリデーション
- [ ] race_date-生年月日
- [ ] 親の戦績（リークしないように注意）
    - [ ] 何歳（race_date-生年月日）のときにどのくらい戦績だったとか
- [x] コーナー→過去脚質（7項目の和を100%としたときの割合）と今距離、今馬場、競馬場、枠
- [x] スタッキングでrank, regを特徴にして再度rank
- [x] catboost
- [x] prizeでターゲットエンコーディング
- [x] dbを軽くする

## How to use
- 過去のレース情報をスクレイピングしてDBに保存
    - `python netkeiba.py`
- 特徴量などを前処理してDBに保存
    - `python feature.py`
- ランキングと回帰のモデルをtrainする
    - `python train.py`
- レースを予測する
    - `python predict.py --raceid 202206010111`
- レースを予測する（Webアプリ経由）
    - Webアプリ立ち上げ
        - `docker compose up`
    - ブラウザからアクセス
        - 馬券予測
            - http://localhost:8080/predict/202205021211
        - 馬券オッズ取得、結果ページ作成
            - http://localhost:8080/create/202205021211
        - 結果ページ
            - http://localhost:8080/result/202205021211

## 使用データ
- [netkeiba.com](https://race.netkeiba.com/top/?rf=navi)からスクレイピングしたデータ
- 2008年1月1日〜2024年12月31日（17年分）
