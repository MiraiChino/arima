# arima

## TODO
- [ ] テストデータでシミュレーションできるようにする
- [x] 馬券の買い方を表示する
- [x] 騎手、調教師も特徴を計算できるようにする
- [ ] 過去の対戦馬の情報もいれたい
- [x] ワイド
- [x] 複勝
- [x] 時系列データのクロスバリデーション
- [ ] 今-生年月日
- [ ] 親の戦績（リークしないように注意）
    - [ ] 何歳（race_date-生年月日）のときにどのくらい戦績だったとか
- [ ] コーナー→過去脚質（7項目の和を100%としたときの割合）と今距離、今馬場、競馬場、枠
- [x] prizeでターゲットエンコーディング

## How to use
- 過去のレース情報をスクレイピングしてDBに保存
    - `python netkeiba.py`
- 特徴量などを前処理してDBに保存
    - `python feature_extractor.py`
- ランキングと回帰のモデルをtrainする
    - `python train.py`
- レースを予測する
    - `python predict.py --raceid 202206010111`
- レースを予測する（Webアプリ経由）
    - Webアプリ立ち上げ
        - `docker compose up`
    - ブラウザからアクセス
        - 予測タスク開始
            - http://localhost:8080/predict/202205021211
        - 予測タスクの状態表示
            - http://localhost:8080/status/202205021211
        - 予測の結果表示
            - http://localhost:8080/result/202205021211

## 使用データ
- [netkeiba.com](https://race.netkeiba.com/top/?rf=navi)からスクレイピングしたデータ
- 2008年1月1日〜2022年5月8日（13年分、49527レース分）

## 特徴量一覧
当日17+過去92*5=477
| どんな情報か             | 変数              | 中身                                                          | Xのパターン  | 
| ------------------------ | ----------------- | ------------------------------------------------------------- | ------------ | 
| 当日とれる(17)           | gate              | 枠番                                                          |              | 
|                          | horse_no          | 馬番                                                          |              | 
|                          | name              | 名前                                                          |              | 
|                          | sex               | 性別                                                          |              | 
|                          | age               | 年齢                                                          |              | 
|                          | penalty           | 斤量                                                          |              | 
|                          | jockey            | 騎手                                                          |              | 
|                          | barn              | 厩舎                                                          |              | 
|                          | fielad            | 芝orダ                                                        |              | 
|                          | corner            | 右or左                                                        |              | 
|                          | distance          | 距離                                                          |              | 
|                          | place_code        | 開催場所                                                      |              | 
|                          | weather           | 天気                                                          |              | 
|                          | field_condition   | 馬場                                                          |              | 
|                          | race_condition    | レース条件                                                    |              | 
|                          | cos_racedate      | 開催日cos                                                     |              | 
|                          | cos_starttime     | 発走時刻cos                                                   |              | 
| 同馬の過去から計算(43)   | horse_interval_X  | 過去X走からの平均レース間日数                                 | 1,2,3,10,all | 
|                          | horse_odds_X      | 過去X走までの平均オッズ                                       | 1,2,3,10,all | 
|                          | horse_pop_X       | 過去X走までの平均人気                                         | 1,2,3,10,all | 
|                          | horse_result_X    | 過去X走までの平均着順                                         | 1,2,3,10,all | 
|                          | horse_penalty_X   | 過去X走までの平均斤量                                         | 1,2,3,10,all | 
|                          | horse_weather_X   | 過去X走までの当日と同じ天気を走った回数                       | 1,2,3,10,all | 
|                          | horse_time_X      | 過去X走までの平均タイムとの差の平均                           | 1,2,3,10,all | 
|                          | horse_margin_X    | 過去X走までの平均着差                                         | 1,2,3,10,all | 
|                          | horse_corner3_X   | 過去X走までの3コーナー平均通過順位                            | 1,2,3,10,all | 
|                          | horse_corner4_X   | 過去X走までの4コーナー平均通過順位                            | 1,2,3,10,all | 
|                          | horse_last3f_X    | 過去X走までの平均last3f                                       | 1,2,3,10,all | 
|                          | horse_weight_X    | 過去X走までの平均体重                                         | 1,2,3,10,all | 
|                          | horse_wc_X        | 過去X走までの平均体重変化                                     | 1,2,3,10,all | 
|                          | horse_prize_X     | 過去X走までの平均獲得賞金                                     | 1,2,3,10,all | 
|                          | horse_Yprize_X    | 過去X走までの同じ条件Yのレースの平均獲得賞金                  | 1,2,3,10,all | 
|                          | Yの17パターン     | 場所　距離　芝/ダ　馬場　右/左　騎手                          |              | 
|                          |                   | 芝/ダ∧右/左　芝/ダ∧距離　芝/ダ∧馬場                        |              | 
|                          |                   | 場所∧距離　場所∧芝/ダ　場所∧芝/ダ∧右/左                   |              | 
|                          |                   | 場所∧芝/ダ∧距離　場所∧芝/ダ∧馬場　距離∧芝/ダ∧馬場       |              | 
|                          |                   | 場所∧芝/ダ∧距離∧馬場　場所∧芝/ダ∧距離∧右/左             |              | 
|                          | horse_Ydrize_X    | 過去X走までの同じ条件Yのレースの平均(1-距離差)/距離*賞金      | 1,2,3,10,all | 
|                          | Yの12パターン     | なし　場所　芝/ダ　馬場　右/左　騎手                          |              | 
|                          |                   | 芝/ダ∧右/左　芝/ダ∧馬場　場所∧芝/ダ                        |              | 
|                          |                   | 場所∧芝/ダ∧右/左　場所∧芝/ダ∧馬場                         |              | 
|                          |                   | 場所∧芝/ダ∧馬場∧右/左                                      |              | 
| 同騎手の過去から計算(27) | jockey_interval_X | 過去X走からの平均レース間日数                                 | 1,2,3,10,all | 
|                          | jockey_odds_X     | 過去X走までの平均オッズ                                       | 1,2,3,10,all | 
|                          | jockey_pop_X      | 過去X走までの平均人気                                         | 1,2,3,10,all | 
|                          | jockey_result_X   | 過去X走までの平均着順                                         | 1,2,3,10,all | 
|                          | jockey_weather_X  | 過去X走までの当日と同じ天気を走った回数                       | 1,2,3,10,all | 
|                          | jockey_time_X     | 過去X走までの平均タイムとの差の平均                           | 1,2,3,10,all | 
|                          | jockey_margin_X   | 過去X走までの平均着差                                         | 1,2,3,10,all | 
|                          | jockey_corner3_X  | 過去X走までの3コーナー平均通過順位                            | 1,2,3,10,all | 
|                          | jockey_corner4_X  | 過去X走までの4コーナー平均通過順位                            | 1,2,3,10,all | 
|                          | jockey_last3f_X   | 過去X走までの平均last3f                                       | 1,2,3,10,all | 
|                          | jockey_prize_X    | 過去X走までの平均獲得賞金                                     | 1,2,3,10,all | 
|                          | jockey_Yprize_X   | 過去X走までの同じ条件Yのレースの平均獲得賞金                  | 1,2,3,10,all | 
|                          | Yの16パターン     | 場所　距離　芝/ダ　馬場　右/左                                |              | 
|                          |                   | 芝/ダ∧右/左　芝/ダ∧距離　芝/ダ∧馬場                        |              | 
|                          |                   | 場所∧距離　場所∧芝/ダ　場所∧芝/ダ∧右/左                   |              | 
|                          |                   | 場所∧芝/ダ∧距離　場所∧芝/ダ∧馬場　距離∧芝/ダ∧馬場       |              | 
|                          |                   | 場所∧芝/ダ∧距離∧馬場　場所∧芝/ダ∧距離∧右/左             |              | 
| 同厩舎の過去から計算(22) | barn_interval_X   | 過去X走からの平均レース間日数                                 | 1,2,3,10,all | 
|                          | barn_odds_X       | 過去X走までの平均オッズ                                       | 1,2,3,10,all | 
|                          | barn_pop_X        | 過去X走までの平均人気                                         | 1,2,3,10,all | 
|                          | barn_result_X     | 過去X走までの平均着順                                         | 1,2,3,10,all | 
|                          | barn_time_X       | 過去X走までの平均タイムとの差の平均                           | 1,2,3,10,all | 
|                          | barn_margin_X     | 過去X走までの平均着差                                         | 1,2,3,10,all | 
|                          | barn_corner3_X    | 過去X走までの3コーナー平均通過順位                            | 1,2,3,10,all | 
|                          | barn_corner4_X    | 過去X走までの4コーナー平均通過順位                            | 1,2,3,10,all | 
|                          | barn_last3f_X     | 過去X走までの平均last3f                                       | 1,2,3,10,all | 
|                          | barn_prize_X      | 過去X走までの平均獲得賞金                                     | 1,2,3,10,all | 
|                          | barn_Yprize_X     | 過去X走までの同じ条件Yのレースの平均獲得賞金                  | 1,2,3,10,all | 
|                          | Yの12パターン     | 場所　距離　芝/ダ　馬場　芝/ダ∧距離　芝/ダ∧馬場             |              | 
|                          |                   | 場所∧距離　場所∧芝/ダ　場所∧芝/ダ∧距離                    |              | 
|                          |                   | 場所∧芝/ダ∧馬場　距離∧芝/ダ∧馬場　場所∧芝/ダ∧距離∧馬場 |              | 

## 参考
- スクレイピング
    - [netkeibaのWebスクレイピングをPythonで行う【競馬開催日の抽出】 | ジコログ](https://self-development.info/netkeiba%E3%81%AEweb%E3%82%B9%E3%82%AF%E3%83%AC%E3%82%A4%E3%83%94%E3%83%B3%E3%82%B0%E3%82%92python%E3%81%A7%E8%A1%8C%E3%81%86%E3%80%90%E7%AB%B6%E9%A6%AC%E9%96%8B%E5%82%AC%E6%97%A5%E3%81%AE%E6%8A%BD/)
    - [PythonによるnetkeibaのWebスクレイピング【レースIDの抽出】 | ジコログ](https://self-development.info/python%e3%81%ab%e3%82%88%e3%82%8bnetkeiba%e3%81%aeweb%e3%82%b9%e3%82%af%e3%83%ac%e3%82%a4%e3%83%94%e3%83%b3%e3%82%b0%e3%80%90%e3%83%ac%e3%83%bc%e3%82%b9id%e3%81%ae%e6%8a%bd%e5%87%ba%e3%80%91/)
    - [Pythonで競馬サイトWebスクレイピング - Qiita](https://qiita.com/Mokutan/items/89c871eac16b8142b5b2)
    - [umihico/docker-selenium-lambda: Minimum demo of headless chrome and selenium on container image on AWS Lambda](https://github.com/umihico/docker-selenium-lambda)
- ランク学習
    - [LightGBM でかんたん Learning to Rank - 霧でも食ってろ](https://knuu.github.io/ltr_by_lightgbm.html)
    - [機械学習で競馬必勝本に勝てるのか？ 〜Pythonで実装するランク学習〜 - エニグモ開発者ブログ](https://tech.enigmo.co.jp/entry/2020/12/09/100000)
    - [ランク学習でバーチャルスクリーニングする - tonetsの日記](https://tonets.hatenablog.com/entry/2019/12/23/135131)
    - [[競馬予想AI] ランク学習で着順予想するとなかなか強力だったお話｜とりまる｜note](https://note.com/dataij/n/n5a6d121b13ab?magazine_key=mfc655f2636e0)
- LightGBM
    - [機械学習で競馬の回収率100%超えを達成した話 - Qiita](https://qiita.com/Mshimia/items/6c54d82b3792925b8199)
- 特徴量
    - [LightGBMのCategorical Featureによって精度が向上するか？ - Qiita](https://qiita.com/sinchir0/items/b038757e578b790ec96a)
    - [特徴量からの周期性の抽出方法 - Qiita](https://qiita.com/squash/items/299f73a21bc46766c60f)
    - [Python: Target Encoding のやり方について - CUBE SUGAR CONTAINER](https://blog.amedama.jp/entry/target-mean-encoding-types)
    - [Kaggle TalkingData Fraud Detection コンペの解法まとめ(基本編) | Ad-Tech Lab Blog](https://blog.recruit.co.jp/rco/kaggle_talkingdata_basic/)
- その他
    - [たった数行でpandasを高速化する2つのライブラリ(pandarallel/swifter) - フリーランチ食べたい](https://blog.ikedaosushi.com/entry/2020/07/26/173109)