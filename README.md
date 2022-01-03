# arima

## TODO
- [x] レースIDから出走情報をスクレイピングする
- [x] スクレイピングした出走情報の特徴量をつくってモデルに入れられるようにする
- [ ] train
- [ ] predict
- [ ] lambdaで外部からアクセスできるWebページをつくる
- [ ] ランクと回帰の重みづけを考える
- [ ] テストデータでシミュレーションできるようにする
- [ ] レースIDからのオッズ情報をスクレイピングする
- [ ] オッズ情報をと予測結果を対比して見れるようにする
- [ ] 的中率が最大になるように馬券の買い方を検索する
    - [ ] 馬連馬単・3連複3連単が被る場合は期待値のいい方を買うようにして、他の馬券まで伸ばす
- [ ] 騎手、調教師も特徴を計算できるようにする
- [ ] 過去の対戦馬の情報もいれたい

## 過去のレース情報をスクレイピングしてDBに保存
- `python netkeiba.py --from 2008-01 --to 2021-12 --out netkeiba.sqlite`

## 使用データ
- [netkeiba.com](https://race.netkeiba.com/top/?rf=navi)からスクレイピングしたデータ
- 2008年1月1日〜2021年12月12日（13年分、48207レース分）
    - train: 2008年1月1日〜2017年12月31日（9年分、34533レース分）
    - valid: 2018年1月1日〜2020年12月31日（3年分、10362レース分）
    - test: 2021年1月1日〜2021年12月12日（1年分、3312レース分）

## 特徴量一覧
当日17+過去18*7=143

| いつの情報か | 変数             | 中身                                      | Xのパターン      | 
| ------------ | ---------------- | ----------------------------------------- | ---------------- | 
| 当日とれる   | gate             | 枠番                                      |                  | 
|              | horse_no         | 馬番                                      |                  | 
|              | name             | 名前                                      |                  | 
|              | sex              | 性別                                      |                  | 
|              | age              | 年齢                                      |                  | 
|              | penalty          | 斤量                                      |                  | 
|              | jockey           | 騎手                                      |                  | 
|              | barn             | 厩舎                                      |                  | 
|              | fielad           | 芝orダ                                    |                  | 
|              | turn             | 右or左                                    |                  | 
|              | distance         | 距離                                      |                  | 
|              | place_code       | 開催場所                                  |                  | 
|              | weather          | 天気                                      |                  | 
|              | field_condition  | 馬場                                      |                  | 
|              | race_condition   | レース条件                                |                  | 
|              | cos_racedate     | 開催日cos                                 |                  | 
|              | cos_starttime    | 発走時刻cos                               |                  | 
| 過去から計算 | horse_interval_X | 過去X走からの平均レース間日数             | 1,2,3,4,5,10,all | 
|              | horse_place_X    | 過去X走までの当日と同じ開催場所の経験回数 | 1,2,3,4,5,10,all | 
|              | horse_odds_X     | 過去X走までの平均オッズ                   | 1,2,3,4,5,10,all | 
|              | horse_pop_X      | 過去X走までの平均人気                     | 1,2,3,4,5,10,all | 
|              | horse_result_X   | 過去X走までの平均着順                     | 1,2,3,4,5,10,all | 
|              | horse_jockey_X   | 過去X走までの当日と同じ騎手と走った回数   | 1,2,3,4,5,10,all | 
|              | horse_penalty_X  | 過去X走までの平均斤量                     | 1,2,3,4,5,10,all | 
|              | horse_distance_X | 過去X走までの平均距離と当日の距離の差     | 1,2,3,4,5,10,all | 
|              | horse_weather_X  | 過去X走までの当日と同じ天気を走った回数   | 1,2,3,4,5,10,all | 
|              | horse_fc_X       | 過去X走までの当日と同じ馬場を走った回数   | 1,2,3,4,5,10,all | 
|              | horse_time_X     | 過去X走までの平均タイムとの差の平均       | 1,2,3,4,5,10,all | 
|              | horse_margin_X   | 過去X走までの平均着差                     | 1,2,3,4,5,10,all | 
|              | horse_corner3_X  | 過去X走までの3コーナー平均通過順位        | 1,2,3,4,5,10,all | 
|              | horse_corner4_X  | 過去X走までの4コーナー平均通過順位        | 1,2,3,4,5,10,all | 
|              | horse_last3f_X   | 過去X走までの平均last3f                   | 1,2,3,4,5,10,all | 
|              | horse_weight_X   | 過去X走までの平均体重                     | 1,2,3,4,5,10,all | 
|              | horse_wc_X       | 過去X走までの平均体重変化                 | 1,2,3,4,5,10,all | 
|              | horse_prize_X    | 過去X走までの平均獲得賞金                 | 1,2,3,4,5,10,all | 

## 参考
- スクレイピング
    - [netkeibaのWebスクレイピングをPythonで行う【競馬開催日の抽出】 | ジコログ](https://self-development.info/netkeiba%E3%81%AEweb%E3%82%B9%E3%82%AF%E3%83%AC%E3%82%A4%E3%83%94%E3%83%B3%E3%82%B0%E3%82%92python%E3%81%A7%E8%A1%8C%E3%81%86%E3%80%90%E7%AB%B6%E9%A6%AC%E9%96%8B%E5%82%AC%E6%97%A5%E3%81%AE%E6%8A%BD/)
    - [PythonによるnetkeibaのWebスクレイピング【レースIDの抽出】 | ジコログ](https://self-development.info/python%e3%81%ab%e3%82%88%e3%82%8bnetkeiba%e3%81%aeweb%e3%82%b9%e3%82%af%e3%83%ac%e3%82%a4%e3%83%94%e3%83%b3%e3%82%b0%e3%80%90%e3%83%ac%e3%83%bc%e3%82%b9id%e3%81%ae%e6%8a%bd%e5%87%ba%e3%80%91/)
    - [Pythonで競馬サイトWebスクレイピング - Qiita](https://qiita.com/Mokutan/items/89c871eac16b8142b5b2)
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