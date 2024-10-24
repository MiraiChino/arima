import datetime
import pathlib
import pickle
import traceback
from typing import List

import polars as pl
from domonic.html import *

import css
import netkeiba
import predict

DONE = "-- DONE"
NO_ID = 'no_id'

class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class Logs(list):
    def append(self, val):
        now = datetime.datetime.now().strftime('%H:%M:%S')
        super(Logs, self).append(f"{now}  {val}")

class Task():
    id = NO_ID
    logs = Logs()
    race_id = NO_ID
    doing = False

    def get_id(self):
        return self.id

    def get_logs(self):
        return self.logs

    def init(self):
        self.logs = Logs()
        self.race_id = NO_ID
        self.doing = False

    def done(self):
        self.doing = False
        self.logs.append(DONE)

    def is_doing(self):
        return self.doing
    
    def get_raceid(self):
        return self.race_id

def get_doing_task(tasks: List[Task]):
    for task in tasks:
        if task.is_doing():
            return task

class PredictBaken(Task):

    def __call__(self, id, race_id, next_url):
        self.id = id
        self.race_id = race_id
        self.doing = True
        self.logs.append(f"-- START {id}")
        baken_pickle = pathlib.Path(f"{race_id}.predict")
        try:
            if not race_id:
                self.logs.append(f"Invalid race_id.")
                self.done()
                return
            if baken_pickle.is_file():
                self.logs.append(f"Already {baken_pickle} exists.")
                self.done()
                return
            self.logs.append(f"scraping: https://race.netkeiba.com/race/shutuba.html?race_id={race_id}")
            race_data, horses = netkeiba.scrape_shutuba(race_id)
            df_horses = pl.DataFrame(horses, schema=netkeiba.HORSE_COLUMNS, orient="row")
            df_races = pl.DataFrame([race_data], schema=netkeiba.RACE_PRE_COLUMNS, orient="row")
            df = df_horses.join(df_races, on='race_id', how='left')
            race_info = DotDict(df.row(0, named=True))
            if df['horse_no'].null_count() == len(df['horse_no']):
                self.logs.append(f"assign temporary horse number because failed to scrape it")
                temporary_horse_no = pl.Series([i for i in range(1, 1+len(df['horse_no']))])
                df.replace('horse_no', temporary_horse_no)
            self.logs.append(f"scraped horse data")
            for no, name in zip(df["horse_no"].to_list(), df["name"].to_list()):
                self.logs.append(f"{no}: {name}")
            predict.load_models_and_configs(task_logs=self.logs)
            df_feat = predict.search_df_feat(df, task_logs=self.logs)
            result_prob = predict.result_prob(df_feat, task_logs=self.logs)
            names = {no: name for no, name in zip(df["horse_no"].to_list(), df["name"].to_list())}
            for no, name, p in zip(df["horse_no"].to_list(), df["name"].to_list(), result_prob.values()):
                self.logs.append(f"{p*100:.2f}% {no}: {name}")
            baken = predict.baken_prob(result_prob, names)
            self.logs.append(f"Predict bakenhit")
            bakenhit_prob = predict.result_bakenhit(df_feat, task_logs=self.logs)

            with open(baken_pickle, 'wb') as f:
                pickle.dump((baken, bakenhit_prob, race_info), f)
            self.logs.append(f"{baken_pickle} saved")
            self.logs.append(f"<a href='{next_url}'>Go /create/{race_id}</a>")
            self.done()
        except Exception as e:
            self.logs.append(f"{traceback.format_exc()}")

class CreateBakenHTML(Task):

    def __call__(self, id, race_id, top, odd_th, next_url):
        self.id = id
        self.race_id = race_id
        self.doing = True
        self.logs.append(f"-- START {id}")
        baken_html = pathlib.Path(f"{race_id}.html")
        try:
            with open(f"{race_id}.predict", 'rb') as f:
                baken, bakenhit_prob, r = pickle.load(f)
            self.logs.append(f"loaded {race_id}.predict")
        except Exception as e:
            self.logs.append(f"{traceback.format_exc()}")

        try:
            baken = predict.calc_odds(baken, race_id, top, self.logs)
            baken = predict.pretty_baken(baken, top)
            baken = predict.good_baken(baken, odd_th)
            self.logs.append(f"start creating {baken_html}")
        except Exception as e:
            self.logs.append(f"{traceback.format_exc()}")
            with open(f"{race_id}.predict", 'rb') as f:
                baken, r = pickle.load(f)
            baken = predict.pretty_prob(baken, top)
            pagebody = body(
                h3(f"{r.race_num}R {r.race_name}　{r.year}年{r.race_date} {netkeiba.PLACE[r.place_code]}"),
                div(f"{r.start_time}発走 / {r.field}{r.distance}m ({r.turn}) / 天候:{r.weather} / 馬場:{r.field_condition}"),
                div(f"{r.race_condition} / 本賞金:{r.prize1},{r.prize2},{r.prize3},{r.prize4},{r.prize5}万円"),
                hr(),
                div(f"馬券の的中率: {bakenhit_prob:.2%} "),
                div(f"※予測トップ3の各馬券（単勝1枚,複勝3枚,ワイド3枚,馬連1枚,馬単1枚,3連複1枚,3連単1枚）計11枚を買ったときに当たる馬券枚数の期待値"),
                hr(),
                h3("単勝", _class="clear"),
                baken["単勝"].df.to_html(classes='left'),
                h3("複勝", _class="clear"),
                baken["複勝"].df.to_html(classes='left'),
                h3("ワイド", _class="clear"),
                baken["ワイド"].df.to_html(classes='left'),
                h3("馬連", _class="clear"),
                baken["馬連"].df.to_html(classes='left'),
                h3("馬単", _class="clear"),
                baken["馬単"].df.to_html(classes='left'),
                h3("三連複", _class="clear"),
                baken["三連複"].df.to_html(classes='left'),
                h3("三連単", _class="clear"),
                baken["三連単"].df.to_html(classes='left'),
            )
        else:
            pagebody = body(
                h3(f"{r.race_num}R {r.race_name}　{r.year}年{r.race_date} {netkeiba.PLACE[r.place_code]}"),
                div(f"{r.start_time}発走 / {r.field}{r.distance}m ({r.turn}) / 天候:{r.weather} / 馬場:{r.field_condition}"),
                div(f"{r.race_condition} / 本賞金:{r.prize1},{r.prize2},{r.prize3},{r.prize4},{r.prize5}万円"),
                hr(),
                h3("単勝", _class="clear"),
                baken["単勝"].df.to_html(classes='left'),
                baken["単勝"].df_return.to_html(index=False),
                h3("複勝", _class="clear"),
                baken["複勝"].df.to_html(classes='left'),
                baken["複勝"].df_return.to_html(index=False),
                h3("ワイド", _class="clear"),
                baken["ワイド"].df.to_html(classes='left'),
                baken["ワイド"].df_return.to_html(index=False),
                h3("馬連", _class="clear"),
                baken["馬連"].df.to_html(classes='left'),
                baken["馬連"].df_return.to_html(index=False),
                h3("馬単", _class="clear"),
                baken["馬単"].df.to_html(classes='left'),
                baken["馬単"].df_return.to_html(index=False),
                h3("三連複", _class="clear"),
                baken["三連複"].df.to_html(classes='left'),
                baken["三連複"].df_return.to_html(index=False),
                h3("三連単", _class="clear"),
                baken["三連単"].df.to_html(classes='left'),
                baken["三連単"].df_return.to_html(index=False),
            )

        try:
            page = str(html(
                head(
                    meta(_charset="UTF-8"),
                    style(css.jupyter_like_style),
                ),
                pagebody,
                _lang="ja"
            ))
            with open(baken_html, 'w') as f:
                f.write(page)
            self.logs.append(f"{baken_html} saved")
            self.logs.append(f"<a href='{next_url}'>Go /result/{race_id}</a>")
            self.done()
        except Exception as e:
            self.logs.append(f"{traceback.format_exc()}")
