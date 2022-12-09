import pathlib
import pickle
from typing import List

import pandas as pd
from domonic.html import *

import css
import netkeiba
import predict


DONE = "-- DONE"

class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class Task():
    id = ''
    logs = []
    race_id = 'no_id'

    def get_id(self):
        return self.id

    def get_logs(self):
        return self.logs

    def init(self):
        self.logs = []
        self.race_id = "no_id"

    def is_doing(self):
        return self.logs and self.logs[-1] != DONE
    
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
        self.logs.append(f"-- START {id}")
        baken_pickle = pathlib.Path(f"{race_id}.predict")
        try:
            if not race_id:
                self.logs.append(f"Invalid race_id.")
                self.logs.append(DONE)
                return
            if baken_pickle.is_file():
                self.logs.append(f"Already {baken_pickle} exists.")
                self.logs.append(DONE)
                return
            self.logs.append(f"scraping: https://race.netkeiba.com/race/shutuba.html?race_id={race_id}")
            horses = [horse for horse in netkeiba.scrape_shutuba(race_id)]
            df_original = pd.DataFrame(horses, columns=netkeiba.COLUMNS)
            race_info = DotDict(df_original.loc[0, :].to_dict())
            result_prob = predict.result_prob(df_original, task_logs=self.logs)
            names = {no: name for no, name in zip(df_original["horse_no"].to_list(), df_original["name"].to_list())}
            self.logs.append(f"predicting: 単勝 複勝 ワイド 馬連 馬単 三連複 三連単")
            baken = predict.baken_prob(result_prob, names)
            with open(baken_pickle, 'wb') as f:
                pickle.dump((baken, race_info), f)
            self.logs.append(f"{baken_pickle} saved")
            self.logs.append(f"<a href='{next_url}'>Go /create/{race_id}</a>")
            self.logs.append(DONE)
        except Exception as e:
            self.logs.append(f"{e}")

class CreateBakenHTML(Task):

    def __call__(self, id, race_id, top, odd_th, next_url):
        self.id = id
        self.race_id = race_id
        self.logs.append(f"-- START {id}")
        baken_html = pathlib.Path(f"{race_id}.html")
        try:
            with open(f"{race_id}.predict", 'rb') as f:
                baken, r = pickle.load(f)
        except Exception as e:
            self.logs.append(f"{e}")

        try:
            self.logs.append(f"loaded {race_id}.predict")
            baken = predict.calc_odds(baken, race_id, top, self.logs)
            baken = predict.good_baken(baken, odd_th)
        except Exception as e:
            self.logs.append(f"{e}")
            self.logs.append(f"Not found odds data")
            pagebody = body(
                h3(f"{r.race_num}R {r.race_name}　{r.year}年{r.race_date} {netkeiba.PLACE[r.place_code]}"),
                div(f"{r.start_time}発走 / {r.field}{r.distance}m ({r.turn}) / 天候:{r.weather} / 馬場:{r.field_condition}"),
                div(f"{r.race_condition} / 本賞金:{r.prize1},{r.prize2},{r.prize3},{r.prize4},{r.prize5}万円"),
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
                baken["単勝"].df2.to_html(index=False),
                h3("複勝", _class="clear"),
                baken["複勝"].df.to_html(classes='left'),
                baken["複勝"].df2.to_html(index=False),
                h3("ワイド", _class="clear"),
                baken["ワイド"].df.to_html(classes='left'),
                baken["ワイド"].df2.to_html(index=False),
                h3("馬連", _class="clear"),
                baken["馬連"].df.to_html(classes='left'),
                baken["馬連"].df2.to_html(index=False),
                h3("馬単", _class="clear"),
                baken["馬単"].df.to_html(classes='left'),
                baken["馬単"].df2.to_html(index=False),
                h3("三連複", _class="clear"),
                baken["三連複"].df.to_html(classes='left'),
                baken["三連複"].df2.to_html(index=False),
                h3("三連単", _class="clear"),
                baken["三連単"].df.to_html(classes='left'),
                baken["三連単"].df2.to_html(index=False),
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
            self.logs.append(DONE)
        except Exception as e:
            self.logs.append(f"{e}")
