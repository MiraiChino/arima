import pathlib
import pickle

import pandas as pd
import uvicorn
from domonic.html import *
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import HTMLResponse

import css
import netkeiba
import predict


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class PredictAndDumpBaken():
    logs = []

    def __call__(self, race_id):
        self.logs.append(f"-- START")
        self.logs.append(f"race_id: {race_id}")
        baken_pickle = pathlib.Path(f"{race_id}.pickle")
        try:
            if not race_id:
                self.logs.append(f"Invalid race_id")
                self.logs.append(f"-- DONE")
                return
            if baken_pickle.is_file():
                self.logs.append(f"Already {baken_pickle} exists")
                self.logs.append(f"-- DONE")
                return
            self.logs.append(f"scraping shutsuba horse")
            horses = [horse for horse in netkeiba.scrape_shutuba(race_id)]
            df_original = pd.DataFrame(horses, columns=netkeiba.COLUMNS)
            race_info = DotDict(df_original.loc[0, :].to_dict())
            self.logs.append("start to prepare for predicting race results")
            result_prob = predict.result_prob(df_original, task_logs=self.logs)
            names = {no: name for no, name in zip(df_original["horse_no"].to_list(), df_original["name"].to_list())}
            self.logs.append(f"predicting baken probabilities")
            baken = predict.baken_prob(result_prob, names)
            with open(baken_pickle, 'wb') as f:
                pickle.dump((baken, race_info), f)
            self.logs.append(f"{baken_pickle} saved")
        except Exception as e:
            self.logs.append(f"{e}")
        self.logs.append(f"-- DONE")
    
    def get_logs(self):
        return self.logs

app = FastAPI()
task = PredictAndDumpBaken()

@app.get("/predict/{race_id}")
def predict_race(race_id: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(task, race_id)
    return {"message": f"race_id {race_id}: start to predict race result"}

@app.get("/status/{race_id}", response_class=HTMLResponse)
def log(race_id: str):
    return str(html(
        body(
            ul(''.join([f'{li(log)}' for log in task.get_logs()])),
        )
    ))

@app.get("/result/{race_id}", response_class=HTMLResponse)
def race_html(race_id: str, top: int=100, odd_th=2.0):
    baken_pickle = pathlib.Path(f"{race_id}.pickle")
    if baken_pickle.is_file():
        with open(baken_pickle, 'rb') as f:
            baken, r = pickle.load(f)
    else:
        return str(html(body(div(f"Error: Did not find {baken_pickle}. Need to access /predict/{race_id} first."))))
    baken = predict.calc_odds(baken, race_id, top)
    baken = predict.good_baken(baken, odd_th)
    page = str(html(
        head(
            meta(_charset="UTF-8"),
            style(css.jupyter_like_style)
        ),
        body(
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
        ),
        _lang="ja"
    ))
    return page

if __name__ == "__main__":
    uvicorn.run(app, debug=True, host="0.0.0.0", port=8080)
