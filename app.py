import pandas as pd
import uvicorn
from domonic.html import *
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

import css
import netkeiba
import predict

app = FastAPI()

class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

@app.get("/race/{race_id}", response_class=HTMLResponse)
async def arima(race_id: str, top: int=30):
    if not race_id:
        return str(html(body(div(f"Invalid race_id: {race_id}"))))
    horses = [horse for horse in netkeiba.scrape_shutuba(race_id)]
    df_original = pd.DataFrame(horses, columns=netkeiba.COLUMNS)
    d = DotDict(df_original.loc[0, :].to_dict())
    result_prob = predict.result_prob(df_original)
    names = {no: name for no, name in zip(df_original["horse_no"].to_list(), df_original["name"].to_list())}
    baken = predict.baken_prob(result_prob, names, race_id, top)

    page = str(html(
        head(
            meta(_charset="UTF-8"),
            style(css.jupyter_like_style)
        ),
        body(
            h3(f"{d.race_num}R {d.race_name}　{d.year}年{d.race_date} {netkeiba.PLACE[d.place_code]}"),
            div(f"{d.start_time}発走 / {d.field}{d.distance}m ({d.turn}) / 天候:{d.weather} / 馬場:{d.field_condition}"),
            div(f"{d.race_condition} / 本賞金:{d.prize1},{d.prize2},{d.prize3},{d.prize4},{d.prize5}万円"),
            hr(),
            h3("単勝"),
            baken["単勝"].df.to_html(),
            h3("馬連"),
            baken["馬連"].df.to_html(),
            h3("馬単"),
            baken["馬単"].df.to_html(),
            h3("三連複"),
            baken["三連複"].df.to_html(),
            h3("三連単"),
            baken["三連単"].df.to_html(),
        ),
        _lang="ja"
    ))
    return page

if __name__ == "__main__":
    uvicorn.run(app, debug=True, host="0.0.0.0", port=8080)
