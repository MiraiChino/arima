import pandas as pd
from domonic.html import *

import css
import netkeiba
import predict


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def handler(event, context):
    if event and 'pathParameters' in event and 'race_id' in event["pathParameters"]:
        race_id = str(event["pathParameters"]["race_id"])
    else:
        return {
            "statusCode": 200,
            "body": "No race_id"
        }
    if event and 'queryStringParameters' in event and 'top' in event["queryStringParameters"]:
        top = event["queryStringParameters"]["top"]
    else:
        top = 30

    horses = [horse for horse in netkeiba.scrape_shutuba(race_id)]
    df_original = pd.DataFrame(horses, columns=netkeiba.COLUMNS)
    d = DotDict(df_original.loc[0, :].to_dict())
    result_prob = predict.result_prob(df_original)
    baken = predict.baken_prob(result_prob, df_original["name"].to_list(), race_id, top)

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
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "text/html"},
        "body": page
    }

if __name__ == "__main__":
    handler(
        event={
            "pathParameters": {
                "race_id": "202106050811"
            },
            "queryStringParameters": {
                "top": 30
            }
        },
        context=None
    )
