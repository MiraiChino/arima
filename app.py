import pandas as pd
from domonic.html import *

import netkeiba
import predict


def handler(event, context):
    race_id = str(event.pathParameters.race_id)
    horses = [horse for horse in netkeiba.scrape_shutuba(race_id)]
    df_original = pd.DataFrame(horses, columns=netkeiba.COLUMNS)
    result_prob = predict.result_prob(df_original)
    baken = predict.baken_prob(result_prob, race_id)

    page = str(html(
        head(
            meta(_charset="UTF-8")
        ),
        body(
            h1(f'Hello {race_id}'),
            baken["単勝"].df.to_html(index=False)
        ),
        _lang="ja"
    ))
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "text/html"},
        "body": page
    }
