import re
import time
import traceback
import argparse
from functools import wraps
from pathlib import Path

import pandas as pd
import polars as pl
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from selenium.webdriver.common.by import By

import chrome
import config
import text
import utils

BASE_URL = "https://race.netkeiba.com"
HORSE_COLUMNS = (
    "result", "gate", "horse_no", "name", "horse_id", "sex", "age", "penalty", "jockey", "jockey_id", "time", "margin",
    "pop", "odds", "last3f", "corner", "trainer", "trainer_id", "weight", "weight_change", "race_id"
)
RACE_PRE_COLUMNS = (
    "race_id", "race_name", "start_time", "field", "distance", "turn",
    "weather", "field_condition", "race_condition", "prize1", "prize2",
    "prize3", "prize4", "prize5", "year", "place_code", "hold_num", "day_num",
    "race_num", "race_date",
)
RACE_PAY_COLUMNS = (
    "tanno1", "tanno2", "hukuno1", "hukuno2", "hukuno3", "tan1", "tan2", "huku1", "huku2", "huku3", "wide1", "wide2", "wide3",
    "ren", "uma1", "uma2", "puku", "san1", "san2"
)
RACE_AFTER_COLUMNS = RACE_PRE_COLUMNS + RACE_PAY_COLUMNS

PLACE = {
    1: "札幌",
    2: "函館",
    3: "福島",
    4: "新潟",
    5: "東京",
    6: "中山",
    7: "中京",
    8: "京都",
    9: "阪神",
    10: "小倉",
}
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100',
}

netkeiba_date = re.compile(r"netkeiba(\d+)-(\d+).*")

def scraping(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        time.sleep(1)
        try:
            return func(*args, **kwargs)
        except:
            print(traceback.format_exc())
    return wrapper

def soup_html(html):
    html.encoding = html.apparent_encoding
    return soup(html.text)

def soup(html_text):
    return BeautifulSoup(html_text, 'html.parser')

@scraping
def scrape_racedates(year, month):
    calendar_url = f"{BASE_URL}/top/calendar.html?year={year}&month={month}"
    print(f"scraping: {calendar_url}")
    calendar_html = requests.get(calendar_url, headers=headers)
    if table := soup_html(calendar_html).find(class_="Calendar_Table"):
        elems = table.find_all("td", class_="RaceCellBox")
        for elem in elems:
            if a_tag := elem.find("a"):
                href = a_tag.attrs['href']
                match = text.racedate.findall(href)
                if len(match) > 0:
                    race_date = match[0]
                    yield race_date
        
@scraping
def scrape_raceids(driver, race_date):
    racelist_url = f"{BASE_URL}/top/race_list.html?kaisai_date={race_date}"
    print(f"scraping: {racelist_url}")
    driver.get(racelist_url)
    if driver.wait_all_elements():
        for race in driver.find_elements(By.CSS_SELECTOR, "li.RaceList_DataItem"):
            href = race.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
            match = text.raceid.findall(href)
            if len(match) > 0:
                race_id = match[0]
                yield race_id

@scraping
def scrape_results(race_id):
    race_url = f"{BASE_URL}/race/result.html?race_id={race_id}&rf=race_list"
    print(f"scraping: {race_url}")
    race_html = requests.get(race_url, headers=headers)
    s = soup_html(race_html)
    racename_div = s.find("div", class_="RaceName")
    racename_h1 = s.find("h1", class_="RaceName")
    racename = text.remove_trash(racename_div.text) if racename_div else text.remove_trash(racename_h1.text) if racename_h1 else None
    racedata11 = text.extract_racedata11(s.find("div", class_="RaceData01").text)
    racedata12 = text.extract_racedata12(s.find("div", class_="RaceData01").text)
    racedata2 = text.extract_racedata2(s.find("div", class_="RaceData02"))
    racedata3 = text.extract_racedata3(race_id)
    racedate = s.find("dd", class_="Active").text
    div_payback = s.find("div", class_="FullWrap")
    tr_paybacks = {c: div_payback.find("tr", class_=c) for c in ("Tansho", "Fukusho", "Umaren", "Wide", "Umatan", "Fuku3", "Tan3")}
    tanno, hukuno, payback = text.extract_payback(**tr_paybacks)
    race_data = [race_id, racename, *racedata11, *racedata12, *racedata2, *racedata3, racedate, *tanno, *hukuno, *payback]
    horses = []
    for horse_html in s.find_all("tr",class_="HorseList"):
        result = [*text.extract_result(horse_html), race_id]
        if result.count(None) != len(result):
            horses.append(result)
    return race_data, horses 

@scraping
def scrape_shutuba(race_id):
    shutuba_url = f"{BASE_URL}/race/shutuba.html?race_id={race_id}"
    print(f"scraping: {shutuba_url}")
    shutuba_html = requests.get(shutuba_url, headers=headers)
    s = soup_html(shutuba_html)
    racename_div = s.find("div", class_="RaceName")
    racename_h1 = s.find("h1", class_="RaceName")
    racename = text.remove_trash(racename_div.text) if racename_div else text.remove_trash(racename_h1.text) if racename_h1 else None
    racedata11 = text.extract_racedata11(s.find("div", class_="RaceData01").text)
    racedata12 = text.extract_racedata12(s.find("div", class_="RaceData01").text)
    racedata2 = text.extract_racedata2(s.find("div", class_="RaceData02"))
    racedata3 = text.extract_racedata3(race_id)
    racedate = s.find("dd", class_="Active").text
    race_data = [race_id, racename, *racedata11, *racedata12, *racedata2, *racedata3, racedate]
    horses = []
    for horse_html in s.find_all("tr",class_="HorseList"):
        shutuba_horse = [*text.extract_shutuba(horse_html), race_id]
        if shutuba_horse.count(None) != len(shutuba_horse):
            horses.append(shutuba_horse)
    return race_data, horses 

@scraping
def scrape_odds(driver, odds_url, convert_func=None):
    @chrome.retry(10, verb=True)
    def scrape_oddstable():
        tables = soup(driver.page_source).find_all("table", class_="RaceOdds_HorseList_Table")
        odds_list = text.extract_odds(tables[0])
        if convert_func:
            result = convert_func(odds_list)
        else:
            result = odds_list
        return result
    print(f"scraping: {odds_url}")
    driver.get(odds_url)
    if driver.wait_all_elements():
        odds_list = scrape_oddstable()
        return odds_list

def scrape_tanhuku(driver, race_id):
    tanhuku_url = f"{BASE_URL}/odds/index.html?race_id={race_id}"
    to_float = lambda huku_str: round(sum(float(h) for h in huku_str.split('-'))/2, 1)
    def convert(odds_list):
        result = []
        for odds in odds_list:
            try:
                pop, waku, no, _, name, tan, huku, _ = odds
                result.append((int(no), float(tan), to_float(huku)))
            except Exception as e:
                print(f"{e}: {odds}")
        return result
    tanhuku_odds = scrape_odds(driver, tanhuku_url, convert)
    tanshou_odds = {no: tan for no, tan, huku in tanhuku_odds}
    hukushou_odds = {no: huku for no, tan, huku in tanhuku_odds}
    return tanshou_odds, hukushou_odds

@scraping
def scrape_12odds(driver, url, odds_convert=None):
    print(f"scraping: {url}")
    driver.get(url)
    umaren_odds = {}
    if driver.wait_all_elements():
        no12_odds = []
        for table in driver.find_elements(By.CSS_SELECTOR, "table.Odds_Table"):
            col_label = table.find_element(By.CSS_SELECTOR, "tr.col_label")
            no2 = int(text.remove_trash(col_label.text))
            odds_list = text.extract_odds(table.get_attribute("innerHTML"))
            converted_odds = []
            for odds in odds_list:
                try:
                    no3, odds, _ = odds
                    if odds_convert:
                        converted_odds.append((no2, int(no3), float(odds_convert(odds))))
                    else:
                        converted_odds.append((no2, int(no3), float(odds)))
                except Exception as e:
                    print(f"{e}: ")
            no12_odds += converted_odds
        for no1, no2, odds in no12_odds:
            umaren_odds[(no1, no2)] = odds
    return umaren_odds

def scrape_umatan(driver, race_id):
    umatan_url = f"{BASE_URL}/odds/index.html?type=b6&race_id={race_id}&housiki=c0"
    return scrape_12odds(driver, umatan_url)

def scrape_umaren(driver, race_id):
    umaren_url = f"{BASE_URL}/odds/index.html?type=b4&race_id={race_id}&housiki=c0"
    return scrape_12odds(driver, umaren_url)

def scrape_wide(driver, race_id):
    wide_url = f"{BASE_URL}/odds/index.html?type=b5&race_id={race_id}&housiki=c0"
    odds_convert = lambda odds: round(sum(float(o) for o in odds.split('-'))/2, 1)
    return scrape_12odds(driver, wide_url, odds_convert)

@scraping
def scrape_ninki(driver, ninki_url, convert_func=None):
    @chrome.retry(10, verb=True)
    def scrape_oddstable():
        time.sleep(1)
        table = driver.find_element(By.CSS_SELECTOR, "table.RaceOdds_HorseList_Table")
        odds_list = text.extract_odds(table.get_attribute("innerHTML"))
        if convert_func:
            result = convert_func(odds_list)
        else:
            result = odds_list
        return result
    print(f"scraping: {ninki_url}")
    driver.get(ninki_url)
    if driver.wait_all_elements():
        for value in driver.select_options("ninki_select"):
            print(f"select ninki{int(value)+1}~{int(value)+100}")
            odds_list = scrape_oddstable()
            yield odds_list

def scrape_sanrentan_generator(driver, race_id):
    sanrentan_url = f"{BASE_URL}/odds/index.html?type=b8&race_id={race_id}&&housiki=c99"
    def convert(odds_list):
        result = {}
        for odds in odds_list[1:]:
            try:
                pop, _, sanren, odds, *_ = odds
                result[text.split_rentan(sanren)] = float(odds)
            except Exception as e:
                print(f"{e}: {odds}")
        return result
    return scrape_ninki(driver, sanrentan_url, convert)

def scrape_sanrenpuku_generator(driver, race_id):
    sanrenpuku_url = f"{BASE_URL}/odds/index.html?type=b7&race_id={race_id}&housiki=c99"
    def convert(odds_list):
        result = {}
        for odds in odds_list[1:]:
            try:
                pop, _, sanren, odds, *_ = odds
                result[text.split_rentan(sanren)] = float(odds)
            except Exception as e:
                print(f"{e}: {odds}")
        return result
    return scrape_ninki(driver, sanrenpuku_url, convert)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Netkeiba Scraper")
    parser.add_argument("--dry-run", action="store_true", help="Run the script without making any file changes")
    args = parser.parse_args()

    last_updated = sorted(Path("netkeiba").iterdir(), key=lambda p: p.stat().st_mtime)[-1]
    if ".DS_Store" in last_updated.stem:
        last_updated = sorted(Path("netkeiba").iterdir(), key=lambda p: p.stat().st_mtime)[-2]
    if match := netkeiba_date.match(last_updated.stem):
        year, month = match.groups()
        if not args.dry_run:
            Path(f"netkeiba/netkeiba{year}-{month}.races.feather").unlink(missing_ok=True)
            print(f"remove netkeiba/netkeiba{year}-{month}.races.feather")
            Path(f"netkeiba/netkeiba{year}-{month}.horses.feather").unlink(missing_ok=True)
            print(f"remove netkeiba/netkeiba{year}-{month}.horses.feather")
        else:
            print(f"dry-run: would remove netkeiba/netkeiba{year}-{month}.races.feather")
            print(f"dry-run: would remove netkeiba/netkeiba{year}-{month}.horses.feather")

    with chrome.driver() as driver:
        for year, month in utils.daterange(config.from_date, config.to_date):
            print(f"-- {year}-{month}")
            race_file = f"netkeiba/netkeiba{year}-{month}.races.feather"
            horse_file = f"netkeiba/netkeiba{year}-{month}.horses.feather"
            if Path(race_file).is_file() and Path(horse_file).is_file():
                print(f"already exists {race_file} and {horse_file}")
                continue

            races, horses = [], []
            try:
                for race_date in scrape_racedates(year, month):
                    for race_id in scrape_raceids(driver, race_date):
                        race_data, horses_data = scrape_results(race_id)
                        races.append(race_data)
                        horses += horses_data
            except Exception as e:
                print(f"Error(race_id={race_id}): {e}")

            try:
                race_df = pd.DataFrame(races, columns=RACE_AFTER_COLUMNS)
                race_df = pl.from_pandas(race_df)
                if not args.dry_run:
                    race_df.write_ipc(race_file)
                    print(f"saved: {year}-{month} races -> {race_file}")
                else:
                    print(f"dry-run: would save {year}-{month} races -> {race_file}")

                horse_df = pd.DataFrame(horses, columns=HORSE_COLUMNS)
                horse_df = pl.from_pandas(horse_df)
                if not args.dry_run:
                    horse_df.write_ipc(horse_file)
                    print(f"saved: {year}-{month} horses -> {horse_file}")
                else:
                    print(f"dry-run: would save {year}-{month} horses -> {horse_file}")

            except Exception as e:
                print(e)
                import pdb; pdb.set_trace()

    race_chunks, horse_chunks = [], []
    for year, month in tqdm(list(utils.daterange(config.from_date, config.to_date))):
        race = (
            pl.read_ipc(f"netkeiba/netkeiba{year}-{month}.races.feather")
            .lazy()
            .with_columns([
                pl.col(c).cast(pl.Float64) for c in RACE_PAY_COLUMNS
            ])
        )
        horse = (
            pl.read_ipc(f"netkeiba/netkeiba{year}-{month}.horses.feather")
            .lazy()
        )
        race_chunks.append(race)
        horse_chunks.append(horse)
    ldf_races = pl.concat(race_chunks)
    ldf_horses = pl.concat(horse_chunks)
    df = ldf_horses.join(ldf_races, on='race_id', how='left').collect()
    if not args.dry_run:
        df.write_ipc(config.netkeiba_file)
    else:
        print(f"dry-run: would save final dataframe to {config.netkeiba_file}")
