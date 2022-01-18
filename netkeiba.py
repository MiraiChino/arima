import datetime
import sqlite3
import time
import traceback
from functools import wraps

import pandas as pd
import requests
from bs4 import BeautifulSoup

import chrome
import config
import text

BASE_URL = "https://race.netkeiba.com"
COLUMNS = (
    "result", "gate", "horse_no", "name", "sex", "age", "penalty", "jockey",
    "time", "margin", "pop", "odds", "last3f", "corner", "barn", "weight",
    "weight_change", "race_name", "start_time", "field", "distance", "turn",
    "weather", "field_condition", "race_condition", "prize1", "prize2",
    "prize3", "prize4", "prize5", "year", "place_code", "hold_num", "day_num",
    "race_num", "race_date"
)
# 長さ:36
# (16, 4, 8, 'ロイヤルパープル', '牡', 3, 56.0, 'マーフ',
# '1:16.0', '3/4', 2, 3.1, 39.9, '13-13', '美浦加藤征', 516,
# 2, '3歳未勝利', '10:55', 'ダ', 1200, '右',
# '晴', '良', 'サラ系３歳未勝利', 510, 200,
# 130, 77, 51, 2020, 6, 1, 1,
# 3, '1月19日(日)')

def date_type(date_str):
    return datetime.datetime.strptime(date_str, "%Y-%m")

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
    calendar_html = requests.get(calendar_url)
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
        for race in driver.find_elements_by_css_selector("li.RaceList_DataItem"):
            href = race.find_element_by_css_selector("a").get_attribute("href")
            match = text.raceid.findall(href)
            if len(match) > 0:
                race_id = match[0]
                yield race_id

@scraping
def scrape_results(race_id):
    race_url = f"{BASE_URL}/race/result.html?race_id={race_id}&rf=race_list"
    print(f"scraping: {race_url}")
    race_html = requests.get(race_url)
    s = soup_html(race_html)
    racename = text.remove_trash(s.find("div", class_="RaceName").text)
    racedata11 = text.extract_racedata11(s.find("div", class_="RaceData01").text)
    racedata12 = text.extract_racedata12(s.find("div", class_="RaceData01").text)
    racedata2 = text.extract_racedata2(s.find("div", class_="RaceData02"))
    racedata3 = text.extract_racedata3(race_id)
    racedate = s.find("dd", class_="Active").text
    for horse_html in s.find_all("tr",class_="HorseList"):
        result = text.extract_result(horse_html)
        if result.count(None) != len(result):
            yield *result, racename, *racedata11, *racedata12, *racedata2, *racedata3, racedate

@scraping
def scrape_shutuba(race_id):
    shutuba_url = f"{BASE_URL}/race/shutuba.html?race_id={race_id}"
    print(f"scraping: {shutuba_url}")
    shutuba_html = requests.get(shutuba_url)
    s = soup_html(shutuba_html)
    racename = text.remove_trash(s.find("div", class_="RaceName").text)
    racedata11 = text.extract_racedata11(s.find("div", class_="RaceData01").text)
    racedata12 = text.extract_racedata12(s.find("div", class_="RaceData01").text)
    racedata2 = text.extract_racedata2(s.find("div", class_="RaceData02"))
    racedata3 = text.extract_racedata3(race_id)
    racedate = s.find("dd", class_="Active").text
    for horse_html in s.find_all("tr",class_="HorseList"):
        shutuba_horse = text.extract_shutuba(horse_html)
        if shutuba_horse.count(None) != len(shutuba_horse):
            yield *shutuba_horse, racename, *racedata11, *racedata12, *racedata2, *racedata3, racedate

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

def scrape_tanshou(driver, race_id):
    tanshou_url = f"{BASE_URL}/odds/index.html?race_id={race_id}"
    convert = lambda odds_list: {int(no): float(odds) for pop, waku, no, _, name, odds, huku, _ in odds_list}
    tanshou_odds = scrape_odds(driver, tanshou_url, convert)
    return tanshou_odds

@scraping
def scrape_12odds(driver, url):
    print(f"scraping: {url}")
    driver.get(url)
    umaren_odds = {}
    if driver.wait_all_elements():
        no12_odds = []
        for table in driver.find_elements_by_css_selector("table.Odds_Table"):
            col_label = table.find_element_by_css_selector("tr.col_label")
            no2 = int(text.remove_trash(col_label.text))
            odds_list = text.extract_odds(table.get_attribute("innerHTML"))
            no12_odds += [(no2, int(no3), float(odds)) for no3, odds, _ in odds_list]
        for no1, no2, odds in no12_odds:
            umaren_odds[(no1, no2)] = odds
    return umaren_odds

def scrape_umatan(driver, race_id):
    umatan_url = f"{BASE_URL}/odds/index.html?type=b6&race_id={race_id}&housiki=c0"
    return scrape_12odds(driver, umatan_url)

def scrape_umaren(driver, race_id):
    umaren_url = f"{BASE_URL}/odds/index.html?type=b4&race_id={race_id}&housiki=c0"
    return scrape_12odds(driver, umaren_url)

@scraping
def scrape_ninki(driver, ninki_url, convert_func=None):
    @chrome.retry(10, verb=True)
    def scrape_oddstable():
        time.sleep(1)
        table = driver.find_element_by_css_selector("table.RaceOdds_HorseList_Table")
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
    convert = lambda odds_list: {text.split_rentan(sanren): float(odds) for pop, _, sanren, odds, *_ in odds_list[1:]}
    return scrape_ninki(driver, sanrentan_url, convert)

def scrape_sanrenpuku_generator(driver, race_id):
    sanrenpuku_url = f"{BASE_URL}/odds/index.html?type=b7&race_id={race_id}&housiki=c99"
    convert = lambda odds_list: {text.split_rentan(sanren): float(odds) for pop, _, sanren, odds, *_ in odds_list[1:]}
    return scrape_ninki(driver, sanrenpuku_url, convert)

def daterange(from_date, to_date):
    from_date = date_type(from_date)
    to_date = date_type(to_date)
    if to_date < from_date:
        return
    for month in range(from_date.month, 12+1):
        yield from_date.year, month
    for year in range(from_date.year+1, to_date.year):
        for month in range(1, 12+1):
            yield year, month
    if from_date.year < to_date.year:
        for month in range(1, to_date.month+1):
            yield to_date.year, month

if __name__ == "__main__":
    with chrome.driver() as driver:
        for year, month in daterange(config.from_date, config.to_date):
            horses = []
            for race_date in scrape_racedates(year, month):
                for race_id in scrape_raceids(driver, race_date):
                    for horse in scrape_results(race_id):
                        horses.append(horse)
            with sqlite3.connect(config.netkeiba_db) as conn:
                df = pd.DataFrame(horses, columns=COLUMNS)
                df.to_sql('horse', con=conn, if_exists='append', index=False)
            print(f"database: inserted race data in {year}-{month}")
