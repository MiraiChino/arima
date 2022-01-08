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

def soup(html):
    html.encoding = html.apparent_encoding
    return BeautifulSoup(html.text, 'html.parser')

@scraping
def scrape_racedates(year, month):
    calendar_url = f"{BASE_URL}/top/calendar.html?year={year}&month={month}"
    print(f"scraping: {calendar_url}")
    calendar_html = requests.get(calendar_url)
    if table := soup(calendar_html).find(class_="Calendar_Table"):
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
    if element := chrome.wait_element("RaceTopRace", driver):
        racelist_text = driver.page_source
        races = BeautifulSoup(racelist_text, 'html.parser').find_all("li", class_="RaceList_DataItem")
        for race in races:
            if a_tag := race.find("a"):
                href = a_tag.attrs['href']
                match = text.raceid.findall(href)
                if len(match) > 0:
                    race_id = match[0]
                    yield race_id

@scraping
def scrape_results(race_id):
    race_url = f"{BASE_URL}/race/result.html?race_id={race_id}&rf=race_list"
    print(f"scraping: {race_url}")
    race_html = requests.get(race_url)
    s = soup(race_html)
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
    s = soup(shutuba_html)
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

def daterange(from_date, to_date):
    from_date = date_type(from_date)
    to_date = date_type(to_date)
    if to_date < from_date:
        return
    for month in range(from_date.month, 12+1):
        yield from_date.year, month
    for year in range(from_date.year+1, to_date.year):
        for month in range(1, 12):
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
