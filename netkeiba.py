import re
import time
import traceback
from functools import wraps

import requests
from bs4 import BeautifulSoup

import chrome
import text

BASE_URL = "https://race.netkeiba.com"

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

def race_url(race_id):
    return f"{BASE_URL}/race/result.html?race_id={race_id}&rf=race_list"

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
def scrape_horses(race_id):
    race_url = f"{BASE_URL}/race/result.html?race_id={race_id}&rf=race_list"
    print(f"scraping: {race_url}")
    race_html = requests.get(race_url)
    s = soup(race_html)
    racename = text.remove_trash(s.find("div", class_="RaceName").text)
    racedata1 = text.extract_racedata1(s.find("div", class_="RaceData01").text)
    racedata2 = text.extract_racedata2(s.find("div", class_="RaceData02").text)
    racedata3 = text.extract_racedata3(race_id)
    for horse_html in s.find_all("tr",class_="HorseList"):
        horse = text.extract_horse(horse_html)
        yield *horse, racename, *racedata1, *racedata2, *racedata3
# 長さ:35
# (16, 4, 8, 'ロイヤルパープル', '牡', 3, 56.0, 'マーフ', '1:16.0', '3/4', 2, 3.1, 39.9, '13-13', '美浦加藤征', 516, 2,
# '3歳未勝利',
# '10:55', 'ダ', 1200, '右', '晴', '良',
# 'サラ系３歳未勝利', 510, 200, 130, 77, 51,
# 2020, 6, 1, 1, 3)

if __name__ == "__main__":
    import sqlite
    with sqlite.open("netkeiba.sqlite") as conn:
        conn.execute(sqlite.CREATE_TABLE_HORSE)
        db = conn.cursor()
        with chrome.driver() as driver:
            for year in range(2000, 2021+1):
                for month in range(1, 12+1):
                    for race_date in scrape_racedates(year, month):
                        for race_id in scrape_raceids(driver, race_date):
                            for horse in scrape_horses(race_id):
                                db.execute(sqlite.INSERT_INTO_HORSE, horse)
                        db.commit()
