import re
import unicodedata

newline = re.compile(r"\n")
blank = re.compile(r" ")
td = re.compile(r"</td>")
tag = re.compile(r"<[^>]*?>")
bracket_l = re.compile(r"\[")
bracket_r = re.compile(r"\]")
racedata1 = re.compile(r"(.+)発走/(.)(\d+)m.*\((.)\)/天候:(.)/馬場:(.)")
racedata2 = re.compile(r"日目(.+)\xa0\xa0\xa0\xa0\xa0.*本賞金:(\d+),(\d+),(\d+),(\d+),(\d+)")
racedata3 = re.compile(r"(\d\d\d\d)(\d\d)(\d\d)(\d\d)(\d\d)")
horse_weight = re.compile(r"(\d+)\((.*)\)")
raceid = re.compile(r"\/race\/result.html\?race_id=(.*)&rf=race_list")
racedate = re.compile(r"\/top\/race_list.html\?kaisai_date=(.*)$")

def remove_trash(text):
    text = str(text)
    text = newline.sub("", text)
    text = blank.sub("", text)
    return text

#着順,枠,馬番,馬名,性,齢,斤量,騎手,タイム,着差,人気,単勝オッズ,後3F,コーナー通過順,厩舎,馬体重,体重増減
def extract_horse(text):
    text = remove_trash(text)
    text = td.sub("," ,text)
    text = tag.sub("", text)
    text = bracket_l.sub("", text)
    text = bracket_r.sub("", text)
    result, gate, horse_no, name, sa, penalty, jockey,\
        time, margin, pop, odds, last3f, corner, barn, w, _ = text.split(",")
    weight, weight_change = horse_weight.match(w).groups()
    sex, age = sa[:1], sa[1:]
    age = unicodedata.normalize("NFKC", age)
    return int(result), int(gate), int(horse_no), name, sex, int(age), float(penalty), jockey, \
            time, margin, int(pop), float(odds), float(last3f), corner, barn, int(weight), int(weight_change)

#発走時刻,ダ芝,距離,右左,天候,馬場
def extract_racedata1(racedata1_text):
    text = remove_trash(racedata1_text)
    start_time, field, distance, turn, weather, field_condition = racedata1.search(text).groups()
    return start_time, field, int(distance), turn, weather, field_condition

#レース条件,1位賞金,2位賞金,3位賞金,4位賞金,5位賞金
def extract_racedata2(racedata2_text):
    text = remove_trash(racedata2_text)
    race_condition, prize1, prize2, prize3, prize4, prize5 = racedata2.search(text).groups()
    return race_condition, int(prize1), int(prize2), int(prize3), int(prize4), int(prize5)

#開催年,競馬場コード,開催回数,日数,レース数
def extract_racedata3(raceid_text):
    year, place_code, hold_num, day_num, race_num = racedata3.match(raceid_text).groups()
    return int(year), int(place_code), int(hold_num), int(day_num), int(race_num)
