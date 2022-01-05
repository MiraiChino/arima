import re
import unicodedata

newline = re.compile(r"\n")
blank = re.compile(r" ")
td = re.compile(r"</td>")
tag = re.compile(r"<[^>]*?>")
bracket_l = re.compile(r"\[")
bracket_r = re.compile(r"\]")
racedata11 = re.compile(r"(.+)発走/(.)(\d+)m.*\((.*)\)")
racedata12 = re.compile(r"天候:(.*)/馬場:(.*)")
racedata2 = re.compile(r"日目(</span><span>)+(.+?)</span>.*本賞金:(\d+),(\d+),(\d+),(\d+),(\d+)")
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
def extract_result(text):
    text = remove_trash(text)
    text = td.sub("," ,text)
    text = tag.sub("", text)
    text = bracket_l.sub("", text)
    text = bracket_r.sub("", text)
    result, gate, horse_no, name, sa, penalty, jockey,\
        time, margin, pop, odds, last3f, corner, barn, w, _ = text.split(",")
    try:
        result, gate, horse_no = int(result), int(gate), int(horse_no)
        sex, age = sa[:1], sa[1:]
        age = int(unicodedata.normalize("NFKC", age))
        penalty, pop, odds, last3f = float(penalty), int(pop), float(odds), float(last3f)
        if match := horse_weight.match(w):
            weight, weight_change = match.groups()
            weight, weight_change = int(weight), int(weight_change)
        else:
            weight = w
            weight_change = 0
        return result, gate, horse_no, name, sex, age, penalty, jockey, \
                time, margin, pop, odds, last3f, corner, barn, weight, weight_change
    except:
        return None, None, None, None, None, None, None, None, \
                None, None, None, None, None, None, None, None, None
        
#発走時刻,ダ芝,距離,右左,天候,馬場
def extract_racedata11(racedata1_text):
    text = remove_trash(racedata1_text)
    if match := racedata11.search(text):
        start_time, field, distance, turn = match.groups()
        return start_time, field, int(distance), turn
    else:
        return None, None, None, None

#天候,馬場
def extract_racedata12(racedata1_text):
    text = remove_trash(racedata1_text)
    if match := racedata12.search(text):
        weather, field_condition = match.groups()
        return weather, field_condition
    else:
        return None, None

#レース条件,1位賞金,2位賞金,3位賞金,4位賞金,5位賞金
def extract_racedata2(racedata2_html):
    text = remove_trash(racedata2_html)
    if match := racedata2.search(text):
        _, race_condition, prize1, prize2, prize3, prize4, prize5 = match.groups()
        return race_condition, int(prize1), int(prize2), int(prize3), int(prize4), int(prize5)
    else:
        return None, None, None, None, None, None

#開催年,競馬場コード,開催回数,日数,レース数
def extract_racedata3(raceid_text):
    if match := racedata3.match(raceid_text):
        year, place_code, hold_num, day_num, race_num = match.groups()
        return int(year), int(place_code), int(hold_num), int(day_num), int(race_num)
    else:
        return None, None, None, None, None

def extract_shutuba(text):
    text = remove_trash(text)
    text = td.sub("," ,text)
    text = tag.sub("", text)
    text = bracket_l.sub("", text)
    text = bracket_r.sub("", text)
    gate, horse_no, _, name, sa, penalty, jockey, barn, *_ = text.split(",")
    try:
        if gate and horse_no:
            gate, horse_no = int(gate), int(horse_no)
        else:
            gate, horse_no = None, None
        sex, age =sa[:1], sa[1:]
        age = unicodedata.normalize("NFKC", age)
        return None, gate, horse_no, name, sex, age, penalty, jockey, \
                None, None, None, None, None, None, barn, None, None
    except:
        return None, None, None, None, None, None, None, None, \
                None, None, None, None, None, None, None, None, None