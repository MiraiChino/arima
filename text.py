import re
import unicodedata

from utils import index_exists

newline = re.compile(r"\n")
blank = re.compile(r" ")
li = re.compile(r"</li>")
tr = re.compile(r"</tr>")
td = re.compile(r"</td>")
comma = re.compile(r",")
tag = re.compile(r"<[^>]*?>")
hyphen = re.compile(r"<spanclass=\"Hyphen\"id=\"odds_min_Hyphen\"style=\"\"></span>")
bracket_l = re.compile(r"\[")
bracket_r = re.compile(r"\]")
horse_id = re.compile(r"\/horse\/(\d+)")
jockey_id = re.compile(r"\/jockey\/result\/recent\/(\d+)")
trainer_id = re.compile(r"\/trainer\/result\/recent\/(\d+)")
racedata11 = re.compile(r"(.+)発走/(.)(\d+)m.*\((.*)\)")
racedata12 = re.compile(r"天候:(.*)/馬場:(.*)")
racedata2 = re.compile(r"日目(</span><span>)+(.+?)</span>.*本賞金:(\d+),(\d+),(\d+),(\d+),(\d+)")
racedata3 = re.compile(r"(\d\d\d\d)(\d\d)(\d\d)(\d\d)(\d\d)")
horse_weight = re.compile(r"(\d+)\((.*)\)")
raceid = re.compile(r"\/race\/result.html\?race_id=(.*)&rf=race_list")
racedate = re.compile(r"\/top\/race_list.html\?kaisai_date=(.*)$")
no = re.compile(r"<span>(\d+)</span>")

def remove_trash(text):
    text = str(text)
    text = newline.sub("", text)
    text = blank.sub("", text)
    return text

#着順,枠,馬番,馬名,性,齢,斤量,騎手,タイム,着差,人気,単勝オッズ,後3F,コーナー通過順,厩舎,馬体重,体重増減
def extract_result(text):
    text = remove_trash(text)

    h_id, j_id, t_id = None, None, None
    try:
        if match := horse_id.search(text):
            h_id = match.groups()[0]
        if match := jockey_id.search(text):
            j_id = match.groups()[0]
        if match := trainer_id.search(text):
            t_id = match.groups()[0]
    except:
        print("horse_id, jockey_id, trainer_id = None, None, None")

    text = td.sub("," ,text)
    text = tag.sub("", text)
    text = bracket_l.sub("", text)
    text = bracket_r.sub("", text)
    result, gate, horse_no, name, sa, penalty, jockey,\
        time, margin, pop, odds, last3f, corner, trainer, w, _ = text.split(",")
    try:
        result = int(result)
    except Exception as e:
        print(e)
        result = None

    try:
        gate, horse_no = int(gate), int(horse_no)
        sex, age = sa[:1], sa[1:]
        age = int(unicodedata.normalize("NFKC", age))
    except Exception as e:
        gate, horse_no, sex, age = None, None, None, None

    try:
        penalty, pop, odds, last3f = float(penalty), int(pop), float(odds), float(last3f)
        if match := horse_weight.match(w):
            weight, weight_change = match.groups()
            weight, weight_change = int(weight), int(weight_change)
        else:
            weight = int(w)
            weight_change = 0
    except Exception as e:
        penalty, pop, odds, last3f, weight, weight_change = None, None, None, None, None, None

    return result, gate, horse_no, name, h_id, sex, age, penalty, jockey, j_id, \
        time, margin, pop, odds, last3f, corner, trainer, t_id, weight, weight_change

def extract_tanhuku_no(td_tan, td_huku):
    text = remove_trash(td_tan)
    tanno1, tanno2 = None, None
    if match := no.findall(text):
        if len(match) == 1:
            tanno1 = int(match[0])
        elif len(match) == 2:
            tanno1 = int(match[0])
            tanno2 = int(match[1])

    text = remove_trash(td_huku)
    hukuno1, hukuno2, hukuno3 = None, None, None
    if match := no.findall(text):
        if len(match) == 1:
            hukuno1 = int(match[0])
        elif len(match) == 2:
            hukuno1 = int(match[0])
            hukuno2 = int(match[1])
        elif len(match) == 3:
            hukuno1 = int(match[0])
            hukuno2 = int(match[1])
            hukuno3 = int(match[2])
    return tanno1, tanno2, hukuno1, hukuno2, hukuno3

def extract_payback(Tansho, Fukusho, Umaren, Wide, Umatan, Fuku3, Tan3):
    def get_pays(tr):
        td = tr.find("td", class_="Payout")
        text = td.text[:-1]
        text = comma.sub("", text)
        splitted = text.split("円")
        return splitted
    
    def get_results(tr, max=2):
        td = tr.find("td", class_="Result")
        text = remove_trash(td)
        if match := no.findall(text):
            nums = [int(match[i]) if index_exists(match, i) else None for i in range(0, max)]
        else:
            nums = [None for i in range(1, max+1)]
        return nums

    try:
        tanno = get_results(Tansho, 2) if Tansho else [None, None]
    except Exception as e:
        import traceback
        print(traceback.format_exc())
    
    try:
        hukuno = get_results(Fukusho, 3) if Fukusho else [None, None, None]
    except Exception as e:
        import traceback
        print(traceback.format_exc())

    try:
        pays = get_pays(Tansho) if Tansho else []
        tan1 = int(pays[0]) if pays else None
        tan2 = int(pays[1]) if index_exists(pays, 1) else None
    except Exception as e:
        import traceback
        print(traceback.format_exc())

    try:
        pays = get_pays(Fukusho) if Fukusho else []
        huku1 = int(pays[0]) if pays else None
        huku2 = int(pays[1]) if index_exists(pays, 1) else None
        huku3 = int(pays[2]) if index_exists(pays, 2) else None
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        
    try:
        pays = get_pays(Umaren) if Umaren else []
        ren = int(pays[0]) if pays else None
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        
    try:
        pays = get_pays(Wide) if Wide else []
        wide1 = int(pays[0]) if pays else None
        wide2 = int(pays[1]) if index_exists(pays, 1) else None
        wide3 = int(pays[2]) if index_exists(pays, 2) else None
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        
    try:
        pays = get_pays(Umatan) if Umatan else []
        uma1 = int(pays[0]) if pays else None
        uma2 = int(pays[1]) if index_exists(pays, 1) else None
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        
    try:
        pays = get_pays(Fuku3) if Fuku3 else []
        puku = int(pays[0]) if pays else None
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        
    try:
        pays = get_pays(Tan3) if Tan3 else []
        san1 = int(pays[0]) if pays else None
        san2 = int(pays[1]) if index_exists(pays, 1) else None
    except Exception as e:
        import traceback
        print(traceback.format_exc())
    return tanno, hukuno, [tan1, tan2, huku1, huku2, huku3, ren, wide1, wide2, wide3, uma1, uma2, puku, san1, san2]

#発走時刻,ダ芝,距離,右左
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
    h_id, j_id, t_id = None, None, None
    try:
        if match := horse_id.search(text):
            h_id = match.groups()[0]
        if match := jockey_id.search(text):
            j_id = match.groups()[0]
        if match := trainer_id.search(text):
            t_id = match.groups()[0]
    except:
        print("horse_id, jockey_id, trainer_id = None, None, None")

    text = td.sub("," ,text)
    text = tag.sub("", text)
    text = bracket_l.sub("", text)
    text = bracket_r.sub("", text)
    gate, horse_no, _, name, sa, penalty, jockey, trainer, *_ = text.split(",")

    try:
        if gate and horse_no:
            gate, horse_no = int(gate), int(horse_no)
        else:
            gate, horse_no = None, None
        sex, age =sa[:1], sa[1:]
        age = unicodedata.normalize("NFKC", age)
        return None, gate, horse_no, name, h_id, sex, age, penalty, jockey, j_id, \
                None, None, None, None, None, None, trainer, t_id, None, None
    except Exception as e:
        print(e)
        return None, None, None, None, None, None, None, None, \
                None, None, None, None, None, None, None, None, None

def extract_odds(text):
    text = remove_trash(text)
    text = hyphen.sub("-", text)
    text = tr.sub("@", text)
    text = td.sub(",", text)
    text = li.sub("$" ,text)
    text = tag.sub("", text)
    text = bracket_l.sub("", text)
    text = bracket_r.sub("", text)
    results = text.split("@")[1:-1]
    results = [row.split(",") for row in results]
    return results

def split_rentan(text):
    return tuple(int(n) for n in text[:-1].split("$$"))
