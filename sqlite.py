import sqlite3
import traceback
from contextlib import contextmanager


@contextmanager
def open(dbfile):
    try:
        conn = sqlite3.connect(dbfile)
        yield conn
        conn.commit()
    except:
        print(traceback.format_exc())
        conn.rollback()
    finally:
        conn.close()

CREATE_TABLE_HORSE = """
CREATE TABLE IF NOT EXISTS horse (
	id INTEGER PRIMARY KEY,
    result INTEGER NOT NULL,
    gate INTEGER NOT NULL,
    horse_no INTEGER NOT NULL,
    name TEXT NOT NULL,
    sex TEXT NOT NULL,
    age INTEGER NOT NULL,
    penalty REAL NOT NULL,
    jockey TEXT NOT NULL,
    time TEXT,
    margin TEXT,
    pop INTEGER,
    odds REAL,
    last3f REAL,
    corner TEXT,
    barn TEXT NOT NULL,
    weight INTEGER,
    weight_change INTEGER,
    race_name TEXT NOT NULL,
    start_time TEXT NOT NULL,
    field TEXT NOT NULL,
    distance INTEGER NOT NULL,
    turn TEXT NOT NULL,
    weather TEXT NOT NULL,
    field_condition TEXT NOT NULL,
    race_condition TEXT NOT NULL,
    prize1 INTEGER NOT NULL,
    prize2 INTEGER NOT NULL,
    prize3 INTEGER NOT NULL,
    prize4 INTEGER NOT NULL,
    prize5 INTEGER NOT NULL,
    year INTEGER NOT NULL,
    place_code INTEGER NOT NULL,
    hold_num INTEGER NOT NULL,
    day_num INTEGER NOT NULL,
    race_num INTEGER NOT NULL
);
"""

INSERT_INTO_HORSE = """
INSERT INTO horse (
    result,
    gate,
    horse_no,
    name,
    sex,
    age,
    penalty,
    jockey,
    time,
    margin,
    pop,
    odds,
    last3f,
    corner,
    barn,
    weight,
    weight_change,
    race_name,
    start_time,
    field,
    distance,
    turn,
    weather,
    field_condition,
    race_condition,
    prize1,
    prize2,
    prize3,
    prize4,
    prize5,
    year,
    place_code,
    hold_num,
    day_num,
    race_num
) VALUES (
    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
)
"""