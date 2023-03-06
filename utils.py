import datetime


def date_type(date_str):
    return datetime.datetime.strptime(date_str, "%Y-%m")

def daterange(from_date, to_date):
    from_date = date_type(from_date)
    to_date = date_type(to_date)
    if to_date < from_date:
        return
    if from_date.year == to_date.year:
        for month in range(from_date.month, to_date.month+1):
            yield to_date.year, month
    else:
        for month in range(from_date.month, 12+1):
            yield from_date.year, month
        for year in range(from_date.year+1, to_date.year):
            for month in range(1, 12+1):
                yield year, month
        for month in range(1, to_date.month+1):
            yield to_date.year, month

def index_exists(ls, i):
    if i < len(ls):
        return True
    else:
        return False

def flatten(x):
    return [z for y in x for z in (flatten(y) if hasattr(y, '__iter__') and not isinstance(y, str) else (y,))]
