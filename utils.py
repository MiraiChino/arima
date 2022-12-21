import datetime

import numpy as np


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    reduction = 100 * (start_mem - end_mem) / start_mem
    if verbose: print(f'{df.shape} memory usage decreased {start_mem:5.2f}Mb to {end_mem:5.2f}Mb ({reduction:.1f}% reduction)')
    return df

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
