import os
import pandas as pd
import numpy as np
import pickle
import json
import datetime
import torch as th
import os
import shutil


class MyEncoder(json.JSONEncoder):
    """
    Taken from:
    https://stackoverflow.com/questions/27050108/convert-numpy-type-to-python
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime.datetime):
            return obj.__str__()
        else:
            return super(MyEncoder, self).default(obj)


def make_dir(directory, clean=False):
    if not os.path.exists(directory):
        os.makedirs(directory)
    elif clean:
        shutil.rmtree(directory)
        os.makedirs(directory)


def load(filename):
    _, ext = os.path.splitext(filename)
    if ext == '.json':
        return load_json(filename)
    elif ext in '.gzip':
        return load_df(filename)
    elif ext == '.pkl':
        return load_obj(filename)
    elif ext == '.pt':
        return load_th(filename)
    elif ext == '.csv':
        return load_df(filename)
    else:
        raise ValueError(f'Unknown filetype {filename}')


def store_json(obj, path, name):
    make_dir(path)
    filename = os.path.join(path, name + '.json')
    with open(filename, 'w') as f:
        json.dump(obj, f, cls=MyEncoder)


def store_obj(obj, path, name):
    make_dir(path)
    filename = os.path.join(path, name + '.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


# TODO: introduce fixed fileextensions for pandas + X
def store_df(obj, path, name, ftype='parquet'):
    make_dir(path)
    if ftype == 'parquet':
        filename = os.path.join(path, name + '.parquet.gzip')
        obj.to_parquet(filename, compression='gzip')
    elif ftype == 'json':
        filename = os.path.join(path, name + '.json')
        obj.to_json(filename, orient='records')
    elif ftype == 'csv':
        filename = os.path.join(path, name + '.csv')
        obj.to_csv(filename)
    else:
        raise ValueError(f'Unknown filetype {ftype} to store with pandas.')


def store_df_csv(obj, path, name):
    store_df(obj, path, name, ftype='csv')


def load_json(filename):
    with open(filename) as f:
        return json.load(f)


def load_df(filename):
    parquet_ext = 'parquet.gzip'
    json_ext = 'json'
    csv_ext = 'csv'
    if filename[-len(parquet_ext):] == parquet_ext:
        with open(filename, 'rb') as f:
            return pd.read_parquet(f)
    elif filename[-len(json_ext):] == json_ext:
        with open(filename, 'rb') as f:
            return pd.read_json(f)
    elif filename[-len(csv_ext):] == csv_ext:
        with open(filename, 'rb') as f:
            return pd.read_csv(f)
    else:
        raise Exception("Unkown filetype.")


def load_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def store_th(objs, path, name):
    make_dir(path)
    filename = os.path.join(path, name + '.pt')
    th.save(
        objs,
        filename
    )


def load_th(filename):
    return th.load(filename)
