import json
import yaml
import sys
import shutil
import os


# basics


def save_json(obj, filename):
    with open(filename, 'w') as f:
        json.dump(obj, f)


def load_json(filename):
    with open(filename) as f:
        data = json.load(f)
    return data


def save_yaml(obj, filename):
    with open(filename, 'w') as f:
        yaml.dump(obj, f)


def load_yaml(filename):
    with open(filename) as f:
        data = yaml.safe_load(f)
    return data


def get_subfolders(d):
    return [o for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]


def get_subfiles(d):
    return [o for o in os.listdir(d)]


def make_dir(directory, clean=False):
    if not os.path.exists(directory):
        os.makedirs(directory)
    elif clean:
        shutil.rmtree(directory)
        os.makedirs(directory)
