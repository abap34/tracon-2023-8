import yaml
import random
import string
import pandas as pd
import datetime
import glob
import tqdm
import os


def load_dict(file_path):
    with open(file_path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def save_dict(file_path, d):
    with open(file_path, "w") as f:
        yaml.dump(d, f)


def rand_string(n):
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


def load_feather(colum_name: str):
    return pd.read_feather(f"../data/columns/{colum_name}.feather")


def info(*args):
    print(f"[{datetime.datetime.now()}]", *args)


def save_all_columns(df, feature_type: str, overwrite=False):
    all_save_columns = glob.glob("../data/columns/*.feather")
    assert feature_type in ["train", "test", "target"]
    pbar = tqdm.tqdm(df.columns, desc="save columns")

    for col in pbar:
        pbar.set_postfix_str("save {}".format(col))
        path = feather_path(col, feature_type)
        if feature_type == "train":
            if path in all_save_columns and not overwrite:
                raise ValueError(f"{path} is already exists")
            df[[col]].to_feather(path)
        elif feature_type == "test":
            if path in all_save_columns and not overwrite:
                raise ValueError(f"{path} is already exists")
            df[[col]].to_feather(path)
        elif feature_type == "target":
            if path in all_save_columns and not overwrite:
                raise ValueError(f"{path} is already exists")
            df[[col]].to_feather(path)

def save_columns(column: pd.Series, feature_type: str, col_rename: str, overwrite=False):
    df = pd.DataFrame()
    df[col_rename] = column
    all_save_columns = glob.glob("../data/columns/*.feather")
    assert feature_type in ["train", "test", "target"]
    path = feather_path(col_rename, feature_type)
    if feature_type == "train":
        if path in all_save_columns and not overwrite:
            raise ValueError(f"{path} is already exists")
        df.to_feather(path)
    elif feature_type == "test":
        if path in all_save_columns and not overwrite:
            raise ValueError(f"{path} is already exists")
        df.to_feather(path)
    elif feature_type == "target":
        if path in all_save_columns and not overwrite:
            raise ValueError(f"{path} is already exists")
        df.to_feather(path)





def feather_path(col, feature_type):
    PATH = os.getenv("TRACON_DATA_PATH")
    return PATH + f"/columns/{feature_type}/{col}.feather"


def model_path(name, val_loss, id):
    PATH = os.getenv("TRACON_DATA_PATH")
    return PATH + f"/models/{name}_{val_loss:.4f}_{id}.model"


def submission_path(name):
    PATH = os.getenv("TRACON_DATA_PATH")
    return PATH + f"/submissions/{name}.csv"
