import sys
sys.path.append('../../src')

import pandas as pd

from run import run
from utils import save_all_columns


train = pd.read_csv('../../data/train.csv')
test = pd.read_csv('../../data/test.csv')


save_all_columns(train.drop("score", axis=1), feature_type="train")
save_all_columns(test, feature_type="test")
save_all_columns(train[["score"]], feature_type="target")


run(
    in_columns=["anime_id", "user"],
    target_column="score",
    run_config={
        "name": "test",
        "model": "catboost",
        "submit": True,
    },
    model_params={
        "name": "test",
        "iterations": 100,
        "learning_rate": 0.1,
        "depth": 3,
        "cat_features": ["anime_id", "user"],
    },
    train_params={
        "early_stopping_rounds": 10,   
        "verbose": False,
    }
)