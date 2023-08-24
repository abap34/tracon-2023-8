import sys
sys.path.append('../../src')

import pandas as pd

from run import run
from utils import save_all_columns


train = pd.read_csv('../../data/train.csv')
test = pd.read_csv('../../data/test.csv')

# 0, 11を削除
train = train[train["score"] != 0]
train = train[train["score"] != 11].reset_index()

# scoreをrename
train = train.rename(columns={"score": "score_removed_0_11"})

save_all_columns(train.drop("score_removed_0_11", axis=1), feature_type="train")
save_all_columns(test, feature_type="test")
save_all_columns(train[["score_removed_0_11"]], feature_type="target")


run(
    in_columns=["anime_id", "user"],
    target_column="score_removed_0_11",
    run_config={
        "name": "class",
        "model": "catboost_class",
        "submit": True,
        "task": "class",
    },
    model_params={
        "name": "test",
        "iterations": 100,
        "learning_rate": 0.1,
        "depth": 5,
        "cat_features": ["anime_id", "user"],
    },
    train_params={
        "early_stopping_rounds": 10,   
        "verbose": False,
    }
)