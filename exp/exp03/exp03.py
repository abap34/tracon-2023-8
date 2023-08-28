import sys
sys.path.append('../../src')

import pandas as pd

from run import run
from utils import save_all_columns


train = pd.read_csv('../../data/train.csv')
test = pd.read_csv('../../data/test.csv')



run(
    in_columns=["anime_id", "user"],
    target_column="score",
    test_columns=["test_seen_anime_id", "test_seen_user"],
    test_id="test_seen_ID",
    run_config={
        "name": "reg-seen",
        "model":"catboost",
        "submit": False,
        "task": "reg",
        "submit_path": "seen.csv",
    },
    model_params={
        "name": "seen-model-reg-v1",
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




run(
    in_columns=["anime_id", "user"],
    target_column="score",
    test_columns=["test_unseen_anime_id", "test_unseen_user"],
    test_id="test_unseen_ID",
    run_config={
        "name": "reg-unseen",
        "model":"catboost",
        "submit": False,
        "task": "reg",
        "submit_path": "unseen.csv",
    },
    model_params={
        "name": "unseen-model-reg-v1",
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

