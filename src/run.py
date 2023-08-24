import wandb
from tqdm import tqdm
import subprocess
import pandas as pd

from train import train
from modelregister import ModelRegister
from utils import feather_path, submission_path, info


def run(
    in_columns: list[str],
    target_column: str,
    run_config: dict,
    model_params: dict,
    train_params: dict,
):
    params = {
        "in_columns": in_columns,
        "target_column": target_column,
        "model_params": model_params,
        "train_params": train_params,
    }

    wandb.init(
        project="tracon-2023-8",
        name=run_config["name"],
        config=params,
    )

    # load feather for all colum
    train_df = pd.DataFrame()
    for col in tqdm(in_columns, desc="load train columns"):
        df = pd.read_feather(feather_path(col, "train"))
        train_df[col] = df[col].values

    test_df = pd.DataFrame()
    for col in tqdm(in_columns, desc="load test columns"):
        df = pd.read_feather(feather_path(col, "test"))
        test_df[col] = df[col].values

    target_df = pd.read_feather(feather_path(target_column, "target"))

    model = ModelRegister.get(run_config["model"])(model_params)

    info("start training")
    
    val_loss = train(model, train_df, target_df, train_params)

    wandb.alert(
        title="Finish Training: {}".format(run_config["name"]),
        text="val_loss: {}".format(val_loss),
    )

    if run_config["submit"]:
        pred = model.predict(test_df)
        pred_df = pd.DataFrame()
        pred_df["ID"] = pd.read_feather(feather_path("ID", "test"))["ID"].values
        pred_df["score"] = pred
        submit_file_path = submission_path(run_config["name"])
        pred_df.to_csv(submit_file_path, index=False)
        subprocess.run(
            [
                "kaggle",
                "competitions",
                "submit",
                "-c",
                "trap-competetion-2023-summer",
                "-f",
                submit_file_path,
                "-m",
                run_config["name"],
            ]
        )
        info("submit success")
