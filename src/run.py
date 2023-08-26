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
    test_columns: list[str],
    test_id: str,
    run_config: dict,
    model_params: dict,
    train_params: dict,
):
    assert len(in_columns) == len(test_columns)
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
    for col in tqdm(test_columns, desc="load test columns"):
        df = pd.read_feather(feather_path(col, "test"))
        test_df[col] = df[col].values

    target_df = pd.read_feather(feather_path(target_column, "target"))

    model = ModelRegister.get(run_config["model"])(model_params)

    info("start training")
    
    if "groupkfold" in run_config:
        group_id = run_config["groupkfold"]
        group = pd.read_feather(feather_path(group_id, "train"))
    else:
        group = None

    val_loss = train(model, train_df, target_df, train_params, group=group)

    wandb.alert(
        title="Finish Training: {}".format(run_config["name"]),
        text="val_loss: {}".format(val_loss),
    )

    feature_name_map = {}
    for i in range(len(in_columns)):
        feature_name_map[test_columns[i]] = in_columns[i]
    test_df = test_df.rename(columns=feature_name_map)

    pred = model.predict(test_df)
    pred_df = pd.DataFrame()
    pred_df["ID"] = pd.read_feather(feather_path(test_id, "test"))[test_id]
    pred_df["score"] = pred
    if "submit_path" in run_config:
        submit_file_path = run_config["submit_path"]
    else:
        submit_file_path = submission_path(run_config["name"])

    pred_df.to_csv(submit_file_path, index=False)

    if run_config["submit"]:
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
