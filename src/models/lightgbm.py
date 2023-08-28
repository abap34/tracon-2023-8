from wandb.lightgbm import wandb_callback, log_summary
import lightgbm as lgb
import wandb

from model import AbstractModel
from utils import save_dict, load_dict


class LightGBMRegressionModel(AbstractModel):
    def __init__(self, params):
        name = params.pop("name", "")
        super().__init__(name, params)

    def init_model(self):
        pass

    def train_fold(self, fold, train_x, train_y, val_x, val_y, params=None):
        fold_name = "fold_{}".format(fold)
        if params is None:
            params = {}
        early_stopping_rounds = params.pop("early_stopping_rounds", 100)
        early_stop_callback = lgb.early_stopping(early_stopping_rounds)
        train_data = lgb.Dataset(train_x, train_y)
        val_data = lgb.Dataset(val_x, val_y)
        gbm = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            callbacks=[wandb_callback(), early_stop_callback]
        )
        val_pred = gbm.predict(val_x)
        log_summary(gbm, save_model_checkpoint=True)
        self.model = gbm
        return val_pred

    def predict(self, test_x):
        return self.model.predict(test_x)


    def save_model(self, path):
        self.model.save_model(path)


    def load_model(self, path):
        self.model = lgb.Booster(model_file=path)
        params = load_dict(path + "_params.yaml")
        self.model.set_params(params)


