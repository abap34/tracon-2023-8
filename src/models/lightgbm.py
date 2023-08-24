from wandb.lightgbm import wandb_callback, log_summary
import lightgbm as lgb

from model import AbstractModel
from utils import save_dict, load_dict


class LightGBMRegressionModel(AbstractModel):
    def __init__(self, params):
        name = params.pop("name", "")
        super().__init__(name, params)

    def init_model(self):
        self.model = lgb.LGBMRegressor(**self.params)

    def train_fold(self, fold, train_x, train_y, val_x, val_y, params=None):
        fold_name = "fold_{}".format(fold)
        if params is None:
            params = {}
        early_stopping_rounds = params.pop("early_stopping_rounds", 100)
        params["eval_metric"] = params.pop("eval_metric", "rmse")
        params["verbose"] = params.pop("verbose", 100)
        early_stop_callback = lgb.early_stopping(early_stopping_rounds)
        gbm = self.model.fit(
            train_x,
            train_y,
            eval_set=(val_x, val_y),
            callbacks=[wandb_callback(), early_stop_callback],
            **params
        )
        val_pred = self.model.predict(val_x)
        log_summary(gbm, save_model_checkpoint=True)
        return val_pred

    def predict(self, test_x):
        return self.model.predict(test_x)

    def save_model(self, path):
        self.model.save_model(path)
        save_dict(path + "_params.yaml", self.model.get_params())

    def load_model(self, path):
        self.model = lgb.Booster(model_file=path)
        params = load_dict(path + "_params.yaml")
        self.model.set_params(params)


