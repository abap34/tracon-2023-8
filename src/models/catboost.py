import wandb
from catboost import CatBoostRegressor, CatBoostClassifier

from model import AbstractModel 
from utils import save_dict, load_dict


class CBWandbLogger:
    def __init__(self, fold_name, loss_name):
        self.fold_name = fold_name
        self.loss_name = loss_name

    def after_iteration(self, info):
        wandb.log(
            {"train/{}".format(self.fold_name): info.metrics["learn"][self.loss_name][-1]}
        )
        wandb.log(
            {"val/{}".format(self.fold_name): info.metrics["validation"][self.loss_name][-1]}
        )
        return True


class CatBoostRegressionModel(AbstractModel):
    def __init__(self, params):
        name = params.pop("name", "")
        super().__init__(name, params)

    def init_model(self):
        self.model = CatBoostRegressor(**self.params)

    def train_fold(self, fold, train_x, train_y, val_x, val_y, params=None):
        fold_name = "fold_{}".format(fold)
        logger = CBWandbLogger(fold_name, "RMSE")
        if params is None:
            params = {}
        self.model.fit(
            train_x, train_y, eval_set=[(val_x, val_y)], callbacks=[logger], **params
        )
        val_pred = self.model.predict(val_x)
        return val_pred

    def predict(self, test_x):
        return self.model.predict(test_x)

    def save_model(self, path):
        self.model.save_model(path)
        save_dict(path + "_params.yaml", self.model.get_params())

    def load_model(self, path):
        self.model.load_model(path)
        params = load_dict(path + "_params.yaml")
        self.model.set_params(params)


class CatBoostClassificationModel(AbstractModel):
    def __init__(self, params):
        name = params.pop("name", "")
        super().__init__(name, params)

    def init_model(self):
        self.model = CatBoostClassifier(**self.params)

    def train_fold(self, fold, train_x, train_y, val_x, val_y, params=None):
        fold_name = "fold_{}".format(fold)
        logger = CBWandbLogger(fold_name, "MultiClass")
        if params is None:
            params = {}
        self.model.fit(
            train_x, train_y, eval_set=[(val_x, val_y)], callbacks=[logger], **params
        )
        val_pred = self.model.predict(val_x)
        return val_pred

    def predict(self, test_x):
        return self.model.predict(test_x)

    def save_model(self, path):
        self.model.save_model(path)
        save_dict(path + "_params.yaml", self.model.get_params())

    def load_model(self, path):
        self.model.load_model(path)
        params = load_dict(path + "_params.yaml")
        self.model.set_params(params)