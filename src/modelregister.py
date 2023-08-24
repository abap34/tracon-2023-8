from models.catboost import CatBoostRegressionModel
from models.lightgbm import LightGBMRegressionModel


class ModelRegister:
    models = {}

    @classmethod
    def register(cls, name, model):
        print("register model: {}".format(name))
        cls.models[name] = model

    @classmethod
    def get(cls, name):
        if not name in cls.models:
            raise ValueError(
                "model name {} is not defined. Forgot update `MODEL_DICT`? ".format(
                    name
                )
            )
        else:
            return cls.models[name]


ModelRegister.register("lightgbm", LightGBMRegressionModel)
ModelRegister.register("catboost", CatBoostRegressionModel)
