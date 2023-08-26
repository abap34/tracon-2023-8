import wandb

from utils import rand_string


class AbstractModel:
    def __init__(self, name, params):
        self.name = name
        self.params = params
        self.wandb_id = rand_string(8)

    def init_model(self):
        raise NotImplementedError

    def train_fold(self, fold, train_x, train_y, val_x, val_y, **kwargs):
        raise NotImplementedError

    def predict(self, test_x):
        raise NotImplementedError

    def save_model(self, path):
        raise NotImplementedError

    def load_model(self, *args, **kwargs):
        raise NotImplementedError
