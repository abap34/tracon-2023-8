from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

from model import AbstractModel
from utils import model_path


def train(model: AbstractModel, train_x, train_y, train_params):
    kf = KFold(n_splits=4, shuffle=True, random_state=34)
    val_pred = np.zeros(train_x.shape[0])
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_x, train_y)):
        model.init_model()
        x_train, y_train = train_x.iloc[train_idx], train_y.iloc[train_idx]
        x_val, y_val = train_x.iloc[val_idx], train_y.iloc[val_idx]
        val_pred[val_idx] = model.train_fold(
            fold, x_train, y_train, x_val, y_val, train_params
        )

    val_loss = mean_squared_error(train_y, val_pred)

    model.save_model(model_path(model.name, val_loss, model.wandb_id))

    return val_loss
