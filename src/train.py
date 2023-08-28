from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
import numpy as np


from model import AbstractModel
from utils import model_path, info


def train(model: AbstractModel, train_x, train_y, train_params, initializer, group=None, fold_rule=None):
    val_pred = np.zeros(train_x.shape[0])

    if group is not None:
        folds = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=34)
        splits = folds.split(train_x, train_y, group)
    else:
        folds = StratifiedKFold(n_splits=4, shuffle=True, random_state=34)
        splits = folds.split(train_x, train_y)

    if fold_rule is None:
        fold_rule = lambda x: x


    for fold, (train_idx, val_idx) in enumerate(splits):
        model.init_model()
        initializer()

        info("start fold: {}".format(fold))

        x_train, y_train = train_x.iloc[train_idx], train_y.iloc[train_idx]
        x_val, y_val = train_x.iloc[val_idx], train_y.iloc[val_idx]
        pred = model.train_fold(
            fold, x_train, y_train, x_val, y_val, train_params
        )
        if len(pred.shape) == 2:
            val_pred[val_idx] = pred[:, 0]
        else:
            val_pred[val_idx] = pred

    
    
    val_loss = mean_squared_error(train_y, val_pred)
    info("Finish Train. val_loss: {}".format(val_loss))

    model.save_model(model_path(model.name, val_loss, model.wandb_id))

    return val_loss
