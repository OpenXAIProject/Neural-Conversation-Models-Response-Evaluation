from configs import get_config
from utils import get_loader
import os
import pickle
import solvers


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    config = get_config(mode='train')
    val_config = get_config(mode='valid')
    print(config)

    train_data_loader = get_loader(utters=load_pickle(config.utter_path), batch_size=config.batch_size)
    eval_data_loader = get_loader(utters=load_pickle(val_config.utter_path), batch_size=config.batch_size)

    model_solver = getattr(solvers, "Solver{}".format(config.model))
    solver = model_solver(config, train_data_loader, eval_data_loader, is_train=True)

    solver.build()
    solver.train()
