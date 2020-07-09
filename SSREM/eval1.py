from configs import get_config
from utils import get_loader
import os
import pickle
import solvers
import numpy as np
import scipy.stats
import codecs
import json


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    config = get_config(mode='test')
    print(config)
    with open(os.path.join(config.save_path, 'config_eval1.txt'), 'w') as f:
        print(config, file=f)

    eval_data_loader = get_loader(utters=load_pickle(config.utter_path), batch_size=config.batch_size)

    model_solver = getattr(solvers, "Solver{}".format(config.model))
    solver = model_solver(config, None, eval_data_loader, is_train=False)

    solver.build()
    true_scores_list, false_scores_list = solver.test()

    print("True Score Mean:\t{}".format(np.mean(true_scores_list)))
    print("True Score std:\t{}".format(np.std(true_scores_list)))
    print("True Score err:\t{}".format(scipy.stats.sem(true_scores_list)))
    print("False Score Mean:\t{}".format(np.mean(false_scores_list)))
    print("False Score std:\t{}".format(np.std(false_scores_list)))
    print("False Score err:\t{}".format(scipy.stats.sem(false_scores_list)))

    output_file_path = os.path.join(config.save_path,
                                    "test_{}_scores_from_{}_{}.json".format(config.data, config.model, solver.epoch_i))
    with codecs.open(output_file_path, "w", "utf-8") as json_f:
        json.dump([true_scores_list, false_scores_list], json_f)
