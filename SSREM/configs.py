import os
import argparse
from datetime import datetime
from pathlib import Path
import pprint
from torch import optim
import platform
import torch.nn as nn

project_dir = Path(__file__).resolve().parent.parent
optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}
plt_sys = platform.system()
rnn_dict = {'lstm': nn.LSTM, 'gru': nn.GRU}

if "Windows" in plt_sys:
    save_dir = Path(f"D:/git/conversation-metrics/results/")
elif "Linux" in plt_sys:
    username = Path.home().name
    save_dir = Path(f'/home/{username}/git/conversation-metrics/results/')


def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        if kwargs is not None:
            for key, value in kwargs.items():
                if key == 'optimizer':
                    value = optimizer_dict[value]
                if key == 'rnn':
                    value = rnn_dict[value]
                setattr(self, key, value)

        data_dir = project_dir.joinpath('datasets')

        self.dataset_dir = data_dir.joinpath(self.data)

        self.data_dir = self.dataset_dir.joinpath(self.mode)
        self.word2id_path = self.dataset_dir.joinpath('word2id.pkl')
        self.id2word_path = self.dataset_dir.joinpath('id2word.pkl')

        self.utter_path = self.data_dir.joinpath('utters.pkl')
        self.utter_length_path = self.data_dir.joinpath('utters_length.pkl')
        self.utter_scores_path = self.data_dir.joinpath('utters_scores.pkl')

        if self.mode == 'train' and self.checkpoint is None:
            time_now = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.save_path = save_dir.joinpath(self.data, self.model, time_now)
            self.logdir = str(self.save_path)
            os.makedirs(self.save_path, exist_ok=True)
        elif self.checkpoint is not None:
            assert os.path.exists(self.checkpoint)
            self.save_path = os.path.dirname(self.checkpoint)
            self.logdir = str(self.save_path)

        self.pretrained_wv_path = None
        
        if 'bert' in self.data:
            self.embedding_size = 768
        if '_25' in self.data:
            self.embedding_size = 25

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train')

    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--n_epoch', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=1e-12)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--context_size', type=int, default=3)

    parser.add_argument('--rnn', type=str, default='gru')
    parser.add_argument('--rnncell', type=str, default='gru')
    parser.add_argument('--embedding_size', type=int, default=200)
    parser.add_argument('--encoder_hidden_size', type=int, default=500)
    parser.add_argument('--bidirectional', type=str2bool, default=True)
    parser.add_argument('--adem_gamma', type=float, default=0.01)

    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--num_layers', type=int, default=1)

    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--plot_every_epoch', type=int, default=1)
    parser.add_argument('--save_every_epoch', type=int, default=100)

    parser.add_argument('--data', type=str, default='ubuntu')
    parser.add_argument('--pretrained_wv', type=str2bool, default=False)

    parser.add_argument('--test_target_ng', type=int, default=4)

    # Parse arguments
    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)
