import torch
import models
from utils import TensorboardWriter
import os
import re


class Solver(object):
    def __init__(self, config, train_data_loader, eval_data_loader, is_train=True, model=None):
        self.config = config
        self.epoch_i = 0
        self.train_data_loader = train_data_loader
        self.eval_data_loader = eval_data_loader
        self.is_train = is_train
        self.model = model
        self.writer = None
        self.optimizer = None
        self.epoch_loss = None
        self.validation_loss = None
        self.true_scores = 0
        self.false_scores = 0
        self.eval_epoch_loss = 0

    def build(self, cuda=True):
        if self.model is None:
            self.model = getattr(models, self.config.model)(self.config)

        if torch.cuda.is_available() and cuda:
            self.model.cuda()

        if self.config.checkpoint:
            self.load_model(self.config.checkpoint)

        if self.is_train:
            self.writer = TensorboardWriter(self.config.logdir)
            self.optimizer = self.config.optimizer(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                   lr=self.config.learning_rate)

    def save_model(self, epoch):
        ckpt_path = os.path.join(self.config.save_path, f'{epoch}.pkl')
        print(f'Save parameters to {ckpt_path}')
        torch.save(self.model.state_dict(), ckpt_path)

    def load_model(self, checkpoint):
        print(f'Load parameters from {checkpoint}')
        epoch = re.match(r"[0-9]*", os.path.basename(checkpoint)).group(0)
        self.epoch_i = int(epoch)
        self.model.load_state_dict(torch.load(checkpoint))

    def write_summary(self, epoch_i):
        epoch_loss = getattr(self, 'epoch_loss', None)
        if epoch_loss is not None:
            self.writer.update_loss(loss=epoch_loss, step_i=epoch_i + 1, name='train_loss')

        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError
