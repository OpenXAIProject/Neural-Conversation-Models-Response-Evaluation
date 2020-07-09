from .convscore_solver import SolverConvScore
import torch
from utils import to_var
from tqdm import tqdm
from math import isnan
import numpy as np


class SolverConvScoreSSREM(SolverConvScore):
    def __init__(self, config, train_data_loader, eval_data_loader, is_train=True, model=None):
        super(SolverConvScoreSSREM, self).__init__(config, train_data_loader, eval_data_loader, is_train, model)

    def train(self):
        epoch_loss_history = []
        for epoch_i in range(self.epoch_i, self.config.n_epoch):
            self.epoch_i = epoch_i
            batch_loss_history = []
            self.model.train()
            for batch_i, (contexts, res_true, res_ns1, res_ns2, res_ns3, res_ns4) in \
                    enumerate(tqdm(self.train_data_loader, ncols=80)):
                contexts = to_var(torch.FloatTensor(contexts))
                res_trues = to_var(torch.FloatTensor(res_true))
                res_ns1 = to_var(torch.FloatTensor(res_ns1))
                res_ns2 = to_var(torch.FloatTensor(res_ns2))
                res_ns3 = to_var(torch.FloatTensor(res_ns3))
                res_ns4 = to_var(torch.FloatTensor(res_ns4))

                self.optimizer.zero_grad()

                # Call forward function
                batch_loglikelihood = self.model(contexts, res_trues, res_ns1, res_ns2, res_ns3, res_ns4)
                batch_loss = -torch.sum(batch_loglikelihood)

                assert not isnan(batch_loss.item())
                batch_loss_history.append(batch_loss.item())

                if batch_i % self.config.print_every == 0:
                    tqdm.write(f'Epoch: {epoch_i+1}, iter {batch_i}: loss = {batch_loss.item():.3f}')

                # Back-propagation
                batch_loss.backward()

                # Gradient cliping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)

                # Run optimizer
                self.optimizer.step()

            epoch_loss = np.mean(batch_loss_history)
            epoch_loss_history.append(epoch_loss)
            self.epoch_loss = epoch_loss

            if epoch_i % self.config.save_every_epoch == 0:
                self.save_model(epoch_i + 1)

            self.true_scores, self.false_scores, self.eval_epoch_loss = self.evaluate()

            print_str = f'Epoch {epoch_i + 1} loss average: {epoch_loss:.3f}, ' \
                        f'True:{self.true_scores:.3f}, False:{self.false_scores:.3f}, ' \
                        f'Validation loss: {self.eval_epoch_loss:.3f}'
            print(print_str)

            if epoch_i % self.config.plot_every_epoch == 0:
                self.write_summary(epoch_i)

        self.save_model(self.config.n_epoch)

        return epoch_loss_history

    def evaluate(self):
        self.model.eval()
        true_scores_list = list()
        false_scores_list = list()
        batch_loss_history = list()
        for batch_i, (contexts, res_true, res_ns1, res_ns2, res_ns3, res_ns4) in \
                enumerate(tqdm(self.eval_data_loader, ncols=80)):
            with torch.no_grad():
                contexts = to_var(torch.FloatTensor(contexts))
                res_trues = to_var(torch.FloatTensor(res_true))
                res_ns1 = to_var(torch.FloatTensor(res_ns1))
                res_ns2 = to_var(torch.FloatTensor(res_ns2))
                res_ns3 = to_var(torch.FloatTensor(res_ns3))
                res_ns4 = to_var(torch.FloatTensor(res_ns4))

            # Call forward function
            true_scores = self.model.score(contexts, res_trues)
            false_scores = self.model.score(contexts, res_ns4)

            true_scores_list += true_scores.data.cpu().numpy().tolist()
            false_scores_list += false_scores.data.cpu().numpy().tolist()

            # Call forward function
            batch_loglikelihood = self.model(contexts, res_trues, res_ns1, res_ns2, res_ns3, res_ns4)
            batch_loss = -torch.sum(batch_loglikelihood)

            assert not isnan(batch_loss.item())
            batch_loss_history.append(batch_loss.item())

        epoch_loss = np.sum(batch_loss_history)

        return np.mean(true_scores_list), np.mean(false_scores_list), epoch_loss
