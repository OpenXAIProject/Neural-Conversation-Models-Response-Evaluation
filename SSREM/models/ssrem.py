import torch
import torch.nn as nn
from utils import to_var
import torch.nn.functional as torch_f


class ConvScoreSSREM(nn.Module):
    def __init__(self, config):
        super(ConvScoreSSREM, self).__init__()

        self.config = config
        self.mat_M = nn.Parameter(torch.randn([config.embedding_size, config.embedding_size],
                                              dtype=torch.float32), requires_grad=True)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.score_active_func = nn.Tanh()

    def forward(self, contexts, res_pos, res_neg1=None, res_neg2=None, res_neg3=None, res_neg4=None):
        """
        Forwarding
        :param contexts: [batch_size, emb_size]
        :param res_pos: [batch_size, emb_size]
        :param res_neg1: [batch_size, emb_size]
        :param res_neg2: [batch_size, emb_size]
        :param res_neg3: [batch_size, emb_size]
        :param res_neg4: [batch_size, emb_size]
        :return: [batch_size]
        """
        ress = torch.cat([self._score(contexts, res_pos),
                          self._score(contexts, res_neg1),
                          self._score(contexts, res_neg2),
                          self._score(contexts, res_neg3),
                          self._score(contexts, res_neg4)], dim=1)

        log_ress_softmax = self.logsoftmax(ress)
        return log_ress_softmax[:, 0]

    def _score(self, contexts, res):
        """
        Scoring function
        :param contexts: [batch_size, emb_size]
        :param res: [batch_size, emb_size]
        :return: [batch_size, 1]
        """
        res_t = torch.transpose(res, 0, 1)
        score_compute_bb = torch.mm(torch.mm(contexts, self.mat_M), res_t)
        scores = torch.diagonal(score_compute_bb, 0)

        return torch.unsqueeze(scores, dim=1)

    def score(self, contexts, res):
        """
        Scoring function
        :param contexts: [batch_size, emb_size]
        :param res: [batch_size, emb_size]
        :return: [batch_size]
        """
        res_t = torch.transpose(res, 0, 1)
        score_compute_bb = torch.mm(torch.mm(contexts, self.mat_M), res_t)
        scores = torch.diagonal(score_compute_bb, 0)

        return self.score_active_func(scores)
