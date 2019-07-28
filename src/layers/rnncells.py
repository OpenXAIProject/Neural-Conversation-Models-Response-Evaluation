import torch
import torch.nn as nn


class StackedGRUCell(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedGRUCell, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, x, h):
        h_list = []
        for i, layer in enumerate(self.layers):
            h_i = layer(x, h[i])
            x = h_i
            if i + 1 is not self.num_layers:
                x = self.dropout(x)
            h_list.append(h_i)

        last_h = h_list[-1]
        h_list = torch.stack(h_list)

        return last_h, h_list
