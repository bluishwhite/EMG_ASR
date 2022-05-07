import torch
import torch.nn as nn

from src.utils import init as my_init
from .attention import BahdanauAttention


class CLSTMCell(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 context_size,
                 num_layers):

        super(CLSTMCell, self).__init__()

        self.hidden_size = hidden_size
        self.context_size = context_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            nn.LSTMCell(
                input_size=input_size if layer == 0 else self.hidden_size,
                hidden_size=self.hidden_size,
            )
            for layer in range(self.num_layers)
        ])

        self.attn = BahdanauAttention(query_size=hidden_size, key_size=self.context_size)

        self._reset_parameters()

    def _reset_parameters(self):
        for rnn in self.layers:
            for weight in rnn.parameters():
                my_init.rnn_init(weight)

    #
    # @property
    # def context_size(self):
    #     return self.hidden_size * 2

    def forward(self,
                input,
                hidden,
                cell,
                context,
                context_mask=None,
                cache=None):

        next_hiddens, next_cells = [], []
        hiddens = torch.chunk(hidden, self.num_layers, dim=1)
        cells = torch.chunk(cell, self.num_layers, dim=1)
        for i, rnn in enumerate(self.layers):
            # recurrent cell
            next_hidden, next_cell = rnn(input, (hiddens[i], cells[i]))
            input = next_hidden

            # save state for next time step
            next_hiddens.append(next_hidden)
            next_cells.append(next_cell)

        attn_values, _ = self.attn(query=next_hidden, memory=context,
                                   cache=cache, mask=context_mask)

        next_hiddens = torch.cat(next_hiddens, dim=1)
        next_cells = torch.cat(next_cells, dim=1)

        return (next_hidden, attn_values), next_hiddens, next_cells

    def compute_cache(self, memory):
        return self.attn.compute_cache(memory)