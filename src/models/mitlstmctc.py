# MIT License

# Copyright (c) 2018 the NJUNMT-pytorch authors.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear

import src.utils.init as my_init
from src.decoding.utils import tile_batch, tensor_gather_helper
from src.modules.clstm import CLSTMCell
from src.modules.rnn import RNN
from .base import NMTModel

class Encoder(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 n_layers,
                 dropout=0.1,
                 rnn_dropout=0.4,
                 max_n_words = -1
                 ):
        super(Encoder, self).__init__()
        
        # self.pad = vocab.pad()
        # self.n_words = vocab.max_n_words

        self.num_layers = n_layers


        self.layers = nn.ModuleList([
            RNN(type="lstm", batch_first=True, input_size=input_size if layer == 0 else 2*hidden_size,
                hidden_size=hidden_size, dropout=rnn_dropout, bidirectional=True)
            for layer in range(self.num_layers)
        ])

        self.linear1 = Linear(hidden_size*2,1024)
        self.relu = nn.ReLU(inplace=False)
        self.linear2 = Linear(1024,max_n_words)

    def forward(self, x):
        """
        :param x: Input sequence.
            with shape [batch_size, seq_len, input_size]
        """
        # x_mask = x.detach().eq(0)
        # !x_mask = torch.sum(x, dim=-1).eq(0)
        enc_mask = torch.sum(x, dim=-1).eq(0).unsqueeze(-2)
        x_mask = enc_mask.squeeze(-2)

        final_hiddens, final_cells = [], []

        temp = torch.sum(x,dim=-1).ne(0)
        encoder_output_length = torch.sum(temp.eq(True),dim=-1)
        for i, rnn in enumerate(self.layers):
            # recurrent cell
            ctx, (final_hidden, final_cell) = rnn(x, x_mask)
            x = ctx

            # save state for next time step
            final_hidden = torch.chunk(final_hidden, 2, dim=0)
            final_hidden = torch.cat(final_hidden, dim=2).squeeze(0)
            final_hiddens.append(final_hidden)
            final_cell = torch.chunk(final_cell, 2, dim=0)
            final_cell = torch.cat(final_cell, dim=2).squeeze(0)
            final_cells.append(final_cell)

        final_hiddens = torch.cat(final_hiddens, dim=1)
        final_cells = torch.cat(final_cells, dim=1)

        ctx = self.relu(self.linear1(ctx))
        encoder_log_pro=self.linear2(ctx).log_softmax(dim=-1)
        
        return ctx, x_mask, final_hiddens, final_cells,encoder_log_pro,encoder_output_length





class MITLSTM4ETCTC(NMTModel):

    def __init__(self,feature_size, tgt_vocab, n_layers=1, d_word_vec=512, d_model=512, dropout=0.5,
                 proj_share_weight=False,joint_ctc=False, **kwargs):

        super().__init__(tgt_vocab)
        self.num_layers = n_layers

        self.encoder = Encoder(input_size=feature_size,
                               hidden_size=d_model, n_layers=n_layers,max_n_words=tgt_vocab.max_n_words)


    def forward(self, src_seq, log_probs=True):

        ctx, ctx_mask, final_hidden, final_cell,encoder_log_pro,encoder_output_length = self.encoder(src_seq)


        return ctx, ctx_mask,encoder_log_pro,encoder_output_length

    def encode(self, src_seq):

        ctx, ctx_mask, final_hiddens, final_cells,encoder_log_pro,encoder_output_length = self.encoder(src_seq)

        return {"ctx": ctx, "ctx_mask": ctx_mask,
                "hiddens": final_hiddens, "cells": final_cells,"encoder_log_pro":encoder_log_pro,"encoder_output_length":encoder_output_length}

