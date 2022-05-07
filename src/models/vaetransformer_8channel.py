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

import numpy as np
from numpy.lib.function_base import select
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.decoding.utils import tile_batch, tensor_gather_helper
from src.models.base import NMTModel
from src.modules.basic import BottleLinear as Linear
from src.modules.embeddings import Embeddings
from src.modules.sublayers import PositionwiseFeedForward, MultiHeadedAttention
from src.utils import nest


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.rand_like(std)
    return eps * std + mu

def get_attn_causal_mask(seq):
    ''' Get an attention mask to avoid using the subsequent info.

    :param seq: Input sequence.
        with shape [batch_size, time_steps, dim]
    '''
    assert seq.dim() == 3
    attn_shape = (seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('bool')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()
    return subsequent_mask

class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, d_model, dropout_rate=0.0, max_len=5000):
        """Initialize class.

        :param int d_model: embedding dim
        :param float dropout_rate: dropout rate
        :param int max_len: maximum input length

        """
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) * 
                             -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)

        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Conv2dSubsampling(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length)

    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate
    """

    def __init__(self, idim, odim, dropout_rate=0.0):
        super(Conv2dSubsampling, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(8, odim, 3, 2),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 3, 2),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(odim * (((idim//8 - 1) // 2 - 1) // 2), odim),
            PositionalEncoding(odim, dropout_rate)
        )

    def forward(self, x ,x_mask):
        """Subsample x

        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]
        """
        # x = x.unsqueeze(1)  # (b, c, t, f)
        x = x.transpose(2,1)
        x = self.conv(x)
        
        b, c, t, f = x.size()
        # x = 
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))

        if x_mask is None:
            return x ,None
        return x, x_mask[:,:, :-2:2][:,:, :-2:2]

class EncoderBlock(nn.Module):

    def __init__(self,  d_model, d_inner_hid, n_head, dim_per_head, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)

        self.slf_attn = MultiHeadedAttention(head_count=n_head, model_dim=d_model, dropout=dropout,
                                             dim_per_head=dim_per_head)

        self.pos_ffn = PositionwiseFeedForward(size=d_model, hidden_size=d_inner_hid, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input, slf_attn_mask=None):

        input_norm = self.layer_norm(enc_input)
        context, _, _ = self.slf_attn(input_norm, input_norm, input_norm, slf_attn_mask)
        out = self.dropout(context) + enc_input

        return self.pos_ffn(out)

class ResBlock(nn.Module):
    def __init__(self, num_ins, num_outs, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(num_ins, num_outs, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(num_outs)
        self.conv2 = nn.Conv1d(num_outs, num_outs, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_outs)

        if stride != 1 or num_ins != num_outs:
            self.residual_path = nn.Conv1d(num_ins, num_outs, 1, stride=stride)
            self.res_norm = nn.BatchNorm1d(num_outs)
        else:
            self.residual_path = None

    def forward(self, x):
        input_value = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.residual_path is not None:
            res = self.res_norm(self.residual_path(input_value))
        else:
            res = input_value

        return F.relu(x + res)


class Conv1dSubsampling(nn.Module):
    def __init__(self, odim, dropout_rate=0.0):
        super(Conv1dSubsampling, self).__init__()
        self.conv = nn.Sequential(
            ResBlock(8, odim, 2),
            ResBlock(odim, odim, 2),
            # ResBlock(d_model, d_model, 2),
        )
        self.out = nn.Sequential(
            nn.Linear(odim, odim),
            PositionalEncoding(odim, dropout_rate)
        )

    def forward(self,x,x_mask):
        x = x.transpose(1,2)
        x = self.conv(x)
        x = x.transpose(1,2)
        x = self.out(x)
        if x_mask is None:
            return x ,None
        return x, x_mask[:,:, :-2:2][:,:, :-2:2]



class Encoder(nn.Module):

    def __init__(
            self, input_size, n_layers=6, n_head=8,
            d_model=512, d_inner_hid=1024, dropout=0.1, dim_per_head=None, conv2d=True, resnet=False,joint_ctc=False,max_n_words=-1):
        super().__init__()

        self.num_layers = n_layers
        
        
        self.embeddings = Conv2dSubsampling(input_size,d_model,dropout)
        self.block_stack = nn.ModuleList(
            [EncoderBlock(d_model=d_model, d_inner_hid=d_inner_hid, n_head=n_head, dropout=dropout,
                          dim_per_head=dim_per_head)
             for _ in range(n_layers)])

        self.layer_norm = nn.LayerNorm(d_model)
        self.joint_ctc = joint_ctc
        if self.joint_ctc:
            
            self.fc = Linear(d_model, max_n_words, bias=False)

    def forward(self, inputs):
        # Word embedding look up
        enc_mask = torch.sum(inputs, dim=-1).eq(0).unsqueeze(-2)
        inputs = torch.stack(inputs.split(36,2),2)
        enc_output, enc_mask = self.embeddings(inputs, enc_mask)
        
        enc_output.masked_fill_(enc_mask.transpose(1, 2), 0.0)
        temp = torch.sum(enc_output,dim=-1).ne(0)
        encoder_out_length = torch.sum(temp.eq(True),dim=-1)

        batch_size , _, time_len = enc_mask.size()
        enc_slf_attn_mask = enc_mask.expand(batch_size,time_len,time_len)

        """
        batch_size, src_len= inputs.size()
        enc_mask = inputs.detach().eq(0)
        enc_slf_attn_mask = enc_mask.unsqueeze(1).expand(batch_size, src_len, src_len)
        out = inputs
        """

        encoder_log_pro = None

        for i in range(self.num_layers):
            enc_output = self.block_stack[i](enc_output, enc_slf_attn_mask)

        out = self.layer_norm(enc_output)
        if self.joint_ctc:
            encoder_log_pro = self.fc(out).log_softmax(dim=-1)

        return out, enc_mask, encoder_log_pro,encoder_out_length


class DecoderBlock(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner_hid, n_head, dim_per_head, dropout=0.1):
        super(DecoderBlock, self).__init__()

        self.slf_attn = MultiHeadedAttention(head_count=n_head, model_dim=d_model, dropout=dropout,
                                             dim_per_head=dim_per_head)
        self.ctx_attn = MultiHeadedAttention(head_count=n_head, model_dim=d_model, dropout=dropout,
                                             dim_per_head=dim_per_head)
        self.pos_ffn = PositionwiseFeedForward(size=d_model, hidden_size=d_inner_hid)

        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def compute_cache(self, enc_output):
        return self.ctx_attn.compute_cache(enc_output, enc_output)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None,
                enc_attn_cache=None, self_attn_cache=None):
        # Args Checks
        input_batch, input_len, _ = dec_input.size()

        contxt_batch, contxt_len, _ = enc_output.size()

        input_norm = self.layer_norm_1(dec_input)
        all_input = input_norm

        query, _, self_attn_cache = self.slf_attn(all_input, all_input, input_norm,
                                                  mask=slf_attn_mask, self_attn_cache=self_attn_cache)

        query = self.dropout(query) + dec_input

        query_norm = self.layer_norm_2(query)
        mid, attn, enc_attn_cache = self.ctx_attn(enc_output, enc_output, query_norm,
                                                  mask=dec_enc_attn_mask, enc_attn_cache=enc_attn_cache)

        output = self.pos_ffn(self.dropout(mid) + query)

        return output, attn, self_attn_cache,enc_attn_cache


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, vocab, n_layers=6, n_head=8,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dim_per_head=None, dropout=0.1):

        super(Decoder, self).__init__()

        self.pad = vocab.pad()
        self.n_words = vocab.max_n_words

        self.n_head = n_head
        self.num_layers = n_layers
        self.d_model = d_model

        self.embeddings = Embeddings(self.n_words, d_word_vec,
                                     dropout=dropout, add_position_embedding=True)

        self.block_stack = nn.ModuleList([
            DecoderBlock(d_model=d_model, d_inner_hid=d_inner_hid, n_head=n_head, dropout=dropout,
                         dim_per_head=dim_per_head)
            for _ in range(n_layers)])

        self.out_layer_norm = nn.LayerNorm(d_model)

        self._dim_per_head = dim_per_head

    @property
    def dim_per_head(self):
        if self._dim_per_head is None:
            return self.d_model // self.n_head
        else:
            return self._dim_per_head

    def forward(self, tgt_seq, enc_output, enc_mask, enc_attn_caches=None, self_attn_caches=None):

        batch_size, tgt_len = tgt_seq.size()

        query_len = tgt_len
        key_len = tgt_len

        src_len = enc_output.size(1)

        # Run the forward pass of the TransformerDecoder.
        #print(tgt_seq.dtype)
        emb = self.embeddings(tgt_seq)

        if self_attn_caches is not None:
            emb = emb[:, -1:].contiguous()
            query_len = 1

        # Decode mask
        
        dec_slf_attn_pad_mask = tgt_seq.detach().eq(self.pad).unsqueeze(1).expand(batch_size, query_len, key_len)
        dec_slf_attn_sub_mask = get_attn_causal_mask(emb)

        dec_slf_attn_mask = torch.gt(dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)
        #print("-----")
        #print(dec_slf_attn_mask.shape)

        dec_enc_attn_mask = enc_mask.expand(batch_size,query_len,src_len)

        output = emb
        new_self_attn_caches = []
        new_enc_attn_caches = []
        for i in range(self.num_layers):
            output, attn, self_attn_cache, enc_attn_cache \
                = self.block_stack[i](output,
                                      enc_output,
                                      dec_slf_attn_mask,
                                      dec_enc_attn_mask,
                                      enc_attn_cache=enc_attn_caches[i] if enc_attn_caches is not None else None,
                                      self_attn_cache=self_attn_caches[i] if self_attn_caches is not None else None)

            new_self_attn_caches += [self_attn_cache]
            new_enc_attn_caches += [enc_attn_cache]

        output = self.out_layer_norm(output)

        return output, new_self_attn_caches, new_enc_attn_caches


class Generator(nn.Module):

    def __init__(self, vocab, hidden_size, shared_weight=None):
        super(Generator, self).__init__()

        self.pad = vocab.pad()
        self.n_words = vocab.max_n_words

        self.hidden_size = hidden_size

        self.proj = Linear(self.hidden_size, self.n_words, bias=False)

        if shared_weight is not None:
            self.proj.linear.weight = shared_weight

    def _pad_2d(self, x):

        if self.pad == -1:
            return x
        else:
            x_size = x.size()
            x_2d = x.view(-1, x.size(-1))

            mask = x_2d.new(1, x_2d.size(-1)).zero_()
            mask[0][self.pad] = float('-inf')
            x_2d = x_2d + mask

            return x_2d.view(x_size)

    def forward(self, input, log_probs=True):
        """
        input == > Linear == > LogSoftmax
        """

        logits = self.proj(input)

        logits = self._pad_2d(logits)

        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)
        


class VaeCovn2dTransformer(NMTModel):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, input_size, tgt_vocab, n_layers=6, n_head=8,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dim_per_head=None,
            dropout=0.1, proj_share_weight=True,joint_ctc=False, **kwargs):

        super().__init__(tgt_vocab)

        self.encoder = Encoder(input_size=input_size,
                               n_layers=n_layers, n_head=n_head,
                               d_model=d_model,
                               d_inner_hid=d_inner_hid, dropout=dropout, dim_per_head=dim_per_head,joint_ctc=joint_ctc,max_n_words=self.n_tgt_vocab)

        self.mean_cal = nn.Linear(d_model, d_model // 8)
        self.logvar_cal = nn.Linear(d_model, d_model // 8)

        self.decoder = Decoder(vocab=self.tgt_vocab,
                               n_layers=3, n_head=4,
                               d_word_vec=d_word_vec+d_model // 8, d_model=d_model+d_model // 8,
                               d_inner_hid=d_inner_hid, dropout=0.1, dim_per_head=dim_per_head)

        self.dropout = nn.Dropout(dropout)

        assert d_model == d_word_vec, \
            'To facilitate the residual connections, \
             the dimensions of all module output shall be the same.'

        if proj_share_weight:
            self.generator = Generator(vocab=self.tgt_vocab,
                                       hidden_size=d_word_vec,
                                       shared_weight=self.decoder.embeddings.embeddings.weight)
        else:
            self.generator = Generator(vocab=self.tgt_vocab, hidden_size=d_word_vec)

    def forward(self, src_seq, tgt_seq, log_probs=True):
        #enc_output shape:[batch, timestep,d_model]
        enc_output, enc_mask,encoder_log_pro,encoder_output_length = self.encoder(src_seq)

        mu = self.mean_cal(enc_output)
        mu = mu.masked_fill_(enc_mask.transpose(1, 2), 0.0)
        mu_mean = torch.mean(mu,dim=1,keepdim=True).squeeze(1)

        log_var = self.logvar_cal(enc_output)
        log_var = log_var.masked_fill_(enc_mask.transpose(1, 2), 0.0)
        log_var_mean = torch.mean(log_var, dim=1, keepdim=True).squeeze(1)

        z = reparameterize(mu_mean, log_var_mean)

        batch_size, time_len, z_model = mu.size()
        z = z.unsqueeze(1).expand(batch_size, time_len, z_model)

        enc_output = torch.cat((enc_output, z), dim=2)
        # print(enc_output.shape)
        dec_output, _, _ = self.decoder(tgt_seq, enc_output, enc_mask)

        return self.generator(dec_output, log_probs=log_probs),encoder_log_pro,encoder_output_length,mu_mean, log_var_mean


    def encode(self, src_seq):

        enc_output, ctx_mask,encoder_log_pro,encoder_output_length = self.encoder(src_seq)
        mu = self.mean_cal(enc_output)
        mu = mu.masked_fill_(ctx_mask.transpose(1, 2), 0.0)
        mu_mean = torch.mean(mu, dim=1, keepdim=True).squeeze(1)

        log_var = self.logvar_cal(enc_output)
        log_var = log_var.masked_fill_(ctx_mask.transpose(1, 2), 0.0)
        log_var_mean = torch.mean(log_var, dim=1, keepdim=True).squeeze(1)

        z = reparameterize(mu_mean, log_var_mean)

        batch_size, time_len, z_model = mu.size()
        z = z.unsqueeze(1).expand(batch_size, time_len, z_model)

        ctx = torch.cat((enc_output, z), dim=2)

        return {"ctx": ctx, "ctx_mask": ctx_mask,"encoder_log_pro":encoder_log_pro,"encoder_output_length":encoder_output_length}

    def init_decoder(self, enc_outputs, expand_size=1):

        ctx = enc_outputs['ctx']

        ctx_mask = enc_outputs['ctx_mask']

        if expand_size > 1:
            ctx = tile_batch(ctx, multiplier=expand_size)
            ctx_mask = tile_batch(ctx_mask, multiplier=expand_size)

        return {
            "ctx": ctx,
            "ctx_mask": ctx_mask,
            "enc_attn_caches": None,
            "slf_attn_caches": None
        }

    def decode(self, tgt_seq, dec_states, log_probs=True):

        ctx = dec_states["ctx"]
        ctx_mask = dec_states['ctx_mask']
        enc_attn_caches = dec_states['enc_attn_caches']
        slf_attn_caches = dec_states['slf_attn_caches']

        dec_output, slf_attn_caches, enc_attn_caches = self.decoder(tgt_seq=tgt_seq, enc_output=ctx, enc_mask=ctx_mask,
                                                                    enc_attn_caches=enc_attn_caches,
                                                                    self_attn_caches=slf_attn_caches)

        next_scores = self.generator(dec_output[:, -1].contiguous(), log_probs=log_probs)

        dec_states['enc_attn_caches'] = enc_attn_caches
        dec_states['slf_attn_caches'] = slf_attn_caches

        return next_scores, dec_states

    def reorder_dec_states(self, dec_states, new_beam_indices, beam_size):

        slf_attn_caches = dec_states['slf_attn_caches']

        batch_size = slf_attn_caches[0][0].size(0) // beam_size

        n_head = self.decoder.n_head
        dim_per_head = self.decoder.dim_per_head

        slf_attn_caches = nest.map_structure(
            lambda t: tensor_gather_helper(gather_indices=new_beam_indices,
                                           gather_from=t,
                                           batch_size=batch_size,
                                           beam_size=beam_size,
                                           gather_shape=[batch_size * beam_size, n_head, -1, dim_per_head]),
            slf_attn_caches)

        dec_states['slf_attn_caches'] = slf_attn_caches

        return dec_states
