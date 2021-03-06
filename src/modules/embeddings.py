import math
import torch
import torch.nn as nn
import src.utils.init as my_init
from src.data.vocabulary import PAD
from torch import Tensor


class Embeddings(nn.Module):

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 dropout=0.0,
                 add_position_embedding=True,
                 padding_idx=PAD):

        super().__init__()


        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.padding_idx = padding_idx

        self.embeddings = nn.Embedding(num_embeddings=num_embeddings,
                                       embedding_dim=embedding_dim,
                                       padding_idx=self.padding_idx)

        self.add_position_embedding = add_position_embedding

        self.scale = embedding_dim ** 0.5

        self.reset_parameters()

    def reset_parameters(self):
        if self.add_position_embedding:
            nn.init.uniform_(self.embeddings.weight, - 1.0 / self.scale, 1.0 / self.scale)
        else:
            my_init.embedding_init(self.embeddings.weight)

        with torch.no_grad():
            self.embeddings.weight[self.padding_idx].fill_(0.0)

    def _add_pos_embedding(self, x, min_timescale=1.0, max_timescale=1.0e4):

        batch, length, channels = list(x.size())
        assert (channels % 2 == 0)
        num_timescales = channels // 2
        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                (float(num_timescales) - 1.))
        position = torch.arange(0, length).float()
        inv_timescales = torch.arange(0, num_timescales).float()
        if x.is_cuda:
            position = position.cuda()
            inv_timescales = inv_timescales.cuda()

        inv_timescales.mul_(-log_timescale_increment).exp_().mul_(min_timescale)
        scaled_time = position.unsqueeze(1).expand(
            length, num_timescales) * inv_timescales.unsqueeze(0).expand(length, num_timescales)
        # scaled time is now length x num_timescales
        # length x channels
        signal = torch.cat([scaled_time.sin(), scaled_time.cos()], 1)

        return signal.unsqueeze(0).expand(batch, length, channels)

    def forward(self, x):

        emb = self.embeddings(x)
         # rescale to [-1.0, 1.0]
        if self.add_position_embedding:
            emb = emb * self.scale
            emb += self._add_pos_embedding(emb)

        if self.dropout is not None:
            emb = self.dropout(emb)

        return emb




class PositionalEncoding(nn.Module):
    """
    Positional Encoding proposed in "Attention Is All You Need".
    Since transformer contains no recurrence and no convolution, in order for the model to make
    use of the order of the sequence, we must add some positional information.
    "Attention Is All You Need" use sine and cosine functions of different frequencies:
        PE_(pos, 2i)    =  sin(pos / power(10000, 2i / d_model))
        PE_(pos, 2i+1)  =  cos(pos / power(10000, 2i / d_model))
    """
    def __init__(self, d_model: int = 512, max_len: int = 10000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length: int) -> Tensor:
        return self.pe[:, :length]