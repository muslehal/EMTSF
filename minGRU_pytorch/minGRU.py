# https://arxiv.org/abs/2410.01201v1

import torch
import torch.nn.functional as F
from torch.nn import Linear, Identity, Module

def exists(v):
    return v is not None

# appendix B
# https://github.com/glassroom/heinsen_sequence

def heinsen_associative_scan_log(log_coeffs, log_values):
    a_star = log_coeffs.cumsum(dim = 1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim = 1)
    log_h = a_star + log_h0_plus_b_star
    return log_h.exp()

# appendix B.3

def g(x):
    return torch.where(x >= 0, x + 0.5, x.sigmoid())

def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))


import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

# from minGRU_pytorch.minGRU import minGRU

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * (self.gamma + 1)

def FeedForward(dim, mult = 4):
    dim_inner = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, dim_inner),
        nn.GELU(),
        nn.Linear(dim_inner, dim)
    )

# conv

class CausalDepthWiseConv1d(Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size = kernel_size, groups = dim),
            nn.Conv1d(dim, dim, kernel_size = 1)
        )
    def forward(self, x):
        x = x.transpose(1, 2) # b n d -> b d n
        x = F.pad(x, (self.kernel_size - 1, 0), value = 0.)
        x = self.net(x)
        return x.transpose(1, 2) # b d n -> b n d

# main class
# https://arxiv.org/abs/2410.01201v1

import torch
import torch.nn.functional as F
from torch.nn import Linear, Identity, Module

def exists(v):
    return v is not None

# appendix B
# https://github.com/glassroom/heinsen_sequence

def heinsen_associative_scan_log(log_coeffs, log_values):
    a_star = log_coeffs.cumsum(dim = 1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim = 1)
    log_h = a_star + log_h0_plus_b_star
    return log_h.exp()

# appendix B.3

def g(x):
    return torch.where(x >= 0, x + 0.5, x.sigmoid())

def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

# log-space version of minGRU - B.3.1
# they enforce the hidden states to be positive

class minGRU(Module):
    def __init__(self, dim, expansion_factor = 1.):
        super().__init__()

        dim_inner = int(dim * expansion_factor)
        self.to_hidden_and_gate = Linear(dim, dim_inner * 2, bias = False)
        self.to_out = Linear(dim_inner, dim, bias = False) if expansion_factor != 1. else Identity()

    def forward(self, x, prev_hidden = None, return_next_prev_hidden = False):
        seq_len = x.shape[1]
        hidden, gate = self.to_hidden_and_gate(x).chunk(2, dim = -1)

        if seq_len == 1:
            # handle sequential

            hidden = g(hidden)
            gate = gate.sigmoid()
            out = torch.lerp(prev_hidden, hidden, gate) if exists(prev_hidden) else (hidden * gate)
        else:
            # parallel

            log_coeffs = -F.softplus(gate)

            log_z = -F.softplus(-gate)
            log_tilde_h = log_g(hidden)
            log_values = log_z + log_tilde_h

            if exists(prev_hidden):
                log_values = torch.cat((prev_hidden.log(), log_values), dim = 1)
                log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0))

            out = heinsen_associative_scan_log(log_coeffs, log_values)
            out = out[:, -seq_len:]

        next_prev_hidden = out[:, -1:]

        out = self.to_out(out)

        if not return_next_prev_hidden:
            return out

        return out, next_prev_hidden
    

class minGRULM(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        ff_mult = 4,
        min_gru_expansion = 1.5,
        conv_kernel_size = 3,
        enable_conv = False
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                CausalDepthWiseConv1d(dim, conv_kernel_size) if enable_conv else None,
                RMSNorm(dim),
                minGRU(dim, expansion_factor = min_gru_expansion),
                RMSNorm(dim),
                FeedForward(dim, mult = ff_mult)
            ]))

        self.norm = RMSNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias = False)

        self.can_cache = not enable_conv

    def forward(
        self,
        x,
        return_loss = False,
        return_prev_hiddens = False,
        prev_hiddens = None
    ):

        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        x = self.token_emb(x)

        # handle previous hiddens, for recurrent decoding

        if exists(prev_hiddens):
            x = x[:, -1:]

        next_prev_hiddens = []
        prev_hiddens = iter(default(prev_hiddens, []))

        for conv, norm, mingru, ff_norm, ff in self.layers:

            # conv

            if exists(conv):
                assert not exists(prev_hiddens), 'caching not supported for conv version'
                x = conv(x) + x

            # min gru

            prev_hidden = next(prev_hiddens, None)

            min_gru_out, next_prev_hidden = mingru(
                norm(x),
                prev_hidden,
                return_next_prev_hidden = True
            )

            x = min_gru_out + x
            next_prev_hiddens.append(next_prev_hidden)

            # feedforward

            x = ff(ff_norm(x)) + x

        embed = self.norm(x)
        logits = self.to_logits(embed)

        if not return_loss:
            if not return_prev_hiddens:
                return logits

            return logits, next_prev_hiddens

        loss = F.cross_entropy(
            logits.transpose(1, 2),
            labels
        )

        return loss
    

import torch
import torch.nn as nn
#from minLSTMcell import MinLSTMCell
import torch
import torch.nn as nn

class MinLSTMCell(nn.Module):
    def __init__(self, units, input_shape):
        super(MinLSTMCell, self).__init__()
        self.units = units
        self.input_shape = input_shape
        
        # Initialize the linear layers for the forget gate, input gate, and hidden state transformation
        self.linear_f = nn.Linear(self.input_shape, self.units)
        self.linear_i = nn.Linear(self.input_shape, self.units)
        self.linear_h = nn.Linear(self.input_shape, self.units)

    def forward(self, pre_h, x_t):
        """
        pre_h: (batch_size, units) - previous hidden state (h_prev)
        x_t: (batch_size, input_size) - input at time step t
        """

        # Forget gate: f_t = sigmoid(W_f * x_t)
        f_t = torch.sigmoid(self.linear_f(x_t))  # (batch_size, units)

        # Input gate: i_t = sigmoid(W_i * x_t)
        i_t = torch.sigmoid(self.linear_i(x_t))  # (batch_size, units)

        # Hidden state: tilde_h_t = W_h * x_t
        tilde_h_t = self.linear_h(x_t)  # (batch_size, units)

        # Normalize the gates
        sum_f_i = f_t + i_t
        f_prime_t = f_t / sum_f_i  # (batch_size, units)
        i_prime_t = i_t / sum_f_i  # (batch_size, units)

        # New hidden state: h_t = f_prime_t * pre_h + i_prime_t * tilde_h_t
        h_t = f_prime_t * pre_h + i_prime_t * tilde_h_t  # (batch_size, units)

        return h_t  # (batch_size, units)
    

class MinRNN(nn.Module):
    def __init__(self, units, embedding_size,  input_length):
        super(MinRNN, self).__init__()
        self.input_length = input_length
        self.units = units

        self.embedding = nn.Embedding(self.input_length, embedding_size)
        self.lstm = MinLSTMCell(units, embedding_size)
        self.classification_model = nn.Sequential(
            nn.Linear(units, 64),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, sentence):
        """
        Args:
            sentence: (batch_size, input_length)

        output:
            (batch_size, 1)

        """
        batch_size = sentence.shape[0]

        # Initialize the hidden state, only the h needs to be initialized
        pre_h = torch.zeros(batch_size, self.units, device=sentence.device)

        # Pass the sentence through the embedding layer for the word vectors embeddings
        embedded_sentence = self.embedding(sentence)

        sequence_length = embedded_sentence.shape[1]

        # Pass the entire sequence through the LSTM + hidden_state
        for i in range(sequence_length):
            word = embedded_sentence[:, i, :]  # (batch_size, embedding_size)
            pre_h = self.lstm(pre_h, word)  # Only update h (hidden state)

        return self.classification_model(pre_h)  # Pass the final hidden state into the classification network
    
