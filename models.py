
import numpy as np
import torch
import torch.nn as nn
import torch.fft
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch import nn
import argparse
from einops import rearrange
from src.models.Autoformer import series_decomp, series_decomp_multi, my_Layernorm, moving_avg
from src.models.layers.revin import RevIN   
from StandardNorm import Normalize as RevIN
import argparse
import numpy as np
from xlstm1.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig

from xlstm1.blocks.mlstm.block import mLSTMBlockConfig
from xlstm1.blocks.slstm.block import sLSTMBlockConfig

from src.models.layers.revin import RevIN   
from minGRU_pytorch.minGRULM import minGRULM 
from minGRU_pytorch.minGRU import minGRU, MinRNN
from packaging.version import Version
import torch.optim as optim

from math import fabs
from pickle import TRUE
from re import T
import os
import math

from xlstm1.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig

from xlstm1.blocks.mlstm.block import mLSTMBlockConfig
from xlstm1.blocks.slstm.block import sLSTMBlockConfig


mlstm_config = mLSTMBlockConfig()
slstm_config = sLSTMBlockConfig()



config = xLSTMBlockStackConfig(
        mlstm_block=mlstm_config,
        slstm_block=slstm_config,
        num_blocks=6,
        embedding_dim=256,
        add_post_blocks_norm=True,
        
        _block_map = 1,

        slstm_at="all",
        context_length=512
    )

    



class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
    


class series_decomp2(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp2, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class LD(nn.Module):
    def __init__(self, kernel_size=25):
        super(LD, self).__init__()
        # Define a shared convolution layers for all channels
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, stride=1, padding=int(kernel_size // 2),
                              padding_mode='replicate', bias=True)
        # Define the parameters for Gaussian initialization
        kernel_size_half = kernel_size // 2
        sigma = 1.0  # 1 for variance
        weights = torch.zeros(1, 1, kernel_size)
        for i in range(kernel_size):
            weights[0, 0, i] = math.exp(-((i - kernel_size_half) / (2 * sigma)) ** 2)

        # Set the weights of the convolution layer
        self.conv.weight.data = F.softmax(weights, dim=-1)
        self.conv.bias.data.fill_(0.0)

    def forward(self, inp):
        # Permute the input tensor to match the expected shape for 1D convolution (B, N, T)
        inp = inp.permute(0, 2, 1)
        # Split the input tensor into separate channels
        input_channels = torch.split(inp, 1, dim=1)

        # Apply convolution to each channel
        conv_outputs = [self.conv(input_channel) for input_channel in input_channels]

        # Concatenate the channel outputs
        out = torch.cat(conv_outputs, dim=1)
        out = out.permute(0, 2, 1)
        return out
    



# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp


class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        c_in = configs.enc_in
        context_window = configs.context_points
        target_window = configs.target_points
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = 0.0#configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual =False #configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = 'end'#configs.padding_patch
        
        revin = configs.revin
        affine = 1# configs.affine
        subtract_last = 0#configs.subtract_last
        
        decomposition = 1#configs.decomposition
        kernel_size = 25#configs.kernel_size

        
        
        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
    
    
    def forward(self, x):           # x: [Batch, Input length, Channel]

        # print(x.shape)
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        # print(x.shape)
        return x



class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series.
    """
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        # x shape: (batch_size, seq_len, num_features)
        padding_size = (self.kernel_size - 1) // 2
        # Pad the sequence with replication of edge values
        x = x.permute(0, 2, 1)  # (batch_size, num_features, seq_len)
        x = F.pad(x, (padding_size, padding_size), mode='replicate')
        x = self.avg(x)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, num_features)
        return x


class SeriesDecomp9(nn.Module):
    """
    Series decomposition block for multiple kernel sizes.
    """
    def __init__(self, kernel_sizes):
        super(SeriesDecomp9, self).__init__()
        if not isinstance(kernel_sizes, list):
            kernel_sizes = [kernel_sizes]  # Convert to list if single integer
        self.moving_avg_layers = nn.ModuleList([MovingAvg(kernel_size=ks, stride=1) for ks in kernel_sizes])

    def forward(self, x):
        # Apply each MovingAvg layer and average their outputs
        moving_means = [layer(x) for layer in self.moving_avg_layers]
        moving_mean = sum(moving_means) / len(moving_means)  # Combine outputs
        res = x - moving_mean
        return res, moving_mean


class BatchChannelNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, gamma_init=1.0, beta_init=0.0):
        super(BatchChannelNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Initialize gamma (scale) and beta (shift) with custom values
        self.gamma = nn.Parameter(torch.ones(num_features) * gamma_init)
        self.beta = nn.Parameter(torch.zeros(num_features) + beta_init)
        
        # Running mean and variance for cross-iteration normalization
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        # Ensure running_mean and running_var are on the same device as x
        if self.running_mean.device != x.device:
            self.running_mean = self.running_mean.to(x.device)
            self.running_var = self.running_var.to(x.device)

        # Calculate mean and variance across batch and sequence dimensions
        if self.training:
            mean = x.mean(dim=(0, 2), keepdim=True)
            var = x.var(dim=(0, 2), unbiased=False, keepdim=True)
            
            # Update running statistics for cross-iteration normalization
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
        else:
            mean = self.running_mean.view(1, -1, 1)
            var = self.running_var.view(1, -1, 1)

        # Normalize and apply scaling and shifting parameters
        x = (x - mean) / (torch.sqrt(var + self.eps))
        return self.gamma.view(1, -1, 1) * x + self.beta.view(1, -1, 1)


class CrossIterationChannelNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        super(CrossIterationChannelNorm, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.running_mean = nn.Parameter(torch.zeros(num_features), requires_grad=False)
        self.running_var = nn.Parameter(torch.ones(num_features), requires_grad=False)
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # x shape is (batch_size, num_features, sequence_length)
        batch_mean = x.mean(dim=(0, 2))  # Mean over batch and sequence length
        batch_var = x.var(dim=(0, 2), unbiased=False)  # Variance over batch and sequence length

        # Update running statistics for cross-iteration normalization
        if self.training:
            self.running_mean.data = self.momentum * batch_mean + (1 - self.momentum) * self.running_mean.data
            self.running_var.data = self.momentum * batch_var + (1 - self.momentum) * self.running_var.data

        # Use running stats during evaluation
        mean = self.running_mean if not self.training else batch_mean
        var = self.running_var if not self.training else batch_var

        # Normalize
        x = (x - mean.view(1, -1, 1)) / (torch.sqrt(var.view(1, -1, 1) + self.eps))
        return self.gamma.view(1, -1, 1) * x + self.beta.view(1, -1, 1)


class LD(nn.Module):
    def __init__(self, kernel_size=25):
        super(LD, self).__init__()
        # Define a shared convolution layers for all channels
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, stride=1, padding=int(kernel_size // 2),
                              padding_mode='replicate', bias=True)
        # Define the parameters for Gaussian initialization
        kernel_size_half = kernel_size // 2
        sigma = 1.0  # 1 for variance
        weights = torch.zeros(1, 1, kernel_size)
        for i in range(kernel_size):
            weights[0, 0, i] = math.exp(-((i - kernel_size_half) / (2 * sigma)) ** 2)

        # Set the weights of the convolution layer
        self.conv.weight.data = F.softmax(weights, dim=-1)
        self.conv.bias.data.fill_(0.0)

    def forward(self, inp):
        # Permute the input tensor to match the expected shape for 1D convolution (B, N, T)
        inp = inp.permute(0, 2, 1)
        # Split the input tensor into separate channels
        input_channels = torch.split(inp, 1, dim=1)

        # Apply convolution to each channel
        conv_outputs = [self.conv(input_channel) for input_channel in input_channels]

        # Concatenate the channel outputs
        out = torch.cat(conv_outputs, dim=1)
        out = out.permute(0, 2, 1)
        return out

        
class MultiConvLD(nn.Module):
    def __init__(self, kernel_sizes):
        super(MultiConvLD, self).__init__()
        # Define convolution layers
        self.conv1 = nn.Conv1d(1, 1, kernel_size=kernel_sizes[0], stride=1, 
                               padding=kernel_sizes[0] // 2, padding_mode='replicate', bias=True)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=kernel_sizes[1], stride=1, 
                               padding=kernel_sizes[1] // 2, padding_mode='replicate', bias=True)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=kernel_sizes[0], stride=1, 
                               padding=kernel_sizes[0] // 2, padding_mode='replicate', bias=True)
        self.conv4 = nn.Conv1d(1, 1, kernel_size=kernel_sizes[1], stride=1, 
                               padding=kernel_sizes[1] // 2, padding_mode='replicate', bias=True)
        # Initialize weights
        self.init_weights(self.conv1, kernel_sizes[0])
        self.init_weights(self.conv2, kernel_sizes[1])
        self.init_weights(self.conv3, kernel_sizes[0])
        self.init_weights(self.conv4, kernel_sizes[1])

    def init_weights(self, conv, kernel_size):
        kernel_size_half = kernel_size // 2
        sigma = 1.0
        weights = torch.zeros(1, 1, kernel_size)
        for i in range(kernel_size):
            weights[0, 0, i] = math.exp(-((i - kernel_size_half) / (2 * sigma)) ** 2)
        conv.weight.data = F.softmax(weights, dim=-1)
        conv.bias.data.fill_(0.0)

    
    
    def forward(self, inp):
        #print(f"Original inp shape: {inp.shape}")
        if inp.dim() == 4:
            # inp shape: (batch_size, seq_len, num_experts, output_dim)
            batch_size, seq_len, num_experts, output_dim = inp.size()
            # Initialize list to store outputs from each expert
            expert_outputs = []
            for expert_idx in range(num_experts):
                # Extract data for the current expert
                expert_inp = inp[:, :, expert_idx, :]  # Shape: (batch_size, seq_len, output_dim)
                # Permute to (batch_size, output_dim, seq_len)
                expert_inp = expert_inp.permute(0, 2, 1)
               # print(f"Expert {expert_idx} input shape after permute: {expert_inp.shape}")

                # Split the input into individual channels
                num_channels = expert_inp.size(1)
                input_channels = torch.split(expert_inp, 1, dim=1)  # Split along the channel dimension
                conv_outputs = []

                for idx, input_channel in enumerate(input_channels):
                    #print(f"Expert {expert_idx} - Processing input_channel[{idx}] with shape: {input_channel.shape}")
                    # Apply the first convolution
                    x = self.conv1(input_channel)
                    #print(f"Expert {expert_idx} - After conv1, x shape: {x.shape}")
                    # Apply the subsequent convolutions
                    x = self.conv2(x)
                    x = self.conv3(x)
                    x = self.conv4(x)
                    conv_outputs.append(x)

                # Concatenate outputs from all channels for the current expert
                expert_out = torch.cat(conv_outputs, dim=1)  # (batch_size, num_channels, seq_len)
                expert_out = expert_out.permute(0, 2, 1)  # Back to (batch_size, seq_len, num_channels)
               # print(f"Expert {expert_idx} output shape: {expert_out.shape}")

                expert_outputs.append(expert_out)

            # Stack outputs from all experts along a new dimension
            out = torch.stack(expert_outputs, dim=2)  # Shape: (batch_size, seq_len, num_experts, num_channels)
           # print(f"Final output shape: {out.shape}")
        elif inp.dim() == 3:
            # Existing code for 3D input
            # inp shape: (batch_size, seq_len, num_features)
            # Permute to (batch_size, num_features, seq_len)
            inp = inp.permute(0, 2, 1)
            #print(f"Input shape after permute: {inp.shape}")

            # Split the input into individual channels
            num_channels = inp.size(1)
            input_channels = torch.split(inp, 1, dim=1)  # Split along the channel dimension
            conv_outputs = []

            for idx, input_channel in enumerate(input_channels):
               # print(f"Processing input_channel[{idx}] with shape: {input_channel.shape}")
                # Apply the first convolution
                x = self.conv1(input_channel)
               # print(f"After conv1, x shape: {x.shape}")
                # Apply the subsequent convolutions
                x = self.conv2(x)
                x = self.conv3(x)
                x = self.conv4(x)
                conv_outputs.append(x)

            # Concatenate outputs from all channels
            out = torch.cat(conv_outputs, dim=1)  # (batch_size, num_channels, seq_len)
            out = out.permute(0, 2, 1)  # Back to (batch_size, seq_len, num_channels)
            #print(f"Output shape: {out.shape}")
        

        return out

        


        
class ParallelConvLD(nn.Module):
    def __init__(self, kernel_sizes=[25, 15]):
        super(ParallelConvLD, self).__init__()
        # Multiple convolution layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(1, 1, kernel_size=ks, stride=1, padding=int(ks // 2), padding_mode='replicate', bias=True)
            for ks in kernel_sizes
        ])

        # Initialize Gaussian-like weights for each layer
        for conv, ks in zip(self.convs, kernel_sizes):
            self.init_weights(conv, ks)

    def init_weights(self, conv, kernel_size):
        kernel_size_half = kernel_size // 2
        sigma = 1.0
        weights = torch.zeros(1, 1, kernel_size)
        for i in range(kernel_size):
            weights[0, 0, i] = math.exp(-((i - kernel_size_half) / (2 * sigma)) ** 2)
        conv.weight.data = F.softmax(weights, dim=-1)
        conv.bias.data.fill_(0.0)

    def forward(self, inp):
        inp = inp.permute(0, 2, 1)  # Change to (B, N, T)
        input_channels = torch.split(inp, 1, dim=1)  # Split into channels
        conv_outputs = []

        for input_channel in input_channels:
            # Apply all convolutions in parallel and sum the results
            x = sum(conv(input_channel) for conv in self.convs)
            conv_outputs.append(x)

        # Concatenate and restore shape
        out = torch.cat(conv_outputs, dim=1)
        out = out.permute(0, 2, 1)  # Back to (B, T, N)
        return out


    

    
# ---------------------------------------------------------------------------
# Block Attention Residuals (AttnRes) — MoonshotAI
# Based on: https://github.com/MoonshotAI/Attention-Residuals
# ---------------------------------------------------------------------------

class AttnResRMSNorm(nn.Module):
    """RMSNorm used for normalising AttnRes keys (supports any-rank input)."""
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.gamma


def block_attn_res(blocks, partial_block, proj, norm):
    """Inter-block attention: attend over completed block reps + partial sum.

    Args:
        blocks: list of N tensors of shape [B, T, D] — completed block reps.
        partial_block: [B, T, D] — intra-block partial sum for the current block.
        proj: nn.Linear(D, 1, bias=False) — learned pseudo-query per layer.
        norm: AttnResRMSNorm(D) — normalises keys before dot-product.
    Returns:
        h: [B, T, D] — attended representation.
    """
    V = torch.stack(blocks + [partial_block])      # [N+1, B, T, D]
    K = norm(V)                                     # [N+1, B, T, D]
    logits = torch.einsum('d, n b t d -> n b t', proj.weight.squeeze(), K)       # [N+1, B, T]
    h = torch.einsum('n b t, n b t d -> b t d', logits.softmax(dim=0), V)       # [B, T, D]
    return h


class AttnResTransformerLayer(nn.Module):
    """Single transformer layer with Block AttnRes instead of standard residuals.

    Following the paper pseudocode each sub-layer (attention + MLP) has its own
    ``attn_res_proj`` / ``attn_res_norm`` pair that replaces the plain ``+``
    residual connection.
    """
    def __init__(self, d_model, n_heads, d_ff, block_size, layer_number, dropout=0.1):
        super().__init__()
        self.layer_number = layer_number
        self.block_size = block_size

        # Pre-norm for attention sub-layer
        self.attn_norm = RMSNorm(d_model)
        self.attn = MHSelfAttention(dim=d_model, heads=n_heads, causal=False)

        # Pre-norm for MLP sub-layer
        self.mlp_norm = RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # AttnRes components for the attention sub-layer
        self.attn_res_proj = nn.Linear(d_model, 1, bias=False)
        self.attn_res_norm = AttnResRMSNorm(d_model)

        # AttnRes components for the MLP sub-layer
        self.mlp_res_proj = nn.Linear(d_model, 1, bias=False)
        self.mlp_res_norm = AttnResRMSNorm(d_model)

    def forward(self, blocks, hidden_states):
        """
        Args:
            blocks: list of completed block tensors [B, T, D].
            hidden_states: [B, T, D] — intra-block partial sum entering this layer.
        Returns:
            (blocks, partial_block): updated blocks list and new partial sum.
        """
        partial_block = hidden_states

        # --- Attention sub-layer ---
        h = block_attn_res(blocks, partial_block, self.attn_res_proj, self.attn_res_norm)

        # At every block boundary (every block_size//2 layers) snapshot the
        # current partial_block as a completed block representation.
        if self.layer_number % (self.block_size // 2) == 0:
            blocks = blocks + [partial_block]   # non-destructive append
            partial_block = None

        attn_out = self.attn(self.attn_norm(h))
        partial_block = attn_out if partial_block is None else partial_block + attn_out

        # --- MLP sub-layer ---
        h = block_attn_res(blocks, partial_block, self.mlp_res_proj, self.mlp_res_norm)
        mlp_out = self.mlp(self.mlp_norm(h))
        partial_block = partial_block + mlp_out

        return blocks, partial_block


class AttnResTransformer(nn.Module):
    """Stack of AttnResTransformerLayer with shared Block AttnRes state."""
    def __init__(self, d_model, n_heads, d_ff, n_layers, block_size, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            AttnResTransformerLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                block_size=block_size,
                layer_number=i,
                dropout=dropout,
            )
            for i in range(n_layers)
        ])

    def forward(self, x):
        """
        Args:
            x: [B, T, D]
        Returns:
            [B, T, D]
        """
        blocks = []
        hidden_states = x
        for layer in self.layers:
            blocks, hidden_states = layer(blocks, hidden_states)
        return hidden_states


class model_c(torch.nn.Module):
    """Time-series forecasting model using Block Attention Residuals (AttnRes).

    Keeps the same decomposition + linear projection front-end as the original
    model_c, but replaces the minGRU-based backbone with an AttnRes Transformer.

    Input:  x [Batch, context_points, Channel]
    Output:   [Batch, target_points,  Channel]
    """
    def __init__(self, configs, enc_in):
        super(model_c, self).__init__()
        self.configs = configs
        self.enc_in = enc_in
        self.batch_norm = BatchChannelNorm(enc_in, gamma_init=1.0, beta_init=0.0)

        n_attn_res_layers = 4
        d_model = getattr(configs, 'n2', 256)   # hidden dim (default 256)
        n_heads = 8
        block_size = 4
        d_ff = d_model * 4
        dropout = 0.1

        # Decomposition Kernel Size
        kernel_size = [25, 25]
        self.decomposition = SeriesDecomp9(kernel_size)
        self.LD = MultiConvLD(kernel_size)

        # Linear layers for seasonal and trend components
        self.Linear_Seasonal = nn.Linear(configs.context_points, configs.target_points)
        self.Linear_Trend = nn.Linear(configs.context_points, configs.target_points)
        self.Linear_Seasonal.weight = nn.Parameter(
            (1 / configs.context_points) * torch.ones([configs.target_points, configs.context_points])
        )
        self.Linear_Trend.weight = nn.Parameter(
            (1 / configs.context_points) * torch.ones([configs.target_points, configs.context_points])
        )

        # Project combined seasonal+trend to d_model
        self.mm = nn.Linear(configs.target_points, d_model)

        # Block AttnRes Transformer backbone
        self.attn_res_transformer = AttnResTransformer(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_attn_res_layers,
            block_size=block_size,
            dropout=dropout,
        )

        # Project back to forecast horizon
        self.mm2 = nn.Linear(d_model, configs.target_points)

    def forward(self, x):
        # Series decomposition
        seasonal_init, trend_init = self.decomposition(x)
        trend_refined = self.LD(trend_init)

        # Permute to [B, C, T] for linear projection over time
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_refined = trend_refined.permute(0, 2, 1)

        seasonal_output = self.Linear_Seasonal(seasonal_init)   # [B, C, target_points]
        trend_output = self.Linear_Trend(trend_refined)         # [B, C, target_points]

        x = seasonal_output + trend_output                      # [B, C, target_points]

        x = self.mm(x)                                          # [B, C, d_model]
        x = self.batch_norm(x)                                  # [B, C, d_model]
        x = self.attn_res_transformer(x)                        # [B, C, d_model]
        x = self.mm2(x)                                         # [B, C, target_points]
        x = x.permute(0, 2, 1)                                  # [B, target_points, C]

        return x



#Define Model B
class model_b(nn.Module):
    def __init__(self, configs, enc_in):
        super(model_b, self).__init__()
        self.configs = configs
        self.enc_in = enc_in
        self.batch_norm = nn.BatchNorm1d(self.enc_in)
        self.revin_layer = RevIN(self.enc_in)

        # self.seasonal_encoder = SeasonalEncoder(configs)
        # self.trend_encoder = TrendEncoder(configs)

        kernel_size = 25
        self.decompsition = series_decomp2(kernel_size)
        self.Linear_Seasonal = nn.Linear(configs.context_points, configs.target_points)
        self.Linear_Trend = nn.Linear(configs.context_points, configs.target_points)
        self.Linear_Seasonal.weight = nn.Parameter((1/configs.context_points)*torch.ones([configs.target_points, configs.context_points]))
        self.Linear_Trend.weight = nn.Parameter((1/configs.context_points)*torch.ones([configs.target_points, configs.context_points]))
    
        self.mm = nn.Linear(self.configs.target_points, self.configs.n2)
        self.mm2 = nn.Linear(config.embedding_dim, configs.target_points)
        
        self.mm3 = nn.Linear(configs.context_points, self.configs.n1)
        self.xlstm_stack = xLSTMBlockStack(config)

    def forward(self, x):
        batch_size = x.size(0)
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        x = self.mm(x)
        x = self.batch_norm(x)
        x = self.xlstm_stack(x)
        x = self.mm2(x)
        x = x.permute(0, 2, 1)
        return x
    
class Model1(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, pred_len, seq_len, enc_in):
        super(Model1, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        kernel_size = 25
        self.decompsition = series_decomp2(kernel_size)
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len, self.seq_len]))
        self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len, self.seq_len]))

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        
        self.channels = enc_in
        self.batch_norm = nn.BatchNorm1d(self.channels)
        
        
        self.individual =False
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.pred_len, self.pred_len)
                               
            
    def forward(self, x):
            seasonal_init, trend_init = self.decompsition(x)
            seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
            #x = seasonal_output + trend_output
        

            if self.individual:
                output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
                for i in range(self.channels):
                    output[:,:,i] = self.Linear[i](x[:,:,i])
                x = output
            else:
                x=self.Linear(trend_output).permute(0, 2, 1)
                x=self.Linear(seasonal_output).permute(0, 2, 1)
                #x=F.gelu(self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1))
            # print(x.shape)
            x = seasonal_output + trend_output
            #x=x.permute(0,2,1)
            x=self.batch_norm(x)
            
            x=x.permute(0,2,1)
            return x



class Model2(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, pred_len,seq_len,enc_in):
        super(Model2, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.channels = enc_in
        self.batch_norm = nn.BatchNorm1d(self.channels)
        
        self.individual=False
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                #self.batch_norm(self.Linear.append(nn.Linear(self.seq_len,self.pred_len)))
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)
        
    def forward(self, x):
        
            # x: [Batch, Input length, Channel]
            seq_last = x[:,-1:,:].detach()

            x = x - seq_last
            if self.individual:
                output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
                for i in range(self.channels):
                    output[:,:,i] = self.Linear[i](x[:,:,i])
                x = output
            else:
                x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
            x = x + seq_last

            
            x=x.permute(0,2,1)
            x=self.batch_norm(x)
            x =x.permute(0,2,1)
        
            return x # [Batch, Output length, Channel]

class model_d(nn.Module):
    def __init__(self, configs, enc_in):
        super(model_d, self).__init__()
        self.seq_len = configs.context_points
        self.pred_len = configs.target_points
        self.channels = enc_in
        
        self.batch_norm = nn.BatchNorm1d(self.channels)
        
        self.model2 = Model2(self.pred_len, self.seq_len, enc_in)
        self.model = Model1(self.pred_len, self.seq_len, enc_in)
        
        # Initialize a learnable gate parameter. 
        # Start from a neutral value (e.g. 0.5).
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # Get outputs from both models
        output_model2 = self.model2(x)  # shape: [B, L, C]
        output_model = self.model(x)    # shape: [B, L, C]
        
        # Apply sigmoid to ensure the gate value stays between 0 and 1
        g = torch.sigmoid(self.gate)
        
        # Weighted combination:
        # If g is close to 1, the output will be closer to output_model2;
        # if g is close to 0, the output will be closer to output_model.
        x = g * output_model2 + (1 - g) * output_model
        
        # Apply batch normalization
        x = x.permute(0, 2, 1)
        x = self.batch_norm(x)
        x = x.permute(0, 2, 1)
        
        return x




class GatingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_dim, num_experts)
    
    def forward(self, x):
        #print(x.shape)

        #x = x.permute(0, 2,3,1)
        x = self.relu(self.fc1(x))
       # print(x.shape)
        x = self.fc2(x)
        #print(x.shape)
        return x#.softmax(dim=-1)# F.softmax(gating_weights, dim=2)

    

class MovingAvg2(nn.Module):
    """
    Moving average block to highlight the trend of time series.
    """
    def __init__(self, kernel_size, stride=1):
        super(MovingAvg2, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)

    def forward(self, x):
        # x shape: (batch_size, num_features, num_experts, seq_len)
        batch_size, num_features, num_experts, seq_len = x.size()
        # Reshape x to merge num_features and num_experts
        x = x.view(batch_size * num_features * num_experts, seq_len)
        x = x.unsqueeze(1)  # Shape: (batch_size * num_features * num_experts, 1, seq_len)
        # Apply average pooling
        x = self.avg(x)
        x = x.squeeze(1)  # Shape: (batch_size * num_features * num_experts, seq_len)
        # Reshape back to original dimensions
        x = x.view(batch_size, num_features, num_experts, seq_len)
        return x
 
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.gamma
  

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_seq_length=512, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_length, embedding_dim)
        for pos in range(max_seq_length):
            for i in range(0, embedding_dim, 2):
                pe[pos, i] = math.sin(pos/(10000**(2*i/embedding_dim)))
                if i+1 < embedding_dim:
                    pe[pos, i+1] = math.cos(pos/(10000**((2*i+1)/embedding_dim)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is expected to be [B, Seq, Dim]
        x = x * math.sqrt(self.embedding_dim)
        seq_length = x.size(1)
        pe = self.pe[:, :seq_length].to(x.device)  # [1, Seq, Dim]
        x = x + pe
        x = self.dropout(x)
        return x



class MHSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None, causal=True):
        super().__init__()
        self.dim_head = (dim // heads) if dim_head is None else dim_head
        _dim = self.dim_head * heads
        self.heads = heads
        self.causal = causal
        self.to_qkv = nn.Linear(dim, _dim * 3, bias=False)
        self.W_out = nn.Linear(_dim, dim, bias=False)
        self.scale_factor = self.dim_head ** -0.5

    def set_causal(self, causal):
        self.causal = causal

    def forward(self, x, mask=None):
        # x: [B, N, D]
        qkv = self.to_qkv(x)  # [B, N, 3*heads*dim_head]
        q, k, v = tuple(rearrange(qkv, 'b n (d k h) -> k b h n d', k=3, h=self.heads))
        # q, k, v are now [B, heads, N, dim_head]

        # Compute attention
        scaled_dot_prod = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale_factor

        i, j = scaled_dot_prod.shape[2], scaled_dot_prod.shape[3]
        if self.causal:
            causal_mask = torch.ones(i, j, device=x.device).triu_(j - i + 1).bool()
            scaled_dot_prod = scaled_dot_prod.masked_fill(causal_mask, float('-inf'))

        # if mask is not None:
        #     assert mask.shape == scaled_dot_prod.shape[2:]
        #     scaled_dot_prod = scaled_dot_prod.masked_fill(mask, float('-inf'))

        attention = torch.softmax(scaled_dot_prod, dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attention, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.W_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None, causal=False, dim_linear_block=1024, dropout=0.1):
        super().__init__()

        
        self.mhsa = MHSelfAttention(dim=dim, heads=heads, dim_head=dim_head, causal=causal)
        self.drop = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(96)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.norm_3= RMSNorm(dim)
        self.norm_4= RMSNorm(dim)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.linear = nn.Sequential(
            nn.Linear(dim, dim_linear_block),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_linear_block, dim),
            nn.Dropout(dropout)
        )

    def set_causal(self, causal):
        self.mhsa.set_causal(causal)

    def forward(self, x, mask=None):
        #print(x.shape)
        #y = self.norm_1(x + self.drop(self.mhsa(x, mask)))
        #x=x.permute(0,2,1)
        y= self.norm_3 (x + self.drop(self.mhsa(x, mask)))
        #y=y.permute(02,1)
        return  self.norm_4(y + self.linear(y))#self.batch_norm2(y + self.linear(y)).permute(0,2,1)


class SimpleTransformer(nn.Module):
    def __init__(self, dim, enc_in ,num_experts ,num_layers=6, heads=8, dim_head=None, max_seq_len=1024, causal=True):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.causal = causal

        self.embed_dim = enc_in * num_experts#35 # from your code
        self.num_heads = 4
        self.head_dim = self.embed_dim // self.num_heads
        self.linear = nn.Linear(self.embed_dim, 128)
        self.linear2 = nn.Linear(128, self.embed_dim)
        
        # Create a RotaryEmbedding instance
        # Note: We pass 'dim=self.embed_dim' so that rotary dimensions match the dimension of x before PositionalEncoding.

        self.pos_enc = PositionalEncoding(dim, max_seq_length=max_seq_len)

        self.block_list = [
            TransformerBlock(dim=dim, heads=heads, dim_head=dim_head, causal=causal)
            for _ in range(num_layers)
        ]
        self.layers = nn.ModuleList(self.block_list)

    def set_causal(self, causal):
        for b in self.block_list:
            b.set_causal(causal)

    def forward(self, x, mask=None):
        # x shape: [B, Feat, Exp, S]
        print(x.shape) #torch.Size([64, 7, 5, 96])
        B, Feat, Exp, S = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, S, Feat * Exp)  # [B, S, D]
        x = self.linear(x)  # Now [B, S, 128]

        # # Apply rotary embeddings before positional encoding
        # # Rotary expects shape [B, Seq, H, D_head]. We'll use H=1 for simplicity here.
        # x = rearrange(x, 'b s d -> b s 1 d')

        # # Apply rotary embedding
        # # rotate_queries_or_keys applies rotary embeddings along the seq dimension
        # x = self.rotary_emb.rotate_queries_or_keys(x, seq_dim=1)

        # # Rearrange back to [B, S, D]
        # x = rearrange(x, 'b s 1 d -> b s d')

        # Now apply positional encoding
        x =self.pos_enc(x)

        # Pass through Transformer blocks
        for layer in self.layers:
            x = layer(x, mask)
            #print(x.shape)

        x = self.linear2(x)
        x = x.reshape(B, Feat, Exp, S)
        #print(x.shape)# torch.Size([64, 7, 5, 96])

        return x

class EMTSF(nn.Module):
    def __init__(self, configs, enc_in, num_experts_a=2, num_experts_b=2, num_experts_c=2, num_experts_d=2, freeze_experts=True):
        super(EMTSF, self).__init__()
        self.configs = configs
        self.input_dim = configs.target_points
        self.output_dim = configs.target_points
        self.enc_in = enc_in
        self.hidden_dim = 128
        self.batch_norm = nn.BatchNorm2d(self.enc_in)
        self.batch_norm2 = nn.BatchNorm1d(self.enc_in)
        self.freeze_experts = freeze_experts

        # Create experts list (assuming Model, model_b, model_c, model_d, and GatingNetwork are defined)
        self.experts = nn.ModuleList()

         # Load and add instances of model_a
        for _ in range(num_experts_a):
            expert_a = Model(configs, enc_in)
            model_a_path = os.path.join(configs.save_path, configs.save_model_name + 'model_a.pth')
            if not os.path.isfile(model_a_path):
                raise FileNotFoundError(f"Model file not found: {model_a_path}")
            expert_a.load_state_dict(torch.load(model_a_path))
            if self.freeze_experts:
                for param in expert_a.parameters():
                    param.requires_grad = False
            self.experts.append(expert_a)

        # Load and add instances of model_b
        for _ in range(num_experts_b):
            expert_b = model_b(configs, enc_in)
            model_b_path = os.path.join(configs.save_path, configs.save_model_name + 'model_b.pth')
            if not os.path.isfile(model_b_path):
                raise FileNotFoundError(f"Model file not found: {model_b_path}")
            expert_b.load_state_dict(torch.load(model_b_path))
            if self.freeze_experts:
                for param in expert_b.parameters():
                    param.requires_grad = False
            self.experts.append(expert_b)

        # Load and add instances of model_c
        for _ in range(num_experts_c):
            expert_c = model_c(configs, enc_in)
            model_c_path = os.path.join(configs.save_path, configs.save_model_name + 'model_c.pth')
            if not os.path.isfile(model_c_path):
                raise FileNotFoundError(f"Model file not found: {model_c_path}")
            expert_c.load_state_dict(torch.load(model_c_path))
            if self.freeze_experts:
                for param in expert_c.parameters():
                    param.requires_grad = False
            self.experts.append(expert_c)

        for _ in range(num_experts_d):
             expert_d = model_d(configs, enc_in)
             model_d_path = os.path.join(configs.save_path, configs.save_model_name + 'model_d.pth')#("saved_models/etth2/target_192/model_etth2_epochs100_context512_targetmodel_d.pth")#(configs.save_path, configs.save_model_name + 'model_d.pth')#"saved_models/etth1/target_96/model_etth1_epochs100_context512_target96model_d.pth")#configs.save_path, configs.save_model_name + 'model_d.pth')
             if not os.path.isfile(model_d_path):
                 raise FileNotFoundError(f"Model file not found: {model_d_path}")
             expert_d.load_state_dict(torch.load(model_d_path))
             if self.freeze_experts:
                 for param in expert_d.parameters():
                     param.requires_grad = False
             self.experts.append(expert_d)



        self.num_experts = num_experts_a + num_experts_b + num_experts_c + num_experts_d
        self.gating_network = GatingNetwork(self.output_dim, self.hidden_dim, self.output_dim)
        
        for param in self.gating_network.parameters():
            param.requires_grad = True

        self.embed_dim =128#self.enc_in * self.num_experts  # F * E
        self.num_heads =self.num_experts
        self.head_dim = self.embed_dim // self.num_heads
        self.linear= nn.Linear(self.embed_dim ,128)
        self.linear2= nn.Linear(128,self.embed_dim )

        self.attention_layer = SimpleTransformer(
            dim=self.embed_dim,enc_in=self.enc_in ,num_experts=self.num_experts,
            num_layers=3,
            heads=self.num_heads,
            dim_head=None,
            max_seq_len=self.configs.target_points,
            causal=True
        )
        


    
    def forward(self, x):
       #print(x.shape)
        # expert_outputs: (B, S, F)
        expert_outputs = [expert(x).permute(0, 2, 1) for expert in self.experts] # now (B, F, S)
        # stacked: (B, F, E, S)
        expert_outputs_stacked = torch.stack(expert_outputs, dim=2)

        # gating_weights = self.gating_network(expert_outputs_stacked)  # (B, F, E, S)
        # gating_weights = self.batch_norm(gating_weights)  # (B, F, E, S)

        # B, Feat, Exp, S = gating_weights.shape
        # x = gating_weights.permute(0, 3, 1, 2).reshape(B, S, Feat*Exp)  # [B, S, D]
        # x=self.linear(x)


        attn_output = self.attention_layer(expert_outputs_stacked)  # [B, S, D]
        # attn_output=self.linear2(attn_output)
        # #print(attn_output.shape)

        # attn_output = attn_output.reshape(B, Feat, Exp, S)

        # Softmax over the experts dimension
        gating_weights = F.softmax(attn_output, dim=2)  # (B, Feat, Exp, Seq)

        weighted_outputs = []
        for idx, expert_output in enumerate(expert_outputs):
            # expert_output: (B, F, S)
            gw = gating_weights[:, :, idx, :]  # (B, Feat, Seq)
            weighted_outputs.append(expert_output * gw)

        output = sum(weighted_outputs)  # (B, Feat, Seq)
        x =output.permute(0, 2, 1)
        #print(x.shape)
        return  x # (B, Seq, Feat)

    



