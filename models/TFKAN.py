import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import comKAN, KAN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        # self.embed_size = 128 #embed_size
        self.embed_size = configs.embed_size
        self.hidden_size = 256 #hidden_size
        self.pre_length = configs.pred_len
        self.feature_size = configs.enc_in #channels
        self.seq_length = configs.seq_len
        self.channel_independence = configs.channel_independence

        self.use_bias = configs.use_bias

        self.sparsity_threshold = 0.001
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))

        self.KAN_freq = self._init_kan_layer(KAN.KAN, [self.embed_size, self.hidden_size,
                                                                  self.embed_size])

        self.KAN_time = self._init_kan_layer(KAN.KAN, [self.seq_length, self.hidden_size,
                                                  self.seq_length])

        self.fc = self._init_kan_layer(KAN.KAN, [self.seq_length * self.embed_size, self.hidden_size, self.pre_length])

    def _init_kan_layer(self, kan_class, layers_hidden):
        return kan_class(
            layers_hidden=layers_hidden,
            grid_size=2,
            spline_order=1,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
            regularize_activation=1.0,
            regularize_entropy=1.0,
            update_grid=False,
        )

    # dimension extension
    def tokenEmb(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        # N*T*1 x 1*D = N*T*D
        y = self.embeddings
        return x * y

    # frequency learner
    def KAN_frequency(self, x, B, N, L):
        # [B, N, L, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho') # FFT on L dimension
        
        y_real = self.KAN_freq(x.real)
        y_imag = self.KAN_freq(x.imag)
        y = torch.stack([y_real, y_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        
        x = torch.fft.irfft(y, n=self.seq_length, dim=2, norm="ortho")
        return x

    # temporal learner
    def KAN_temporal(self, x, B, N, L):
        # [B, N, L]
        y = self.KAN_time(x)
        return y

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        B, L, N = x.shape
        # embedding x: [B, N, L, D]
        x_noD = x.permute(0, 2, 1)
        x = self.tokenEmb(x)
        bias = x + x_noD.unsqueeze(-1)
        # [B, N, L, D]
        x = self. KAN_frequency(x, B, N, L)
        x_noD = self. KAN_temporal(x_noD, B, N, L)
        
        x = x + x_noD.unsqueeze(-1)
        if self.use_bias:
            x = x + bias
        else:
            x = x
        x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)
        return x
