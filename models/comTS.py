import torch
import torch.nn as nn

from models import comKAN, KAN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self._init_config(configs)
        self.freq_kan_channel = self._init_kan_layer(comKAN.KAN, [self.seq_length // 2 + 1, self.hidden_size,
                                                                  (self.seq_length + self.pre_length) // 2 + 1], configs)
        self.time_kan_channel = self._init_kan_layer(KAN.KAN, [self.seq_length, self.hidden_size, self.seq_length + self.pre_length],
                                                     configs)
        # self.freq_fc = nn.Linear(self.seq_length, self.seq_length + self.pre_length)
        # self.time_fc = nn.Linear(self.seq_length, self.seq_length + self.pre_length)

    def _init_config(self, configs):
        self.embed_size = configs.embed_size
        self.hidden_size = configs.hidden_size
        self.pre_length = configs.pred_len
        self.feature_size = configs.enc_in
        self.seq_length = configs.seq_len
        self.channel_independence = configs.channel_independence
        self.base_activation = {"silu": nn.SiLU}[configs.base_activation]

    def _init_kan_layer(self, kan_class, layers_hidden, configs):
        return kan_class(
            layers_hidden=layers_hidden,
            grid_size=configs.grid_size,
            spline_order=configs.spline_order,
            scale_noise=configs.scale_noise,
            scale_base=configs.scale_base,
            scale_spline=configs.scale_spline,
            base_activation=self.base_activation,
            grid_eps=configs.grid_eps,
            grid_range=[-1, 1],
            regularize_activation=configs.regularize_activation,
            regularize_entropy=configs.regularize_entropy,
            update_grid=configs.update_grid,
        )

    # frequency channel learner
    def freqKAN_channel(self, x):
        # [B, N, T]
        x = torch.fft.rfft(x, dim=2, norm='ortho') # FFT on T dimension
        y = self.freq_kan_channel(x) # [B, N, T // 2 + 1]
        x = torch.fft.irfft(y, n=self.seq_length + self.pre_length, dim=2, norm="ortho") # [B, N, T+Pre]
        return x

    # time channel learner
    def timeKAN_channel(self, x):
        # [B, N, T]
        x = self.time_kan_channel(x)  # [B, N, T+Pre]
        return x

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        # B, T, N = x.shape
        x = x.permute(0, 2, 1) # [B, N, T]

        # [B, N, T+Pre]
        freq_x = self.freqKAN_channel(x).permute(0, 2, 1) # [B, T+Pre, N]
        time_x = self.timeKAN_channel(x).permute(0, 2, 1) # [B, T+Pre, N]

        return time_x, freq_x

class SecondStage(Model):
    def __init__(self, configs):
        super(SecondStage, self).__init__(configs)
        self._init_config(configs)
        # self.freq_fc = nn.Linear(self.seq_length, self.seq_length + self.new_pre_length)
        # self.time_fc = nn.Linear(self.seq_length, self.seq_length + self.new_pre_length)
        self.freq_kan_channel = self._init_kan_layer(comKAN.KAN, [self.seq_length // 2 + 1, self.hidden_size,
                                                                  (self.seq_length + self.new_pre_length) // 2 + 1],
                                                     configs)
        self.time_kan_channel = self._init_kan_layer(KAN.KAN, [self.seq_length, self.hidden_size,
                                                               self.seq_length + self.new_pre_length],
                                                     configs)

    def _init_config(self, configs):
        self.new_pre_length = configs.new_pred_len

        self.embed_size = configs.embed_size
        self.hidden_size = configs.hidden_size
        self.pre_length = configs.pred_len
        self.feature_size = configs.enc_in
        self.seq_length = configs.seq_len
        self.channel_independence = configs.channel_independence
        self.base_activation = {"silu": nn.SiLU}[configs.base_activation]

        # frequency channel learner

    def freqKAN_channel(self, x):
        # [B, N, T]
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on T dimension
        y = self.freq_kan_channel(x)  # [B, N, T // 2 + 1]
        x = torch.fft.irfft(y, n=self.seq_length + self.new_pre_length, dim=2, norm="ortho")  # [B, N, T+Pre]
        return x

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)  # [B, N, T]

        # [B, N, T+Pre]
        freq_x = self.freqKAN_channel(x).permute(0, 2, 1)  # [B, T+Pre, N]
        time_x = self.timeKAN_channel(x).permute(0, 2, 1)  # [B, T+Pre, N]

        return time_x, freq_x