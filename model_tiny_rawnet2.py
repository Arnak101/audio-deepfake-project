import math
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SincConvFast(nn.Module):

    @staticmethod
    def to_mel(hz: np.ndarray) -> np.ndarray:
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel: np.ndarray) -> np.ndarray:
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(
        self,
        out_channels: int,
        kernel_size: int = 129,
        sample_rate: int = 16000,
        stride: int = 4,
        freq_scale: str = "mel",
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            kernel_size += 1

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.stride = stride
        self.freq_scale = freq_scale.lower()

        nfft = 512
        freqs = int(sample_rate / 2) * np.linspace(0, 1, int(nfft / 2) + 1)

        if self.freq_scale == "mel":
            mel = self.to_mel(freqs)
            mel_points = np.linspace(mel.min(), mel.max(), out_channels + 1)
            hz_points = self.to_hz(mel_points)
            band_edges = hz_points
        elif self.freq_scale == "inverse-mel":
            mel = self.to_mel(freqs)
            mel_points = np.linspace(mel.min(), mel.max(), out_channels + 1)
            hz_points = self.to_hz(mel_points)
            band_edges = np.flip(hz_points)
        else:
            band_edges = np.linspace(freqs.min(), freqs.max(), out_channels + 1)

        band_edges[0] = max(30.0, band_edges[0] + 30.0)
        band_edges[-1] = min(sample_rate / 2 - 100.0, band_edges[-1] - 100.0)

        self.register_buffer("band_edges", torch.tensor(band_edges, dtype=torch.float32))
        hsupp = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
        self.register_buffer("hsupp", hsupp)
        self.register_buffer("window", torch.hamming_window(kernel_size, periodic=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, T]
        returns: [B, C, T']
        """
        device = x.device
        band_edges = self.band_edges.to(device)
        hsupp = self.hsupp.to(device)
        window = self.window.to(device)

        filters = []
        for i in range(self.out_channels):
            fmin = band_edges[i]
            fmax = band_edges[i + 1]

            h_high = (2 * fmax / self.sample_rate) * torch.sinc(2 * fmax * hsupp / self.sample_rate)
            h_low = (2 * fmin / self.sample_rate) * torch.sinc(2 * fmin * hsupp / self.sample_rate)
            hideal = h_high - h_low
            filt = hideal * window
            filters.append(filt)

        filters = torch.stack(filters, dim=0).unsqueeze(1)  # [C,1,K]
        return F.conv1d(x, filters, stride=self.stride, padding=0, bias=None)


class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, first: bool = False):
        super().__init__()
        self.first = first

        self.bn1 = nn.BatchNorm1d(in_ch)
        self.act = nn.LeakyReLU(0.3, inplace=True)

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)

        self.downsample = None
        if in_ch != out_ch:
            self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)

        self.pool = nn.MaxPool1d(kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = x
        if not self.first:
            out = self.bn1(out)
            out = self.act(out)

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.pool(out)
        return out


class ChannelAttention1D(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,T]
        y = self.pool(x).squeeze(-1)         # [B,C]
        y = self.fc(y)
        y = self.sigmoid(y).unsqueeze(-1)    # [B,C,1]
        return x * y + y


class TinyRawNet2(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        input_samples: int = 16000,   # 1 second
        sinc_out: int = 8,
        sinc_kernel: int = 129,
        sinc_stride: int = 4,
        channels_stage1: int = 8,
        channels_stage2: int = 12,
        gru_hidden: int = 8,
        fc_hidden: int = 12,
        num_classes: int = 2,
        freq_scale: str = "mel",
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.input_samples = input_samples

        self.sinc = SincConvFast(
            out_channels=sinc_out,
            kernel_size=sinc_kernel,
            sample_rate=sample_rate,
            stride=sinc_stride,
            freq_scale=freq_scale,
        )

        self.first_bn = nn.BatchNorm1d(sinc_out)
        self.first_act = nn.SELU(inplace=True)

        # 4 residual blocks total instead of original 6
        self.block0 = ResidualBlock1D(sinc_out, channels_stage1, first=True)
        self.att0 = ChannelAttention1D(channels_stage1)

        self.block1 = ResidualBlock1D(channels_stage1, channels_stage1, first=False)
        self.att1 = ChannelAttention1D(channels_stage1)

        self.block2 = ResidualBlock1D(channels_stage1, channels_stage2, first=False)
        self.att2 = ChannelAttention1D(channels_stage2)

        self.block3 = ResidualBlock1D(channels_stage2, channels_stage2, first=False)
        self.att3 = ChannelAttention1D(channels_stage2)

        self.bn_before_gru = nn.BatchNorm1d(channels_stage2)
        self.pre_gru_act = nn.SELU(inplace=True)

        self.gru = nn.GRU(
            input_size=channels_stage2,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )

        self.fc1 = nn.Linear(gru_hidden, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T] or [B,1,T]
        returns logits [B, num_classes]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B,1,T]
        elif x.dim() != 3:
            raise ValueError("Input must have shape [B,T] or [B,1,T].")

        x = self.sinc(x)
        x = F.max_pool1d(torch.abs(x), kernel_size=3)
        x = self.first_bn(x)
        x = self.first_act(x)

        x = self.block0(x)
        x = self.att0(x)

        x = self.block1(x)
        x = self.att1(x)

        x = self.block2(x)
        x = self.att2(x)

        x = self.block3(x)
        x = self.att3(x)

        x = self.bn_before_gru(x)
        x = self.pre_gru_act(x)

        x = x.permute(0, 2, 1)  # [B,C,T] -> [B,T,C]
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:, -1, :]         # last frame

        x = self.fc1(x)
        x = self.fc2(x)
        return x


def random_or_center_crop(
    wav: torch.Tensor,
    target_len: int = 16000,
    train: bool = True,
) -> torch.Tensor:
    """
    wav: [T]
    """
    length = wav.shape[0]

    if length == target_len:
        return wav

    if length > target_len:
        if train:
            start = random.randint(0, length - target_len)
        else:
            start = (length - target_len) // 2
        return wav[start:start + target_len]

    # if too short, repeat
    repeats = math.ceil(target_len / length)
    wav = wav.repeat(repeats)[:target_len]
    return wav


if __name__ == "__main__":
    model = TinyRawNet2()
    x = torch.randn(4, 16000)  # batch of 4 one-second clips
    y = model(x)
    print("logits shape:", y.shape)  # [4,2]