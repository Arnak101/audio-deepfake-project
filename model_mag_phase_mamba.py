import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        self.pool = nn.MaxPool2d(kernel_size=2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        x = self.pool(x)
        return x


class MagnitudeCNNEncoder(nn.Module):
    """
    Input: log-magnitude spectrogram [B, 1, F, T]
    Output: feature vector [B, D]
    """
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock2D(1, 32, pool=True),
            ConvBlock2D(32, 64, pool=True),
            ConvBlock2D(64, 128, pool=True),
            ConvBlock2D(128, 128, pool=False),
        )
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.proj(x)
        return x


class SimpleMambaBlock(nn.Module):
    """
    Not official Mamba package.
    This is a lightweight Mamba-like sequential block:
    projection -> depthwise conv over time -> gating -> residual.
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, 2 * dim)
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=5,
            padding=2,
            groups=dim,
            bias=True,
        )
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        residual = x
        x = self.norm(x)

        x_proj = self.in_proj(x)               # [B, T, 2D]
        x_main, x_gate = torch.chunk(x_proj, 2, dim=-1)

        x_main = x_main.transpose(1, 2)        # [B, D, T]
        x_main = self.dwconv(x_main)
        x_main = x_main.transpose(1, 2)        # [B, T, D]

        x_main = F.gelu(x_main)
        x_gate = torch.sigmoid(x_gate)

        x = x_main * x_gate
        x = self.out_proj(x)
        x = self.dropout(x)

        return residual + x


class PhaseMambaEncoder(nn.Module):
    """
    Input: phase representation [B, 2, F, T]
           channels are cos(phase), sin(phase)
    Output: feature vector [B, D]
    """
    def __init__(
        self,
        freq_bins: int,
        model_dim: int = 192,
        num_blocks: int = 4,
        out_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.freq_bins = freq_bins
        self.input_dim = 2 * freq_bins

        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.Sequential(
            *[SimpleMambaBlock(model_dim, dropout=dropout) for _ in range(num_blocks)]
        )

        self.out_proj = nn.Sequential(
            nn.Linear(model_dim, out_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2, F, T]
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()   # [B, T, 2, F]
        x = x.view(b, t, c * f)                  # [B, T, 2F]

        x = self.input_proj(x)                   # [B, T, D]
        x = self.blocks(x)                       # [B, T, D]

        x = x.mean(dim=1)                        # temporal average pooling
        x = self.out_proj(x)                     # [B, out_dim]
        return x


class MagPhaseMambaClassifier(nn.Module):
    """
    waveform -> STFT -> magnitude + phase
             -> magnitude CNN branch + phase Mamba branch
             -> concat -> final linear classifier
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        mag_out_dim: int = 256,
        phase_model_dim: int = 192,
        phase_blocks: int = 4,
        phase_out_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.freq_bins = n_fft // 2 + 1

        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

        self.mag_encoder = MagnitudeCNNEncoder(out_dim=mag_out_dim)

        self.phase_encoder = PhaseMambaEncoder(
            freq_bins=self.freq_bins,
            model_dim=phase_model_dim,
            num_blocks=phase_blocks,
            out_dim=phase_out_dim,
            dropout=dropout,
        )

        fusion_dim = mag_out_dim + phase_out_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 1),
        )

    def compute_stft_features(self, x: torch.Tensor):
        """
        x: [B, T]
        returns:
            log_mag: [B, 1, F, TT]
            phase_repr: [B, 2, F, TT]
        """
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )  # [B, F, TT]

        mag = torch.abs(stft)                    # [B, F, TT]
        log_mag = torch.log1p(mag).unsqueeze(1) # [B, 1, F, TT]

        phase = torch.angle(stft)               # [B, F, TT]
        phase_cos = torch.cos(phase)
        phase_sin = torch.sin(phase)
        phase_repr = torch.stack([phase_cos, phase_sin], dim=1)  # [B, 2, F, TT]

        return log_mag, phase_repr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T] or [B, 1, T]
        if x.dim() == 3:
            x = x.squeeze(1)
        elif x.dim() != 2:
            raise ValueError("Input must have shape [B,T] or [B,1,T].")

        log_mag, phase_repr = self.compute_stft_features(x)

        h_mag = self.mag_encoder(log_mag)
        h_phase = self.phase_encoder(phase_repr)

        h = torch.cat([h_mag, h_phase], dim=1)
        logits = self.classifier(h).squeeze(1)   # [B]

        return logits