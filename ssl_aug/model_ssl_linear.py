import torch
import torch.nn as nn
import torchaudio


class AttentiveStatsPooling(nn.Module):
    def __init__(self, dim: int, attn_hidden: int = 128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        a = self.attn(x)                 # [B, T, 1]
        a = torch.softmax(a, dim=1)

        mean = torch.sum(a * x, dim=1)
        second = torch.sum(a * (x ** 2), dim=1)
        std = torch.sqrt(torch.clamp(second - mean ** 2, min=1e-6))

        return torch.cat([mean, std], dim=1)


class SSLLinearClassifier(nn.Module):
    def __init__(
        self,
        ssl_name: str = "wav2vec2_xlsr_300m",
        unfreeze_last_n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        if ssl_name == "wav2vec2_xlsr_300m":
            self.bundle = torchaudio.pipelines.WAV2VEC2_XLSR_300M
        elif ssl_name == "wavlm_base":
            self.bundle = torchaudio.pipelines.WAVLM_BASE
        elif ssl_name == "hubert_base":
            self.bundle = torchaudio.pipelines.HUBERT_BASE
        else:
            raise ValueError(f"Unknown ssl_name: {ssl_name}")

        self.ssl = self.bundle.get_model()

        params = self.bundle._params
        if isinstance(params, dict):
            ssl_dim = params["encoder_embed_dim"]
        else:
            ssl_dim = params.encoder_embed_dim

        # Freeze everything first
        for p in self.ssl.parameters():
            p.requires_grad = False

        # Unfreeze last N encoder transformer layers if possible
        self._unfreeze_last_n_layers(unfreeze_last_n_layers)

        self.pool = AttentiveStatsPooling(ssl_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * ssl_dim),
            nn.Dropout(dropout),
            nn.Linear(2 * ssl_dim, 1),
        )

    def _unfreeze_last_n_layers(self, n: int):
        if n <= 0:
            return

        # Torchaudio wav2vec2 bundles usually expose encoder.transformer.layers
        layers = None

        if hasattr(self.ssl, "encoder") and hasattr(self.ssl.encoder, "transformer"):
            tr = self.ssl.encoder.transformer
            if hasattr(tr, "layers"):
                layers = tr.layers

        if layers is None:
            # Fallback: do nothing if structure differs
            return

        total = len(layers)
        n = min(n, total)

        for layer in layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True

        # Also unfreeze encoder layer norm if present
        if hasattr(self.ssl.encoder.transformer, "layer_norm"):
            for p in self.ssl.encoder.transformer.layer_norm.parameters():
                p.requires_grad = True

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: [B, T] or [B, 1, T]
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        elif wav.dim() != 2:
            raise ValueError("Input must have shape [B,T] or [B,1,T].")

        feats_list, _ = self.ssl.extract_features(wav)
        x = feats_list[-1]  # [B, T_ssl, D]

        x = self.pool(x)    # [B, 2D]
        logits = self.classifier(x).squeeze(1)
        return logits