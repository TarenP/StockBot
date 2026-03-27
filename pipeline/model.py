"""
Transformer-based portfolio policy.

Architecture:
  1. Per-asset linear projection  → d_model
  2. Temporal Transformer encoder (self-attention over lookback window)
  3. Cross-asset attention         (attend across assets at final timestep)
  4. Actor head                    → portfolio weights (n_assets + 1)
  5. Critic head                   → scalar value estimate
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model)
        return self.dropout(x + self.pe[:, :x.size(1)])


class AssetEncoder(nn.Module):
    """
    Encodes a single asset's time series (lookback, n_features) → d_model vector.
    Uses a Transformer encoder over the time dimension.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm        = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, n_features)
        x = self.input_proj(x)          # (batch, lookback, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)         # (batch, lookback, d_model)
        return self.norm(x[:, -1, :])   # take last timestep → (batch, d_model)


class PortfolioTransformer(nn.Module):
    """
    Full policy + value network.

    Forward returns:
        logits  : (batch, n_assets + 1)  — raw weights before softmax
        value   : (batch, 1)             — state value estimate
    """

    def __init__(
        self,
        n_assets: int,
        n_features: int,
        lookback: int,
        d_model: int = 64,
        nhead_temporal: int = 4,
        nhead_cross: int = 4,
        num_temporal_layers: int = 2,
        num_cross_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_assets  = n_assets
        self.d_model   = d_model

        # Shared temporal encoder (weights shared across assets)
        self.asset_encoder = AssetEncoder(
            n_features, d_model, nhead_temporal, num_temporal_layers, dropout
        )

        # Cross-asset Transformer (attend across the n_assets dimension)
        cross_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead_cross, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.cross_attn = nn.TransformerEncoder(cross_layer, num_layers=num_cross_layers)

        # Actor: outputs logits for n_assets + 1 (cash)
        self.actor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),   # per-asset score
        )
        self.cash_logit = nn.Parameter(torch.zeros(1))   # learnable cash score

        # Critic: pool across assets → scalar
        self.critic = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, obs: torch.Tensor):
        """
        obs: (batch, lookback, n_assets, n_features)
        """
        batch, lookback, n_assets, n_features = obs.shape

        # Encode each asset independently over time
        # Reshape to (batch * n_assets, lookback, n_features)
        x = obs.permute(0, 2, 1, 3).reshape(batch * n_assets, lookback, n_features)
        asset_emb = self.asset_encoder(x)                          # (B*A, d_model)
        asset_emb = asset_emb.view(batch, n_assets, self.d_model)  # (B, A, d_model)

        # Cross-asset attention
        asset_emb = self.cross_attn(asset_emb)                     # (B, A, d_model)

        # Actor logits
        asset_logits = self.actor(asset_emb).squeeze(-1)           # (B, A)
        cash_logits  = self.cash_logit.expand(batch, 1)            # (B, 1)
        logits       = torch.cat([asset_logits, cash_logits], dim=-1)  # (B, A+1)

        # Critic value
        pooled = asset_emb.mean(dim=1)                             # (B, d_model)
        value  = self.critic(pooled)                               # (B, 1)

        return logits, value

    @torch.no_grad()
    def get_weights(self, obs: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Inference: returns softmax portfolio weights (B, n_assets+1)."""
        logits, _ = self.forward(obs)
        return F.softmax(logits / temperature, dim=-1)
