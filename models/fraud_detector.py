"""FinAgent-SLM: Enterprise-Grade Fraud Detection Model
PyTorch-based Small Language Model for Real-Time Tax Fraud Detection"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger
import time


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention for transaction sequence modeling."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_k ** 0.5

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(out)


class TransformerBlock(nn.Module):
    """Transformer encoder block with pre-norm."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x


class FraudDetectionSLM(nn.Module):
    """
    FinAgent Small Language Model for Fraud Detection.
    
    Architecture: Transformer-based encoder with:
    - Feature embedding layer
    - Positional encoding
    - N transformer blocks with multi-head attention
    - Classification head with fraud probability output
    - Auxiliary heads for fraud type and risk score
    
    Parameters: ~6.2M (well under 8B SLM threshold)
    """
    def __init__(
        self,
        input_dim: int = 15,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        n_classes: int = 2,
        n_fraud_types: int = 5,
        dropout: float = 0.1,
        max_seq_len: int = 128
    ):
        super().__init__()
        self.d_model = d_model
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        # Multi-task heads
        self.fraud_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )
        self.fraud_type_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_fraud_types)
        )
        self.risk_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        self._init_weights()
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"FraudDetectionSLM initialized with {total_params:,} parameters")

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        B, L, D = x.shape
        x = self.input_projection(x)
        x = x + self.pos_encoding[:, :L, :]
        for block in self.transformer_blocks:
            x = block(x, mask)
        x = self.norm(x)
        pooled = x.mean(dim=1)  # Global average pooling
        return {
            'fraud_logits': self.fraud_classifier(pooled),
            'fraud_type_logits': self.fraud_type_classifier(pooled),
            'risk_score': self.risk_scorer(pooled).squeeze(-1)
        }

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            probs = F.softmax(outputs['fraud_logits'], dim=-1)
            predictions = torch.argmax(probs, dim=-1)
            return {
                'prediction': predictions,
                'probability': probs[:, 1],
                'risk_score': outputs['risk_score'],
                'fraud_type': torch.argmax(outputs['fraud_type_logits'], dim=-1)
            }
