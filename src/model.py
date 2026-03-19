"""
src/model.py
============
LSMTransformer: codificador Transformer para reconocimiento de glosas LSM.

Arquitectura:
  - FlatEmbedding    : (B, T, 399) → (B, T, d_model)
  - PositionalEncoding
  - TransformerEncoder (N capas, multi-head attention + FFN)
  - GlosaHead        : (B, d_model) → logits (B, num_classes)
  - EndTriggerHead   : (B, window*d_model) → prob. fin de seña (B, 1)

También incluye el módulo EADM (Energy-Based Attention Dropout Mask)
que se activa sólo durante el entrenamiento.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Codificación posicional sinusoidal estándar (Vaswani et al., 2017).
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 210, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0)          # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, d_model)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# EADM – Energy-Based Attention Dropout Mask
# ─────────────────────────────────────────────────────────────────────────────

class EADMHook:
    """
    Registra pesos de atención de un TransformerEncoderLayer y
    calcula la máscara de dropout estructurado para los embeddings.

    Uso:
        hook = EADMHook(layer, p_drop=0.3, percentile=80)
        # Tras el forward del transformer:
        mask = hook.compute_mask(embeddings)   # (B, T, 1) bool
        embeddings = embeddings * (~mask).float()
    """

    def __init__(
        self,
        layer: nn.TransformerEncoderLayer,
        p_drop: float = 0.3,
        percentile: float = 80.0,
    ):
        self.p_drop = p_drop
        self.percentile = percentile
        self._attn_weights: Optional[Tensor] = None

        # Registrar hook en la sub-capa de atención
        layer.self_attn.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        # output de MultiheadAttention = (attn_output, attn_weights)
        if isinstance(output, tuple) and len(output) == 2:
            self._attn_weights = output[1]   # (B, T, T)

    def compute_dropout(self, embeddings: Tensor) -> Tensor:
        """
        Aplica dropout estructurado sobre posiciones de alta energía.
        embeddings: (B, T, d_model)
        Returns   : (B, T, d_model)
        """
        if self._attn_weights is None or not self.training:
            return embeddings

        # Energía por posición: suma de atención recibida (columnas)
        energy = self._attn_weights.sum(dim=1)           # (B, T)
        threshold = torch.quantile(energy, self.percentile / 100.0, dim=1, keepdim=True)  # (B, 1)
        high_energy = energy > threshold                  # (B, T) bool
        drop = high_energy & (torch.rand_like(energy) < self.p_drop)  # (B, T)
        mask = drop.unsqueeze(-1).float()                # (B, T, 1)
        return embeddings * (1.0 - mask)

    @property
    def training(self) -> bool:
        return self._attn_weights is not None  # Proxy; usar model.training externo


# ─────────────────────────────────────────────────────────────────────────────
# MLP Head genérico
# ─────────────────────────────────────────────────────────────────────────────

class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# LSMTransformer
# ─────────────────────────────────────────────────────────────────────────────

class LSMTransformer(nn.Module):
    """
    Transformer para reconocimiento de glosas de LSM sobre secuencias de landmarks.

    Parameters
    ----------
    num_classes    : Número de glosas (249 en el dataset base).
    input_dim      : 133 × 3 = 399 (landmarks aplanados).
    d_model        : Dimensión interna del Transformer (512).
    nhead          : Número de cabezas de atención (8).
    num_layers     : Capas de encoder (6).
    dim_feedforward: Dim. FFN interna (2048).
    dropout        : Dropout global (0.1).
    tmax           : Longitud máxima de secuencia (200).
    trigger_window : Nro. de frames finales para el head de end-trigger (30).
    use_eadm       : Activar EADM durante entrenamiento.
    eadm_p_drop    : Probabilidad de dropout estructurado EADM.
    eadm_percentile: Percentil de energía para EADM.
    """

    def __init__(
        self,
        num_classes:     int   = 249,
        input_dim:       int   = 399,
        d_model:         int   = 512,
        nhead:           int   = 8,
        num_layers:      int   = 6,
        dim_feedforward: int   = 2048,
        dropout:         float = 0.1,
        tmax:            int   = 200,
        trigger_window:  int   = 30,
        use_eadm:        bool  = True,
        eadm_p_drop:     float = 0.3,
        eadm_percentile: float = 80.0,
    ):
        super().__init__()
        self.d_model        = d_model
        self.tmax           = tmax
        self.trigger_window = trigger_window
        self.use_eadm       = use_eadm

        # 1. Embedding espacial (proyección lineal)
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # 2. Positional Encoding
        self.pos_enc = PositionalEncoding(d_model, max_len=tmax + 10, dropout=dropout)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,          # (B, T, d_model)
            norm_first=True,           # Pre-LN: más estable en entrenamiento
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # EADM: hook sobre la primera capa (proxy de energía)
        if use_eadm:
            first_layer = self.transformer.layers[0]
            # Necesitamos need_weights=True en MultiheadAttention
            first_layer.self_attn.need_weights = True
            self._eadm = EADMHook(first_layer, p_drop=eadm_p_drop, percentile=eadm_percentile)
        else:
            self._eadm = None

        # 4. Head de clasificación de glosa
        self.glosa_head = MLPHead(d_model, d_model, num_classes, dropout=dropout)

        # 5. Head de end-trigger (últimos `trigger_window` frames)
        self.trigger_head = MLPHead(d_model * trigger_window, 128, 1, dropout=dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x: Tensor,                          # (B, T, 399)
        valid_mask: Optional[Tensor] = None,  # (B, T) bool — True en frames reales
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns
        -------
        glosa_logits   : (B, num_classes)
        trigger_logits : (B, 1)  — sin sigmoid; aplicar en loss
        """
        B, T, _ = x.shape

        # Construir padding_mask para el Transformer:
        # True en posiciones que deben ser IGNORADAS (padding)
        if valid_mask is not None:
            src_key_padding_mask = ~valid_mask   # (B, T)
        else:
            src_key_padding_mask = None

        # 1. Embedding + PE
        h = self.embedding(x)                   # (B, T, d_model)
        h = self.pos_enc(h)

        # 2. Transformer
        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)  # (B, T, d_model)

        # 3. EADM (solo training)
        if self.use_eadm and self.training and self._eadm is not None:
            h = self._eadm.compute_dropout(h)

        # 4. Glosa head: promedio de frames válidos (masked mean)
        if valid_mask is not None:
            mask_f = valid_mask.float().unsqueeze(-1)          # (B, T, 1)
            pooled = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)  # (B, d_model)
        else:
            pooled = h.mean(dim=1)

        glosa_logits = self.glosa_head(pooled)                 # (B, num_classes)

        # 5. End-trigger head: últimos `trigger_window` frames
        win = min(self.trigger_window, T)
        last_h = h[:, -win:, :]                               # (B, win, d_model)
        # Pad si la ventana es menor al objetivo
        if win < self.trigger_window:
            pad = torch.zeros(B, self.trigger_window - win, self.d_model, device=h.device)
            last_h = torch.cat([pad, last_h], dim=1)
        trigger_input = last_h.reshape(B, -1)                 # (B, trigger_window*d_model)
        trigger_logits = self.trigger_head(trigger_input)     # (B, 1)

        return glosa_logits, trigger_logits

    # ── inferencia ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        x: Tensor,
        valid_mask: Optional[Tensor] = None,
        glosa_conf_thresh: float = 0.7,
        trigger_thresh:    float = 0.85,
    ) -> dict:
        """
        Inferencia completa con thresholds de decisión.

        Returns
        -------
        dict con 'glosa_idx', 'glosa_conf', 'end_prob', 'should_emit'
        """
        self.eval()
        glosa_logits, trigger_logits = self(x, valid_mask)
        probs = F.softmax(glosa_logits, dim=-1)
        end_prob = torch.sigmoid(trigger_logits)

        glosa_idx  = probs.argmax(dim=-1)
        glosa_conf = probs.max(dim=-1).values
        should_emit = (end_prob.squeeze(-1) > trigger_thresh) & (glosa_conf > glosa_conf_thresh)

        return {
            "glosa_idx":   glosa_idx.cpu(),
            "glosa_conf":  glosa_conf.cpu(),
            "end_prob":    end_prob.squeeze(-1).cpu(),
            "should_emit": should_emit.cpu(),
            "top3_idx":    probs.topk(3, dim=-1).indices.cpu(),
        }

    # ── conteo de parámetros ──────────────────────────────────────────────────

    def param_count(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"Parámetros: {total:,} total | {trainable:,} entrenables"