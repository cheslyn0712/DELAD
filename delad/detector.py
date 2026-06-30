"""Sequential detector operating on chunk-level event feature sequences."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from delad.session_aggregation import aggregate_transition_probabilities, normalize_session_aggregation_method


class ResidualMLPBlock(nn.Module):
    """Pre-norm residual MLP block for stronger event classification."""

    def __init__(self, hidden_dim: int, expansion_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, expansion_dim)
        self.fc2 = nn.Linear(expansion_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x


class SequentialEventDetector(nn.Module):
    """A recurrent detector with gated state updates plus a deep MLP head.

    Input:
        features: [B, T, K]

    Output:
        logits: [B, T, 2]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        mlp_layers: int = 6,
        mlp_expansion: int = 4,
        fusion_with_current: bool = True,
        recurrent_type: str = "lstm",
        head_layers: int = 3,
        head_activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        recurrent_type_norm = str(recurrent_type).strip().lower()
        if recurrent_type_norm not in {"gru", "lstm"}:
            raise ValueError(
                f"Unsupported recurrent_type={recurrent_type!r}. Available: 'gru', 'lstm'."
            )
        recurrent_cls = nn.LSTM if recurrent_type_norm == "lstm" else nn.GRU
        self.recurrent = recurrent_cls(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.recurrent_type = recurrent_type_norm
        self.state_norm = nn.LayerNorm(hidden_dim)
        self.state_dropout = nn.Dropout(dropout)
        self.fusion_with_current = fusion_with_current

        classifier_input_dim = hidden_dim * 2 if fusion_with_current else hidden_dim
        self.fusion = nn.Linear(classifier_input_dim, hidden_dim)
        expansion_dim = hidden_dim * mlp_expansion
        self.blocks = nn.ModuleList(
            [
                ResidualMLPBlock(
                    hidden_dim=hidden_dim,
                    expansion_dim=expansion_dim,
                    dropout=dropout,
                )
                for _ in range(mlp_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        activation_name = str(head_activation).strip().lower()
        if activation_name == "relu":
            act_factory = nn.ReLU
        elif activation_name == "silu":
            act_factory = nn.SiLU
        else:
            activation_name = "gelu"
            act_factory = nn.GELU
        self.head_activation = activation_name
        head_depth = max(1, int(head_layers))
        head_modules = []
        in_dim = hidden_dim
        for _ in range(head_depth - 1):
            out_dim = max(8, in_dim // 2)
            head_modules.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    act_factory(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = out_dim
        head_modules.append(nn.Linear(in_dim, 2))
        self.head = nn.Sequential(*head_modules)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.input_proj(features)
        recurrent_state, _ = self.recurrent(projected)
        recurrent_state = self.state_dropout(self.state_norm(recurrent_state))

        if self.fusion_with_current:
            classifier_input = torch.cat([projected, recurrent_state], dim=-1)
        else:
            classifier_input = recurrent_state

        hidden = self.fusion(classifier_input)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        return self.head(hidden)


def masked_event_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: float = 1.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Compute masked cross entropy over supervised event positions.

    ``pos_weight`` re-weights the positive (anomaly) class to counter the
    severe class imbalance of log anomaly detection; ``label_smoothing`` lets
    Optuna explore a slightly softer target, which often helps when the
    detector overfits the 100+100 balanced train subset.
    """

    num_classes = logits.shape[-1]
    weight_tensor: torch.Tensor | None = None
    if pos_weight != 1.0 and num_classes == 2:
        weight_tensor = torch.tensor(
            [1.0, float(pos_weight)],
            device=logits.device,
            dtype=logits.dtype,
        )
    loss = F.cross_entropy(
        logits.reshape(-1, num_classes),
        labels.reshape(-1),
        weight=weight_tensor,
        label_smoothing=float(label_smoothing),
        reduction="none",
    )
    mask = mask.reshape(-1).float()
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def session_aggregated_bce_loss(
    logits: torch.Tensor,
    chunk_labels: torch.Tensor,
    mask: torch.Tensor,
    method: str,
    topk: int = 3,
    pos_weight: float = 1.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Binary cross-entropy between aggregated session scores and chunk labels."""

    probs = torch.softmax(logits, dim=-1)[..., 1]
    agg_method = normalize_session_aggregation_method(method)
    scores = aggregate_transition_probabilities(probs, mask, agg_method, topk=topk)

    targets = chunk_labels.float()
    if label_smoothing > 0.0:
        targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing

    if pos_weight != 1.0:
        weights = torch.where(chunk_labels == 1, float(pos_weight), 1.0)
        return F.binary_cross_entropy(scores, targets, weight=weights)
    return F.binary_cross_entropy(scores, targets)


@torch.no_grad()
def decode_event_predictions(logits: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Convert detector logits into probabilities and hard predictions."""

    probabilities = torch.softmax(logits, dim=-1)[..., 1]
    predictions = (probabilities >= 0.5).long()
    return {
        "probabilities": probabilities,
        "predictions": predictions,
    }
