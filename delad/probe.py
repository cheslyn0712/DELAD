"""Frozen LLM probing + transcoder feature selection for circuit tracing."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from delad.feature_labels import FeatureExampleStore, build_feature_description, format_feature_for_prompt
from delad.utils import resolve_path
from delad.utils import log_status

try:
    from circuit_tracer.transcoder.single_layer_transcoder import load_transcoder_set
    from circuit_tracer.utils.hf_utils import resolve_transcoder_paths
except Exception:  # pragma: no cover - runtime dependency
    load_transcoder_set = None
    resolve_transcoder_paths = None


EPS = 1e-6
DEFAULT_ACTIVATION_THRESHOLD = 1e-4
DEFAULT_MIN_CLASS_MEAN = 1e-4
DEFAULT_MIN_SUPPORT_EVENTS = 3
DEFAULT_MIN_SUPPORT_FRACTION = 0.02
DEFAULT_MIN_MEAN_SEPARATION = 1e-3


@dataclass
class SelectedFeatureNode:
    rank: int
    layer_idx: int
    neuron_idx: int
    feature_idx: int
    score: float
    mean_norm: float
    mean_anom: float
    std_norm: float
    std_anom: float
    cosine_similarity: float
    cosine_difference: float
    separation: float
    magnitude: float
    support_norm: float
    support_anom: float
    count_norm: int
    count_anom: int
    eligible: bool
    max_anomaly_activation: float
    feature_label: Optional[str] = None
    feature_description: Optional[str] = None
    feature_summary: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": int(self.rank),
            "layer_idx": int(self.layer_idx),
            "neuron_idx": int(self.neuron_idx),
            "feature_idx": int(self.feature_idx),
            "score": float(self.score),
            "mean_norm": float(self.mean_norm),
            "mean_anom": float(self.mean_anom),
            "std_norm": float(self.std_norm),
            "std_anom": float(self.std_anom),
            "cosine_similarity": float(self.cosine_similarity),
            "cosine_difference": float(self.cosine_difference),
            "separation": float(self.separation),
            "magnitude": float(self.magnitude),
            "support_norm": float(self.support_norm),
            "support_anom": float(self.support_anom),
            "count_norm": int(self.count_norm),
            "count_anom": int(self.count_anom),
            "eligible": bool(self.eligible),
            "max_anomaly_activation": float(self.max_anomaly_activation),
            "feature_label": self.feature_label,
            "feature_description": self.feature_description,
            "feature_summary": self.feature_summary,
        }


def _as_float_tensor(data: torch.Tensor | Sequence[float]) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.detach().float()
    return torch.tensor(data, dtype=torch.float32)


def cosine_similarity_scalars(
    mean_norm: torch.Tensor,
    mean_anom: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    denom = mean_norm.abs() * mean_anom.abs() + float(eps)
    return (mean_norm * mean_anom) / denom


def cosine_difference_scalars(
    mean_norm: torch.Tensor,
    mean_anom: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    return 1.0 - cosine_similarity_scalars(mean_norm=mean_norm, mean_anom=mean_anom, eps=eps)


def _activation_support_stats(
    activations: torch.Tensor,
    threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    active = activations > float(threshold)
    counts = active.sum(dim=0).float()
    support = counts / float(max(1, activations.shape[0]))
    return counts, support


def _compute_layer_feature_scores(
    layer_activations: torch.Tensor,
    labels: torch.Tensor,
    *,
    activation_threshold: float,
    min_class_mean: float,
    min_anomaly_support_events: int,
    min_anomaly_support_fraction: float,
    min_normal_support_events: int,
    min_normal_support_fraction: float,
    min_mean_separation: float,
) -> Dict[str, torch.Tensor]:
    acts = layer_activations.float()
    y = labels.long()
    normal_mask = y == 0
    anomaly_mask = y == 1

    n_norm = int(normal_mask.sum().item())
    n_anom = int(anomaly_mask.sum().item())
    if n_norm <= 0 or n_anom <= 0:
        d = acts.shape[1]
        zero = torch.zeros(d, dtype=torch.float32, device=acts.device)
        return {
            "scores": zero,
            "eligible": zero.bool(),
            "mean_norm": zero,
            "mean_anom": zero,
            "std_norm": zero,
            "std_anom": zero,
            "count_norm": zero,
            "count_anom": zero,
            "support_norm": zero,
            "support_anom": zero,
            "cosine_similarity": zero,
            "cosine_difference": zero,
            "separation": zero,
            "magnitude": zero,
            "max_anomaly_activation": zero,
        }

    norm_acts = acts[normal_mask]
    anom_acts = acts[anomaly_mask]
    mean_norm = norm_acts.mean(dim=0)
    mean_anom = anom_acts.mean(dim=0)
    std_norm = norm_acts.std(dim=0, unbiased=False)
    std_anom = anom_acts.std(dim=0, unbiased=False)
    count_norm, support_norm = _activation_support_stats(norm_acts, threshold=activation_threshold)
    count_anom, support_anom = _activation_support_stats(anom_acts, threshold=activation_threshold)
    max_anomaly_activation = anom_acts.max(dim=0).values

    min_norm_support = max(
        int(min_normal_support_events),
        int(math.ceil(float(min_normal_support_fraction) * float(n_norm))),
    )
    min_anom_support = max(
        int(min_anomaly_support_events),
        int(math.ceil(float(min_anomaly_support_fraction) * float(n_anom))),
    )

    magnitude = (mean_anom - mean_norm).abs()
    cosine_sim = cosine_similarity_scalars(mean_norm=mean_norm, mean_anom=mean_anom, eps=EPS)
    cosine_diff = 1.0 - cosine_sim
    relative_sep = magnitude / (mean_anom + mean_norm + EPS)
    separation = torch.maximum(cosine_diff, relative_sep)

    eligible = (
        (mean_norm >= float(min_class_mean))
        & (mean_anom >= float(min_class_mean))
        & (count_norm >= float(min_norm_support))
        & (count_anom >= float(min_anom_support))
        & (magnitude >= float(min_mean_separation))
    )
    anomaly_focus = (mean_anom / (mean_anom + mean_norm + EPS)).clamp(0.05, 1.0)
    scores = separation * magnitude * torch.sqrt(torch.clamp(support_anom, min=0.0)) * anomaly_focus
    scores = torch.where(eligible, scores, torch.zeros_like(scores))
    scores = torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores))

    return {
        "scores": scores,
        "eligible": eligible,
        "mean_norm": mean_norm,
        "mean_anom": mean_anom,
        "std_norm": std_norm,
        "std_anom": std_anom,
        "count_norm": count_norm,
        "count_anom": count_anom,
        "support_norm": support_norm,
        "support_anom": support_anom,
        "cosine_similarity": cosine_sim,
        "cosine_difference": cosine_diff,
        "separation": separation,
        "magnitude": magnitude,
        "max_anomaly_activation": max_anomaly_activation,
    }


def score_neurons(
    layer_activations: Mapping[int, torch.Tensor],
    labels: Sequence[int] | torch.Tensor,
    top_k: int,
    *,
    activation_threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
    min_class_mean: float = DEFAULT_MIN_CLASS_MEAN,
    min_anomaly_support_events: int = DEFAULT_MIN_SUPPORT_EVENTS,
    min_anomaly_support_fraction: float = DEFAULT_MIN_SUPPORT_FRACTION,
    min_normal_support_events: int = DEFAULT_MIN_SUPPORT_EVENTS,
    min_normal_support_fraction: float = DEFAULT_MIN_SUPPORT_FRACTION,
    min_mean_separation: float = DEFAULT_MIN_MEAN_SEPARATION,
) -> Dict[str, Any]:
    y = _as_float_tensor(labels).long()
    all_nodes: List[SelectedFeatureNode] = []
    layer_statistics: Dict[str, Dict[str, Any]] = {}
    num_candidates = 0
    num_eligible = 0

    for layer_idx in sorted(layer_activations.keys()):
        acts = layer_activations[int(layer_idx)].detach().float()
        stats = _compute_layer_feature_scores(
            acts,
            y,
            activation_threshold=activation_threshold,
            min_class_mean=min_class_mean,
            min_anomaly_support_events=min_anomaly_support_events,
            min_anomaly_support_fraction=min_anomaly_support_fraction,
            min_normal_support_events=min_normal_support_events,
            min_normal_support_fraction=min_normal_support_fraction,
            min_mean_separation=min_mean_separation,
        )
        scores = stats["scores"]
        eligible = stats["eligible"]
        num_candidates += int(scores.numel())
        num_eligible += int(eligible.sum().item())

        layer_statistics[str(int(layer_idx))] = {
            "num_features": int(scores.numel()),
            "num_eligible": int(eligible.sum().item()),
            # Keep this compact to avoid giant JSON blobs.
            "scores": scores.detach().cpu()[: min(2048, scores.numel())].tolist(),
        }

        eligible_idx = torch.nonzero(eligible, as_tuple=False).reshape(-1)
        for feat_idx in eligible_idx.tolist():
            all_nodes.append(
                SelectedFeatureNode(
                    rank=-1,
                    layer_idx=int(layer_idx),
                    neuron_idx=int(feat_idx),
                    feature_idx=int(feat_idx),
                    score=float(scores[feat_idx].item()),
                    mean_norm=float(stats["mean_norm"][feat_idx].item()),
                    mean_anom=float(stats["mean_anom"][feat_idx].item()),
                    std_norm=float(stats["std_norm"][feat_idx].item()),
                    std_anom=float(stats["std_anom"][feat_idx].item()),
                    cosine_similarity=float(stats["cosine_similarity"][feat_idx].item()),
                    cosine_difference=float(stats["cosine_difference"][feat_idx].item()),
                    separation=float(stats["separation"][feat_idx].item()),
                    magnitude=float(stats["magnitude"][feat_idx].item()),
                    support_norm=float(stats["support_norm"][feat_idx].item()),
                    support_anom=float(stats["support_anom"][feat_idx].item()),
                    count_norm=int(round(float(stats["count_norm"][feat_idx].item()))),
                    count_anom=int(round(float(stats["count_anom"][feat_idx].item()))),
                    eligible=True,
                    max_anomaly_activation=float(stats["max_anomaly_activation"][feat_idx].item()),
                )
            )

    all_nodes.sort(key=lambda n: n.score, reverse=True)
    selected = [n for n in all_nodes if n.score > 0.0][: int(max(0, top_k))]
    if not selected:
        # Smoke / tiny subsets may make all eligible scores collapse to 0.
        # Keep top-k eligible nodes so the pipeline can still run end-to-end.
        selected = all_nodes[: int(max(0, top_k))]
    for idx, node in enumerate(selected):
        node.rank = int(idx)

    eligibility_metadata = {
        "activation_threshold": float(activation_threshold),
        "min_class_mean": float(min_class_mean),
        "min_anomaly_support_events": int(min_anomaly_support_events),
        "min_anomaly_support_fraction": float(min_anomaly_support_fraction),
        "min_normal_support_events": int(min_normal_support_events),
        "min_normal_support_fraction": float(min_normal_support_fraction),
        "min_mean_separation": float(min_mean_separation),
        "requirements": (
            "mean_normal >= min_class_mean, mean_anomaly >= min_class_mean, "
            "count_normal >= min_normal_support, count_anomaly >= min_anomaly_support, "
            "magnitude >= min_mean_separation"
        ),
    }
    scoring_metadata = {
        **eligibility_metadata,
        "formula": (
            "eligible_only: max(cosine_difference, relative_separation) * magnitude * "
            "sqrt(support_anomaly) * clamp(mean_anomaly / (mean_anomaly + mean_normal), 0.05, 1)"
        ),
    }

    selected_rows = [n.to_dict() for n in selected]
    return {
        "selected_neurons": selected_rows,
        "selected_feature_nodes": selected_rows,
        "num_candidates": int(num_candidates),
        "num_eligible": int(num_eligible),
        "top_k": int(top_k),
        "scoring_name": "discriminative_support_score",
        "eligibility_metadata": eligibility_metadata,
        "scoring_metadata": scoring_metadata,
        "layer_statistics": layer_statistics,
    }


def _normalize_local_config_paths(config: Dict[str, Any], root: Path) -> Dict[str, Any]:
    fixed = dict(config)
    transcoders = fixed.get("transcoders")
    if isinstance(transcoders, list):
        out: List[Any] = []
        for item in transcoders:
            if isinstance(item, str) and not item.startswith("hf://"):
                p = Path(item)
                if not p.is_absolute():
                    p = root / p
                out.append(p.relative_to(root.resolve()).as_posix())
            else:
                out.append(item)
        fixed["transcoders"] = out
    return fixed


def _parse_layer_index_from_name(path: Path) -> int:
    stem = path.stem
    if stem.startswith("layer_"):
        suffix = stem.split("_", 1)[1]
        if suffix.isdigit():
            return int(suffix)
    return 10**9


def _discover_local_transcoders(root: Path) -> List[str]:
    candidates: List[Path] = []
    candidates.extend(root.glob("layer_*.safetensors"))
    candidates.extend(root.glob("layer_*.npz"))
    # Some exports place files under nested folders.
    if not candidates:
        candidates.extend(root.glob("**/layer_*.safetensors"))
        candidates.extend(root.glob("**/layer_*.npz"))
    root_resolved = root.resolve()
    dedup = sorted({p.resolve() for p in candidates}, key=lambda p: (_parse_layer_index_from_name(p), str(p)))
    return [p.relative_to(root_resolved).as_posix() for p in dedup]


class CircuitTracerFeatureProjector:
    """Per-layer transcoder feature projector operating on CPU."""

    def __init__(self, transcoder_root: str | Path) -> None:
        if load_transcoder_set is None or resolve_transcoder_paths is None:
            raise RuntimeError("circuit_tracer is required for CircuitTracerFeatureProjector.")
        self.transcoder_root = resolve_path(transcoder_root)
        cfg_path = self.transcoder_root / "config.yaml"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Missing transcoder config: {cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        self.config = _normalize_local_config_paths(config, self.transcoder_root)
        if "transcoders" not in self.config or not isinstance(self.config.get("transcoders"), list):
            discovered = _discover_local_transcoders(self.transcoder_root)
            if discovered:
                self.config["transcoders"] = discovered
        if "transcoders" in self.config and isinstance(self.config["transcoders"], list):
            normalized: List[str] = []
            for p in self.config["transcoders"]:
                p_str = str(p)
                if p_str.startswith("hf://"):
                    normalized.append(p_str)
                else:
                    normalized.append(
                        str(resolve_path(p_str, base=self.transcoder_root))
                    )
            self.config["transcoders"] = normalized
        if "repo_id" not in self.config:
            model_hint = str(self.config.get("model_name", "")).strip()
            if model_hint:
                self.config["repo_id"] = model_hint
        if "transcoders" not in self.config and "repo_id" not in self.config:
            raise ValueError(
                f"Transcoder config missing both 'transcoders' and 'repo_id': {cfg_path}. "
                "Provide local layer_*.safetensors/.npz files or set repo_id."
            )
        try:
            resolved = resolve_transcoder_paths(self.config)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to resolve transcoders from {cfg_path}. "
                f"config_has_transcoders={isinstance(self.config.get('transcoders'), list)} "
                f"config_has_repo_id={'repo_id' in self.config} "
                f"transcoder_root={self.transcoder_root}"
            ) from exc
        if not resolved:
            raise RuntimeError(f"No transcoders resolved from {cfg_path}")
        missing_hooks = [k for k in ("feature_input_hook", "feature_output_hook") if not self.config.get(k)]
        if missing_hooks:
            raise ValueError(
                f"Transcoder config missing required hook fields {missing_hooks}: {cfg_path}. "
                "Keep the same circuit-tracer config format used in test.py."
            )
        suffix = Path(str(resolved[min(resolved.keys())])).suffix.lower()
        special_load_fn = "gemma-scope" if suffix == ".npz" else None
        self.transcoder_kind = "gemma-scope" if special_load_fn else "relu"
        self.transcoders = load_transcoder_set(
            resolved,
            scan=str(self.transcoder_root),
            feature_input_hook=self.config["feature_input_hook"],
            feature_output_hook=self.config["feature_output_hook"],
            device=torch.device("cpu"),
            dtype=torch.float32,
            special_load_fn=special_load_fn,
            lazy_encoder=False,
            lazy_decoder=True,
        )

    def _get_layer_transcoder(self, layer_idx: int) -> Any:
        if hasattr(self.transcoders, "transcoders"):
            return self.transcoders.transcoders[int(layer_idx)]
        raise RuntimeError("Unexpected transcoder set shape.")

    def layer_dims(self, layer_idx: int) -> Tuple[int, int]:
        tc = self._get_layer_transcoder(layer_idx)
        d_model = int(getattr(tc, "d_model", 0))
        d_trans = int(getattr(tc, "d_transcoder", 0))
        if d_model <= 0 and hasattr(tc, "W_enc"):
            d_model = int(tc.W_enc.shape[0])
            d_trans = int(tc.W_enc.shape[1])
        return d_model, d_trans

    def _encode_full_layer(self, layer_idx: int, hidden: torch.Tensor) -> torch.Tensor:
        tc = self._get_layer_transcoder(layer_idx)
        if hasattr(tc, "encode"):
            acts = tc.encode(hidden.cpu())
            return acts.float().cpu()
        if hasattr(tc, "W_enc"):
            w = tc.W_enc.float().cpu()
            b = getattr(tc, "b_enc", None)
            out = hidden.cpu().float() @ w
            if b is not None:
                out = out + b.float().cpu()
            return torch.relu(out)
        raise RuntimeError(f"Layer {layer_idx} transcoder has no known encode path.")

    def encode_full(self, layer_idx: int, hidden: torch.Tensor) -> torch.Tensor:
        d_model, _ = self.layer_dims(layer_idx)
        if d_model and int(hidden.shape[-1]) != d_model:
            raise ValueError(
                f"d_model mismatch at layer {layer_idx}: hidden={hidden.shape[-1]} vs transcoder={d_model}"
            )
        return self._encode_full_layer(layer_idx, hidden)

    def encode_selected(self, layer_idx: int, hidden: torch.Tensor, feature_indices: Sequence[int]) -> torch.Tensor:
        if not feature_indices:
            return torch.empty(hidden.shape[0], 0, dtype=torch.float32)
        full = self.encode_full(layer_idx, hidden)
        idx = torch.tensor([int(i) for i in feature_indices], dtype=torch.long)
        return full.index_select(dim=1, index=idx)


class FrozenLLMProbe:
    """Probe hidden states from a frozen causal LM and project via transcoders."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = dict(cfg)
        self.model_name = str(self.cfg.get("model_name", "unknown"))
        self.llm_path = str(resolve_path(self.cfg.get("llm_path") or ""))
        self.tokenizer_path = str(
            resolve_path(self.cfg.get("tokenizer_path") or self.llm_path)
        )
        if not self.llm_path:
            raise ValueError("cfg['llm_path'] is required for FrozenLLMProbe")
        self.max_length = int(self.cfg.get("max_length", 512))
        self.capture_layers = [int(x) for x in (self.cfg.get("capture_layers") or [])]
        self.event_feature_position = str(self.cfg.get("event_feature_position", "last_prefix_token"))
        self.batch_size = int(self.cfg.get("selection_batch_size", 4))
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.llm_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.model.eval()
        self.projector = CircuitTracerFeatureProjector(self.cfg["transcoder_path"])
        num_transcoder_layers = len(self.projector.transcoders.transcoders)
        if not self.capture_layers:
            self.capture_layers = list(range(num_transcoder_layers))
        else:
            clipped = [int(x) for x in self.capture_layers if int(x) < num_transcoder_layers]
            if not clipped:
                raise ValueError(
                    f"capture_layers={self.capture_layers} has no valid indices; "
                    f"transcoder has {num_transcoder_layers} layers."
                )
            if len(clipped) != len(self.capture_layers):
                from delad.utils import log_status

                log_status(
                    f"WARN: probe clipped capture_layers "
                    f"{self.capture_layers} -> {clipped} (transcoder_layers={num_transcoder_layers})"
                )
            self.capture_layers = clipped
        self.feature_store = FeatureExampleStore(
            self.cfg.get("circuit_feature_dir")
            or str(resolve_path(self.cfg["transcoder_path"]) / "features")
        )

    @property
    def device(self) -> torch.device:
        try:
            emb = self.model.get_input_embeddings()
            if emb is not None and hasattr(emb, "weight"):
                return emb.weight.device
        except Exception:
            pass
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            pass
        if hasattr(self.model, "device"):
            return torch.device(self.model.device)
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _resolve_feature_token_positions(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Resolve per-sample token position used for feature extraction."""
        last = attention_mask.sum(dim=1) - 1
        mode = str(self.event_feature_position).strip().lower()
        if mode in ("last_prefix_token", "last_token", ""):
            return last
        if mode in ("second_last_token", "penultimate"):
            return torch.clamp(last - 1, min=0)
        # Fallback to legacy behavior for unknown modes.
        return last

    @staticmethod
    def _hidden_state_index(layer_idx: int, num_hidden_states: int) -> int:
        # HF hidden_states[0] is embeddings; transformer block i -> hidden_states[i+1].
        return min(max(1, int(layer_idx) + 1), num_hidden_states - 1)

    def _encode_hidden(self, prompts: Sequence[str]) -> Dict[int, torch.Tensor]:
        batch = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = self.model(**batch, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states
        attn = batch["attention_mask"]
        pos = self._resolve_feature_token_positions(attn)
        per_layer_hidden: Dict[int, torch.Tensor] = {}
        for layer_idx in self.capture_layers:
            hs_idx = self._hidden_state_index(int(layer_idx), len(hidden_states))
            hs = hidden_states[hs_idx]
            gathered = hs[torch.arange(hs.shape[0], device=hs.device), pos]
            per_layer_hidden[int(layer_idx)] = gathered.detach().float().cpu()
        return per_layer_hidden

    @torch.no_grad()
    def probe_prompts(
        self,
        prompts: Sequence[str],
        batch_size: Optional[int] = None,
        desc: str = "probe",
    ) -> Dict[int, torch.Tensor]:
        bs = int(batch_size or self.batch_size)
        out: Dict[int, List[torch.Tensor]] = {int(li): [] for li in self.capture_layers}
        for i in range(0, len(prompts), bs):
            part = prompts[i : i + bs]
            layer_hidden = self._encode_hidden(part)
            for layer_idx, hidden in layer_hidden.items():
                encoded = self.projector.encode_full(layer_idx, hidden)
                out.setdefault(int(layer_idx), []).append(encoded)
        merged: Dict[int, torch.Tensor] = {}
        for layer_idx, chunks in out.items():
            if chunks:
                merged[int(layer_idx)] = torch.cat(chunks, dim=0).float().cpu()
        return merged

    @torch.no_grad()
    def extract_selected_features(
        self,
        prompts: Sequence[str],
        selected_neurons: Sequence[Mapping[str, Any]],
        batch_size: int,
        *,
        split_name: str = "",
        outer_batch_index: Optional[int] = None,
        outer_batch_total: Optional[int] = None,
        event_indices: Optional[Sequence[int]] = None,
        event_chunk_ids: Optional[Sequence[str]] = None,
        log_every_micro_batches: int = 0,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        timing_collector: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        if not prompts or not selected_neurons:
            return torch.empty(len(prompts), 0, dtype=torch.float32)
        by_layer: Dict[int, List[Tuple[int, int]]] = {}
        for col_idx, node in enumerate(selected_neurons):
            layer = int(node.get("layer_idx", 0))
            feat = int(node.get("feature_idx", node.get("neuron_idx", 0)))
            by_layer.setdefault(layer, []).append((col_idx, feat))

        bs = int(max(1, batch_size))
        outputs: List[torch.Tensor] = []
        total_micro = (len(prompts) + bs - 1) // bs
        for i in range(0, len(prompts), bs):
            micro_started = time.perf_counter()
            micro_idx = (i // bs) + 1
            end = min(i + bs, len(prompts))
            part = prompts[i : i + bs]
            layer_hidden = self._encode_hidden(part)
            batch_k = torch.zeros(len(part), len(selected_neurons), dtype=torch.float32)
            for layer_idx, pairs in by_layer.items():
                hidden = layer_hidden.get(int(layer_idx))
                if hidden is None:
                    continue
                feat_idx = [f for _, f in pairs]
                layer_vals = self.projector.encode_selected(layer_idx, hidden, feat_idx)
                for j, (col_idx, _) in enumerate(pairs):
                    batch_k[:, int(col_idx)] = layer_vals[:, j]
            outputs.append(batch_k)
            micro_elapsed = max(0.0, time.perf_counter() - micro_started)
            payload = {
                "split_name": str(split_name),
                "outer_batch_index": int(outer_batch_index) if outer_batch_index is not None else None,
                "outer_batch_total": int(outer_batch_total) if outer_batch_total is not None else None,
                "micro_batch_index": int(micro_idx),
                "micro_batch_total": int(total_micro),
                "micro_batch_size": int(len(part)),
                "processed_in_outer_batch": int(end),
                "total_in_outer_batch": int(len(prompts)),
                "micro_elapsed_seconds": float(micro_elapsed),
                "micro_seconds_per_event": float(micro_elapsed / max(1, len(part))),
                "current_event_index": (
                    int(event_indices[end - 1]) if event_indices is not None and end - 1 < len(event_indices) else -1
                ),
                "current_chunk_id": (
                    str(event_chunk_ids[end - 1]) if event_chunk_ids is not None and end - 1 < len(event_chunk_ids) else ""
                ),
            }
            if timing_collector is not None:
                timing_collector.append(
                    {
                        "split_name": str(split_name),
                        "outer_batch_index": int(outer_batch_index) if outer_batch_index is not None else None,
                        "micro_batch_index": int(micro_idx),
                        "micro_batch_total": int(total_micro),
                        "num_events": int(len(part)),
                        "elapsed_seconds": float(micro_elapsed),
                        "seconds_per_event": float(micro_elapsed / max(1, len(part))),
                    }
                )
            if progress_callback is not None:
                progress_callback(payload)
            elif int(log_every_micro_batches) > 0 and (
                micro_idx == 1 or micro_idx == total_micro or micro_idx % int(log_every_micro_batches) == 0
            ):
                outer = ""
                if outer_batch_index is not None and outer_batch_total is not None:
                    outer = f" outer_batch={int(outer_batch_index)}/{int(outer_batch_total)}"
                log_status(
                    f"[{split_name}] micro progress{outer} "
                    f"micro={micro_idx}/{total_micro} events_in_batch={end}/{len(prompts)} "
                    f"current_chunk={payload['current_chunk_id']} current_event_index={payload['current_event_index']}"
                )
        if not outputs:
            return torch.empty(len(prompts), 0, dtype=torch.float32)
        return torch.cat(outputs, dim=0).float()

    @torch.no_grad()
    def extract_per_token_selected(
        self,
        prompt: str,
        selected_neurons: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        toks = self.tokenizer(
            [prompt],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        toks = {k: v.to(self.device) for k, v in toks.items()}
        with torch.no_grad():
            outputs = self.model(**toks, output_hidden_states=True, use_cache=False)
        token_ids = toks["input_ids"][0].detach().cpu().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

        by_layer: Dict[int, List[int]] = {}
        for node in selected_neurons:
            layer = int(node.get("layer_idx", 0))
            feat = int(node.get("feature_idx", node.get("neuron_idx", 0)))
            by_layer.setdefault(layer, []).append(feat)

        per_layer: Dict[int, torch.Tensor] = {}
        for layer_idx, feat_idx in by_layer.items():
            hs_idx = self._hidden_state_index(int(layer_idx), len(outputs.hidden_states))
            hs = outputs.hidden_states[hs_idx][0].detach().float().cpu()  # [T, D]
            vals = self.projector.encode_selected(layer_idx, hs, feat_idx)  # [T, K_layer]
            per_layer[int(layer_idx)] = vals
        return {"tokens": tokens, "per_layer_acts": per_layer}

    @torch.no_grad()
    def build_debug_feature_dump(
        self,
        prompts: Sequence[str],
        selected_neurons: Sequence[Mapping[str, Any]],
        *,
        top_n: int = 10,
        max_prompts: int = 2,
        dump_layers: Optional[Sequence[int]] = None,
        max_abs_diff_threshold: float = 1e-3,
    ) -> Dict[str, Any]:
        sample_prompts = list(prompts[: max(1, int(max_prompts))])
        if not sample_prompts:
            sample_prompts = ["hello world"]
        layers = [int(x) for x in (dump_layers or self.capture_layers[:1])]
        layer_idx = int(layers[0])
        hidden = self._encode_hidden(sample_prompts)[layer_idx]
        full = self.projector.encode_full(layer_idx, hidden)
        chosen = [n for n in selected_neurons if int(n.get("layer_idx", -1)) == layer_idx]
        if chosen:
            feat_idx = [int(n.get("feature_idx", n.get("neuron_idx", 0))) for n in chosen]
        else:
            feat_idx = list(torch.topk(full.abs().mean(dim=0), k=min(top_n, full.shape[1])).indices.tolist())
        selected = self.projector.encode_selected(layer_idx, hidden, feat_idx)
        full_pick = full[:, feat_idx]
        abs_diff = (full_pick - selected).abs()
        max_diff = float(abs_diff.max().item()) if abs_diff.numel() else 0.0
        top_rows = []
        for rank, idx in enumerate(feat_idx[: max(1, int(top_n))]):
            fv = float(full_pick[0, rank].item())
            sv = float(selected[0, rank].item())
            top_rows.append(
                {
                    "feature_idx": int(idx),
                    "full_value": fv,
                    "selected_value": sv,
                    "abs_diff": float(abs(fv - sv)),
                }
            )
        d_model, d_trans = self.projector.layer_dims(layer_idx)
        return {
            "model_name": self.model_name,
            "transcoder_path": str(self.cfg.get("transcoder_path")),
            "transcoder_kind": self.projector.transcoder_kind,
            "layer_idx": int(layer_idx),
            "d_model": int(d_model),
            "d_transcoder": int(d_trans),
            "input_name": "sample_prompts",
            "activation_mean": float(full.mean().item()),
            "activation_std": float(full.std(unbiased=False).item()),
            "activation_max": float(full.max().item()),
            "activation_min": float(full.min().item()),
            "top_abs_features": top_rows,
            "max_abs_diff_full_vs_selected": max_diff,
            "max_abs_diff_threshold": float(max_abs_diff_threshold),
            "diff_within_threshold": bool(max_diff <= float(max_abs_diff_threshold)),
        }

    def annotate_selected_features(self, selected_neurons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for row in selected_neurons:
            layer_idx = int(row.get("layer_idx", 0))
            feat_idx = int(row.get("feature_idx", row.get("neuron_idx", 0)))
            summary = self.feature_store.get_summary(layer_idx, feat_idx)
            row = dict(row)
            row["feature_summary"] = summary.to_dict()
            row["feature_label"] = format_feature_for_prompt(summary)
            row["feature_description"] = build_feature_description(summary)
            out.append(row)
        return out


class CircuitNeuronSelector:
    """High-level wrapper used by trainer selection stage."""

    def __init__(self, cfg: Dict[str, Any], probe: FrozenLLMProbe) -> None:
        self.cfg = dict(cfg)
        self.probe = probe

    def select(
        self,
        prompts: Sequence[str],
        labels: Sequence[int] | torch.Tensor,
    ) -> Dict[str, Any]:
        layer_activations = self.probe.probe_prompts(
            prompts=prompts,
            batch_size=int(self.cfg.get("selection_batch_size", 4)),
            desc="selection probe",
        )
        selection = score_neurons(
            layer_activations=layer_activations,
            labels=labels,
            top_k=int(self.cfg.get("top_k_neurons", 256)),
            activation_threshold=float(self.cfg.get("selection_activation_threshold", DEFAULT_ACTIVATION_THRESHOLD)),
            min_class_mean=float(self.cfg.get("selection_min_class_mean", DEFAULT_MIN_CLASS_MEAN)),
            min_anomaly_support_events=int(self.cfg.get("selection_min_anomaly_support_events", DEFAULT_MIN_SUPPORT_EVENTS)),
            min_anomaly_support_fraction=float(self.cfg.get("selection_min_anomaly_support_fraction", DEFAULT_MIN_SUPPORT_FRACTION)),
            min_normal_support_events=int(self.cfg.get("selection_min_normal_support_events", DEFAULT_MIN_SUPPORT_EVENTS)),
            min_normal_support_fraction=float(self.cfg.get("selection_min_normal_support_fraction", DEFAULT_MIN_SUPPORT_FRACTION)),
            min_mean_separation=float(self.cfg.get("selection_min_mean_separation", DEFAULT_MIN_MEAN_SEPARATION)),
        )
        rows = self.probe.annotate_selected_features(list(selection["selected_neurons"]))
        selection["selected_neurons"] = rows
        selection["selected_feature_nodes"] = rows
        return {"selection": selection, "layer_activations": layer_activations}


def slice_selected_features_from_activations(
    layer_activations: Mapping[int, torch.Tensor],
    selected_neurons: Sequence[Mapping[str, Any]],
) -> torch.Tensor:
    """Slice a full per-layer activation dict into [N, K] selected matrix."""

    if not selected_neurons:
        first = next(iter(layer_activations.values()), None)
        n = 0 if first is None else first.shape[0]
        return torch.empty(n, 0, dtype=torch.float32)

    n_events = next(iter(layer_activations.values())).shape[0]
    out = torch.zeros(n_events, len(selected_neurons), dtype=torch.float32)
    for col_idx, node in enumerate(selected_neurons):
        layer = int(node.get("layer_idx", 0))
        feat = int(node.get("feature_idx", node.get("neuron_idx", 0)))
        acts = layer_activations[int(layer)]
        out[:, col_idx] = acts[:, feat].float().cpu()
    return out

