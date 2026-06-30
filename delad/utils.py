"""Utilities for reproducibility, metrics, and artifact management."""

from __future__ import annotations

import json
import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, TextIO

import numpy as np
import torch

try:
    import resource
except ImportError:  # pragma: no cover
    resource = None


def repo_root() -> Path:
    """Return the repository root directory."""

    return Path(__file__).resolve().parent.parent


def resolve_path(path: str | Path, *, base: Path | None = None) -> Path:
    """Resolve a path for file I/O (relative paths are anchored to ``base`` or the repo root)."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    root = (base or repo_root()).resolve()
    return (root / candidate).resolve()


def as_relative_path(path: str | Path, *, base: Path | None = None) -> str:
    """Return a repo-relative POSIX path string for logs and JSON artifacts."""

    root = (base or repo_root()).resolve()
    resolved = resolve_path(path, base=base)
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return Path(path).as_posix()


CFG_PATH_KEYS = frozenset(
    {
        "assets_root",
        "shared_root",
        "dataset_dir",
        "train_path",
        "val_path",
        "test_path",
        "output_dir",
        "llm_path",
        "tokenizer_path",
        "transcoder_path",
        "circuit_feature_dir",
        "run_output_dir",
        "metrics_dir",
        "feature_artifacts_dir",
        "selected_neurons_path",
        "selected_features_path",
        "selected_neuron_features_path",
        "selected_features_review_path",
        "selection_feature_bank_path",
        "calc_feature_dir",
        "train_feature_json_path",
        "val_feature_json_path",
        "test_feature_json_path",
        "calc_manifest_path",
        "detector_ckpt_path",
        "prompt_snapshot_path",
        "run_manifest_path",
    }
)


def normalize_cfg_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Store path-like config values as repo-relative strings."""

    out = dict(cfg)
    for key in CFG_PATH_KEYS:
        value = out.get(key)
        if value:
            out[key] = as_relative_path(value)
    return out


def set_seed(seed: int) -> None:
    """Set all major random seeds used by the framework."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""

    directory = resolve_path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(data: Any, path: str | Path) -> None:
    """Save JSON-serializable data with formatting."""

    file_path = resolve_path(path)
    ensure_dir(file_path.parent)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    """Load a JSON artifact."""

    with resolve_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_pickle(data: Any, path: str | Path) -> None:
    """Serialize an artifact with pickle."""

    file_path = resolve_path(path)
    ensure_dir(file_path.parent)
    with file_path.open("wb") as handle:
        pickle.dump(data, handle)


def load_pickle(path: str | Path) -> Any:
    """Load a pickle artifact."""

    with resolve_path(path).open("rb") as handle:
        return pickle.load(handle)


def save_torch(data: Any, path: str | Path) -> None:
    """Save a PyTorch artifact."""

    file_path = resolve_path(path)
    ensure_dir(file_path.parent)
    torch.save(data, file_path)


def load_torch(path: str | Path) -> Any:
    """Load a PyTorch artifact."""

    return torch.load(resolve_path(path), map_location="cpu")


def pretty_print_config(cfg: Dict[str, Any]) -> None:
    """Print only the key runtime summary."""

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    configured_dataset = str(cfg.get("dataset_name", "auto")).strip().lower()
    dataset_name = configured_dataset if configured_dataset != "auto" else Path(cfg["dataset_dir"]).name
    if gpu_count == 0:
        runtime = "cpu"
        gpus_per_process = 0
    elif gpu_count == 1:
        runtime = "single-gpu"
        gpus_per_process = 1
    else:
        runtime = "multi-gpu"
        gpus_per_process = gpu_count

    print("===== Runtime Summary =====")
    print(f"method: {cfg['method']}")
    print(f"mode: {cfg['mode']}")
    print(f"model: {cfg['model_name']}")
    print(f"dataset: {dataset_name}")
    print(f"processes: 1")
    print(f"visible_gpus: {gpu_count}")
    print(f"gpus_per_process: {gpus_per_process}")
    print(f"runtime: {runtime}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"llm_path: {cfg.get('llm_path', '(n/a)')}")
    print(f"transcoder_path: {cfg.get('transcoder_path', '(n/a)')}")
    print("===========================")


def get_resource_snapshot() -> str:
    """Return a compact CPU/GPU memory snapshot for debug logging."""

    parts: List[str] = []
    if resource is not None:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_mb = rss / 1024.0
        parts.append(f"rss_mb={rss_mb:.1f}")

    if torch.cuda.is_available():
        gpu_parts = []
        for idx in range(torch.cuda.device_count()):
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
                used_gb = (total_bytes - free_bytes) / (1024**3)
                total_gb = total_bytes / (1024**3)
                gpu_parts.append(f"cuda:{idx}={used_gb:.1f}/{total_gb:.1f}GB")
            except Exception:
                gpu_parts.append(f"cuda:{idx}=unavailable")
        parts.append("gpu[" + ", ".join(gpu_parts) + "]")

    return " | ".join(parts) if parts else "resource_snapshot_unavailable"


_RUN_LOG_HANDLE: Optional[TextIO] = None


def configure_run_logging(log_path: str | Path | None) -> None:
    """Append ``log_status`` lines to ``log_path`` (UTF-8). Pass None to close."""

    global _RUN_LOG_HANDLE
    if _RUN_LOG_HANDLE is not None:
        try:
            _RUN_LOG_HANDLE.close()
        except OSError:
            pass
        _RUN_LOG_HANDLE = None
    if log_path is None:
        return
    log_file = resolve_path(log_path)
    ensure_dir(log_file.parent)
    _RUN_LOG_HANDLE = log_file.open("a", encoding="utf-8")


def log_status(message: str) -> None:
    """Print a flush-safe status line with a resource snapshot."""

    line = f"[status] {message} | {get_resource_snapshot()}\n"
    print(line.rstrip(), flush=True)
    if _RUN_LOG_HANDLE is not None:
        _RUN_LOG_HANDLE.write(line)
        try:
            _RUN_LOG_HANDLE.flush()
        except OSError:
            pass


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def compute_binary_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Sequence[float] | None = None,
) -> Dict[str, float | None]:
    """Compute common binary classification metrics."""

    true = np.asarray(list(y_true), dtype=np.int64)
    pred = np.asarray(list(y_pred), dtype=np.int64)

    tp = int(((true == 1) & (pred == 1)).sum())
    tn = int(((true == 0) & (pred == 0)).sum())
    fp = int(((true == 0) & (pred == 1)).sum())
    fn = int(((true == 1) & (pred == 0)).sum())

    accuracy = _safe_divide(tp + tn, len(true))
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    metrics: Dict[str, float | None] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "auroc": None,
    }

    if y_score is not None:
        score = np.asarray(list(y_score), dtype=np.float64)
        metrics["auroc"] = compute_auroc(true, score)

    return metrics


def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """Compute AUROC without requiring sklearn."""

    positives = y_true == 1
    negatives = y_true == 0
    num_pos = int(positives.sum())
    num_neg = int(negatives.sum())
    if num_pos == 0 or num_neg == 0:
        return None

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    rank_sum = ranks[positives].sum()
    u_value = rank_sum - (num_pos * (num_pos + 1) / 2.0)
    return float(u_value / (num_pos * num_neg))


def compute_chunk_labels_from_events(
    chunk_ids: Sequence[str],
    event_labels: Sequence[int],
    event_predictions: Sequence[int],
    event_scores: Sequence[float],
) -> Dict[str, List[float]]:
    """Aggregate event outputs into chunk-level predictions and scores."""

    grouped: Dict[str, Dict[str, List[float]]] = {}
    for chunk_id, label, prediction, score in zip(
        chunk_ids, event_labels, event_predictions, event_scores
    ):
        bucket = grouped.setdefault(
            str(chunk_id),
            {"labels": [], "predictions": [], "scores": []},
        )
        bucket["labels"].append(float(label))
        bucket["predictions"].append(float(prediction))
        bucket["scores"].append(float(score))

    chunk_true: List[int] = []
    chunk_pred: List[int] = []
    chunk_score: List[float] = []

    for values in grouped.values():
        chunk_true.append(int(max(values["labels"])))
        chunk_pred.append(int(max(values["predictions"])))
        chunk_score.append(float(max(values["scores"])))

    return {
        "chunk_true": chunk_true,
        "chunk_pred": chunk_pred,
        "chunk_score": chunk_score,
    }


def flatten_masked_tensors(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    probabilities: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, List[float]]:
    """Flatten masked event tensors into Python lists."""

    valid = mask.bool()
    return {
        "labels": labels[valid].detach().cpu().tolist(),
        "predictions": predictions[valid].detach().cpu().tolist(),
        "probabilities": probabilities[valid].detach().cpu().tolist(),
    }
