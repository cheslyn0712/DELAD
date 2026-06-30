"""Training/evaluation orchestration for circuit-tracing pipeline."""

from __future__ import annotations

import copy
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from delad.data import (
    ChunkDataset,
    EventDataset,
    FeatureChunkDataset,
    collate_chunk_batch,
    collate_feature_chunk_batch,
    infer_input_dim,
    resolve_training_paradigm,
    split_flat_features_by_chunk,
)
from delad.detector import (
    SequentialEventDetector,
    decode_event_predictions,
    masked_event_cross_entropy,
    session_aggregated_bce_loss,
)
from delad.session_aggregation import aggregate_transition_probabilities, normalize_session_aggregation_method
from delad.probe import CircuitNeuronSelector, FrozenLLMProbe, slice_selected_features_from_activations
from delad.utils import (
    as_relative_path,
    compute_binary_classification_metrics,
    compute_chunk_labels_from_events,
    ensure_dir,
    flatten_masked_tensors,
    load_json,
    log_status,
    resolve_path,
    save_json,
    save_torch,
    set_seed,
)


@dataclass
class DetectorTrialResult:
    trial_index: int
    params: Dict[str, Any]
    best_epoch: int
    best_score: float
    best_val_metrics: Dict[str, Any]
    best_state_dict: Dict[str, torch.Tensor]
    wall_seconds: float
    history: List[Dict[str, Any]]


def _selection_eligibility_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "activation_threshold": float(cfg.get("selection_activation_threshold", 1e-4)),
        "min_class_mean": float(cfg.get("selection_min_class_mean", 1e-4)),
        "min_anomaly_support_events": int(cfg.get("selection_min_anomaly_support_events", 3)),
        "min_anomaly_support_fraction": float(cfg.get("selection_min_anomaly_support_fraction", 0.02)),
        "min_normal_support_events": int(cfg.get("selection_min_normal_support_events", 3)),
        "min_normal_support_fraction": float(cfg.get("selection_min_normal_support_fraction", 0.02)),
        "min_mean_separation": float(cfg.get("selection_min_mean_separation", 1e-3)),
    }


def _collect_selection_events(dataset: EventDataset) -> Dict[str, Any]:
    prompts: List[str] = []
    labels: List[int] = []
    records: List[Dict[str, Any]] = []
    for idx in range(len(dataset)):
        row = dataset[idx]
        prompts.append(str(row["prompt"]))
        labels.append(int(row["label"]))
        records.append(
            {
                "chunk_id": str(row.get("chunk_id", "")),
                "chunk_index": int(row.get("chunk_index", -1)),
                "event_index": int(row.get("event_index", 0)),
                "label": int(row.get("label", 0)),
                "prompt": str(row.get("prompt", "")),
                "target_log": str(row.get("target_log", "")),
                "target_mode": str(row.get("target_mode", "")),
                "target_pos": int(row.get("target_pos", 0)),
                "dataset_name": str(row.get("dataset_name", "")),
                "num_prefix_logs": len(row.get("prefix_logs", [])),
            }
        )
    return {"prompts": prompts, "labels": labels, "records": records}


def _trim_selection_events_to_single_batch(collected: Dict[str, Any], max_events: int) -> Dict[str, Any]:
    """Keep at most one probe batch for smoke runs while preserving class coverage when possible."""

    max_keep = max(1, int(max_events))
    prompts = list(collected.get("prompts", []))
    labels = [int(x) for x in collected.get("labels", [])]
    records = list(collected.get("records", []))
    total = len(prompts)
    if total <= max_keep:
        return collected

    normal_idx = [i for i, y in enumerate(labels) if y == 0]
    anomaly_idx = [i for i, y in enumerate(labels) if y == 1]
    chosen: List[int] = []
    if normal_idx and anomaly_idx and max_keep >= 2:
        chosen.extend([normal_idx[0], anomaly_idx[0]])
    for idx in range(total):
        if idx in chosen:
            continue
        chosen.append(idx)
        if len(chosen) >= max_keep:
            break
    chosen = sorted(set(chosen))[:max_keep]
    return {
        "prompts": [prompts[i] for i in chosen],
        "labels": [labels[i] for i in chosen],
        "records": [records[i] for i in chosen],
    }


def _validate_selected_neuron_eligibility(selection: Mapping[str, Any], top_k: int) -> None:
    selected = selection.get("selected_neurons") or []
    if not selected:
        raise ValueError("No eligible features selected; check eligibility thresholds.")
    if len(selected) < int(top_k):
        log_status(
            f"WARN: selected features fewer than top_k ({len(selected)} < {int(top_k)}). "
            "Proceeding with available eligible features."
        )
    for row in selected:
        if not bool(row.get("eligible", False)):
            raise ValueError("Selected feature contains eligible=False row; contract violation.")


def _build_selected_features_export(
    selection: Mapping[str, Any],
    layer_activations: Mapping[int, torch.Tensor],
    min_class_mean: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    selected = selection.get("selected_neurons") or []
    for node in selected:
        layer_idx = int(node["layer_idx"])
        feature_idx = int(node.get("feature_idx", node.get("neuron_idx", 0)))
        acts = layer_activations[layer_idx][:, feature_idx].detach().cpu().float()
        rows.append(
            {
                "layer_idx": layer_idx,
                "feature_idx": feature_idx,
                "label": str(node.get("feature_label", f"L{layer_idx}#{feature_idx}")),
                "score": float(node.get("score", 0.0)),
                "mean_normal": float(node.get("mean_norm", 0.0)),
                "mean_anomaly": float(node.get("mean_anom", 0.0)),
                "cosine_similarity": float(node.get("cosine_similarity", 0.0)),
                "cosine_difference": float(node.get("cosine_difference", 0.0)),
                "separation": float(node.get("separation", 0.0)),
                "magnitude": float(node.get("magnitude", 0.0)),
                "support_normal": float(node.get("support_norm", 0.0)),
                "support_anomaly": float(node.get("support_anom", 0.0)),
                "count_normal": int(node.get("count_norm", 0)),
                "count_anomaly": int(node.get("count_anom", 0)),
                "max_anomaly_activation": float(node.get("max_anomaly_activation", 0.0)),
                "eligible": bool(node.get("eligible", True)),
                "min_class_mean": float(min_class_mean),
                "activations": acts.tolist(),
            }
        )
    return rows


def _build_selected_features_review_export(selection: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Human-auditable review rows; additive artifact for manual QA."""

    rows: List[Dict[str, Any]] = []
    for node in selection.get("selected_neurons") or []:
        summary = node.get("feature_summary") or {}
        row = {
            "rank": int(node.get("rank", -1)),
            "layer_idx": int(node.get("layer_idx", 0)),
            "feature_idx": int(node.get("feature_idx", node.get("neuron_idx", 0))),
            "feature_label": str(node.get("feature_label", "")),
            "feature_description": str(node.get("feature_description", "")),
            "score": float(node.get("score", 0.0)),
            "separation": float(node.get("separation", 0.0)),
            "eligible": bool(node.get("eligible", True)),
            "mean_normal": float(node.get("mean_norm", 0.0)),
            "mean_anomaly": float(node.get("mean_anom", 0.0)),
            "count_normal": int(node.get("count_norm", 0)),
            "count_anomaly": int(node.get("count_anom", 0)),
            "support_normal": float(node.get("support_norm", 0.0)),
            "support_anomaly": float(node.get("support_anom", 0.0)),
        }
        for key in ("peak_tokens", "top_logits", "top_contexts"):
            if key in summary:
                row[key] = summary.get(key)
        rows.append(row)
    return rows


def _save_selection_feature_bank(
    cfg: Dict[str, Any],
    records: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
    layer_activations: Mapping[int, torch.Tensor],
) -> None:
    bank_path = Path(cfg["selection_feature_bank_path"])
    bank_dir = ensure_dir(bank_path.parent / "selection_feature_bank")
    event_rows: List[Dict[str, Any]] = []
    n_norm = 0
    n_anom = 0
    for idx, (row, lab) in enumerate(zip(records, labels)):
        lab_i = int(lab)
        if lab_i == 0:
            n_norm += 1
        else:
            n_anom += 1
        event_rows.append(
            {
                "event_index_in_bank": int(idx),
                "chunk_id": str(row.get("chunk_id", "")),
                "chunk_index": int(row.get("chunk_index", -1)),
                "event_index": int(row.get("event_index", 0)),
                "dataset_name": str(row.get("dataset_name", "")),
                "label": lab_i,
                "target_mode": str(row.get("target_mode", "")),
                "target_pos": int(row.get("target_pos", 0)),
                "target_log": str(row.get("target_log", "")),
                "prompt": str(row.get("prompt", "")),
                "num_prefix_logs": int(row.get("num_prefix_logs", 0)),
            }
        )

    layer_rows: List[Dict[str, Any]] = []
    for layer_idx in sorted(layer_activations.keys()):
        acts = layer_activations[layer_idx].detach().cpu().numpy().astype("float32")
        npz_path = bank_dir / f"layer_{int(layer_idx)}.npz"
        import numpy as np

        np.savez_compressed(npz_path, activations=acts)
        layer_rows.append(
            {
                "layer_idx": int(layer_idx),
                "num_features": int(acts.shape[1]),
                "num_events": int(acts.shape[0]),
                "activations_npz": str(npz_path.relative_to(bank_path.parent)),
                "score_stats_keys": ["scores", "eligible", "separation"],
            }
        )

    payload = {
        "artifact_purpose": "selection_feature_bank",
        "model_name": str(cfg.get("model_name", "")),
        "train_split": str(cfg.get("train_path", "")),
        "selection_chunk_config": {
            "test_normal_chunks": int(cfg.get("test_normal_chunks", 100)),
            "test_anomaly_chunks": int(cfg.get("test_anomaly_chunks", 100)),
        },
        "summary": {
            "num_events": len(event_rows),
            "num_normal_events": int(n_norm),
            "num_anomaly_events": int(n_anom),
            "num_layers": len(layer_rows),
            "capture_layers": [int(x) for x in cfg.get("capture_layers", [])],
        },
        "events": event_rows,
        "layers": layer_rows,
    }
    save_json(payload, bank_path)


def _build_probe_fingerprint(cfg: Dict[str, Any], split_path: str, source: str) -> Dict[str, Any]:
    return {
        "model_name": str(cfg.get("model_name", "")),
        "llm_path": str(cfg.get("llm_path", "")),
        "transcoder_path": str(cfg.get("transcoder_path", "")),
        "tokenizer_path": str(cfg.get("tokenizer_path", "")),
        "split_path": as_relative_path(split_path),
        "source": str(source),
        "max_length": int(cfg.get("max_length", 512)),
        "seed": int(cfg.get("seed", 42)),
        "event_feature_position": str(cfg.get("event_feature_position", "last_prefix_token")),
        "llm_dtype": "float16" if torch.cuda.is_available() else "float32",
        "capture_layers": [int(x) for x in cfg.get("capture_layers", [])],
    }


def _extract_split_features(
    *,
    cfg: Dict[str, Any],
    probe: FrozenLLMProbe,
    split_name: str,
    split_path: str,
    selected_neurons: Sequence[Mapping[str, Any]],
    max_normal_chunks: Optional[int],
    max_anomaly_chunks: Optional[int],
    chunk_index_whitelist: Optional[Sequence[int]] = None,
    chunk_id_whitelist: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    def _progress_bar(current: int, total: int, width: int = 14) -> str:
        total_safe = max(1, int(total))
        curr = max(0, min(int(current), total_safe))
        filled = int(round((curr / float(total_safe)) * width))
        return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"

    progress_initialized = False
    last_total_line = ""
    last_micro_line = ""

    def _render_progress(total_line: str, micro_line: str) -> None:
        nonlocal progress_initialized, last_total_line, last_micro_line
        total = str(total_line)
        micro = str(micro_line)
        if not progress_initialized:
            sys.stdout.write(total + "\n" + micro)
            sys.stdout.flush()
            progress_initialized = True
        else:
            # Move cursor up two lines and redraw both lines in place.
            sys.stdout.write("\x1b[2F")
            sys.stdout.write("\x1b[2K" + total + "\n")
            sys.stdout.write("\x1b[2K" + micro)
            sys.stdout.flush()
        last_total_line = total
        last_micro_line = micro

    def _finish_inline() -> None:
        nonlocal progress_initialized
        if progress_initialized:
            sys.stdout.write("\n")
            sys.stdout.flush()
            progress_initialized = False

    calc_workers = int(cfg.get("calc_num_workers", 0) or 0)
    dataset = ChunkDataset(
        path=split_path,
        cfg=cfg,
        max_normal_chunks=max_normal_chunks,
        max_anomaly_chunks=max_anomaly_chunks,
        chunk_index_whitelist=chunk_index_whitelist,
        chunk_id_whitelist=chunk_id_whitelist,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("detector_batch_size", 16)),
        shuffle=False,
        num_workers=max(0, calc_workers),
        collate_fn=collate_chunk_batch,
    )
    sequences: List[Dict[str, Any]] = []
    prompt_samples: List[str] = []
    calc_timing_rows: List[Dict[str, Any]] = []
    total_batches = len(loader)
    started = time.perf_counter()
    last_log = started
    log_every_batches = max(1, int(cfg.get("feature_extract_log_every_batches", 5)))
    log_every_seconds = max(1, int(cfg.get("feature_extract_log_every_seconds", 60)))
    log_every_micro_batches = max(1, int(cfg.get("feature_extract_log_every_micro_batches", 1)))
    silence_warn_seconds = max(log_every_seconds, int(cfg.get("stage_silence_warning_seconds", 300)))
    processed_chunks = 0
    processed_events = 0
    log_status(
        f"[{split_name}] feature extraction start "
        f"chunks={len(dataset)} batches={total_batches} selected_features={len(selected_neurons)} "
        f"batch_size={int(cfg.get('detector_batch_size', 16))} workers={max(0, calc_workers)}"
    )
    for batch_idx, batch in enumerate(loader, start=1):
        if len(prompt_samples) < 3:
            remain = 3 - len(prompt_samples)
            prompt_samples.extend([str(p) for p in batch["flat_prompts"][:remain]])

        def _on_micro_progress(payload: Dict[str, Any]) -> None:
            micro_idx = int(payload.get("micro_batch_index", 0))
            micro_total = int(payload.get("micro_batch_total", 0))
            if micro_idx <= 0 or micro_total <= 0:
                return
            if not (micro_idx == 1 or micro_idx == micro_total or micro_idx % log_every_micro_batches == 0):
                return
            outer_bar = _progress_bar(batch_idx, total_batches, width=12)
            micro_bar = _progress_bar(micro_idx, micro_total, width=12)
            elapsed = max(1e-6, time.perf_counter() - started)
            bps = float(batch_idx - 1 + (micro_idx / float(max(1, micro_total)))) / elapsed
            eta = (float(total_batches) - float(batch_idx - 1 + (micro_idx / float(max(1, micro_total))))) / bps if bps > 0 else float("inf")
            total_pct = 100.0 * float(batch_idx) / float(max(1, total_batches))
            total_line = (
                f"[status] [{split_name}] total B{outer_bar} {batch_idx}/{total_batches} "
                f"{total_pct:5.1f}% chunks={processed_chunks} events={processed_events} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s"
            )
            micro_line = (
                f"[status] [{split_name}] micro M{micro_bar} {micro_idx}/{micro_total} "
                f"events={int(payload.get('processed_in_outer_batch', 0))}/{int(payload.get('total_in_outer_batch', 0))} "
                f"chunk={str(payload.get('current_chunk_id', ''))} "
                f"event={int(payload.get('current_event_index', -1))}"
            )
            _render_progress(total_line, micro_line)

        flat_features = probe.extract_selected_features(
            prompts=batch["flat_prompts"],
            selected_neurons=selected_neurons,
            batch_size=int(cfg.get("feature_batch_size", cfg.get("selection_batch_size", 8))),
            split_name=split_name,
            outer_batch_index=batch_idx,
            outer_batch_total=total_batches,
            event_indices=batch.get("flat_event_indices"),
            event_chunk_ids=batch.get("flat_chunk_ids"),
            log_every_micro_batches=0,
            progress_callback=_on_micro_progress,
            timing_collector=calc_timing_rows,
        )
        seq_rows = split_flat_features_by_chunk(
            flat_features=flat_features,
            flat_labels=batch["flat_labels"],
            flat_supervision_mask=batch["flat_supervision_mask"],
            chunk_ids=batch["chunk_ids"],
            sequence_lengths=batch["sequence_lengths"],
            chunk_labels=batch.get("chunk_labels"),
        )
        sequences.extend(seq_rows)
        processed_chunks += len(batch["chunk_ids"])
        processed_events += len(batch["flat_prompts"])
        now = time.perf_counter()
        current_event_idx = int(batch["flat_event_indices"][-1]) if batch.get("flat_event_indices") else -1
        current_chunk = str(batch["chunk_ids"][-1]) if batch.get("chunk_ids") else ""
        should_log = (
            batch_idx == 1
            or batch_idx == total_batches
            or (batch_idx % log_every_batches == 0)
            or ((now - last_log) >= log_every_seconds)
        )
        if should_log:
            elapsed = max(1e-6, now - started)
            bps = float(batch_idx) / elapsed
            eta = (float(total_batches - batch_idx) / bps) if bps > 0.0 else float("inf")
            pct = 100.0 * float(batch_idx) / float(max(1, total_batches))
            bar = _progress_bar(batch_idx, total_batches, width=20)
            total_line = (
                f"[status] [{split_name}] total B{bar} {pct:5.1f}% "
                f"batch={batch_idx}/{total_batches} chunks={processed_chunks} events={processed_events} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s"
            )
            micro_line = (
                f"[status] [{split_name}] micro (waiting) chunk={current_chunk} event={current_event_idx}"
            )
            _render_progress(total_line, micro_line)
            last_log = now
        elif (now - last_log) >= silence_warn_seconds:
            total_line = (
                f"[status] [{split_name}] total (heartbeat) "
                f"batch={batch_idx}/{total_batches} chunks={processed_chunks} events={processed_events}"
            )
            micro_line = (
                f"[status] [{split_name}] micro (heartbeat) chunk={current_chunk} event={current_event_idx}"
            )
            _render_progress(total_line, micro_line)
            last_log = now
    wall_s = time.perf_counter() - started
    _finish_inline()
    total_calc_events = int(sum(int(r.get("num_events", 0)) for r in calc_timing_rows))
    total_calc_seconds = float(sum(float(r.get("elapsed_seconds", 0.0)) for r in calc_timing_rows))
    avg_seconds_per_event = float(total_calc_seconds / max(1, total_calc_events))
    max_seconds_per_event = float(
        max((float(r.get("seconds_per_event", 0.0)) for r in calc_timing_rows), default=0.0)
    )
    weighted_e2 = float(
        sum(
            (float(r.get("seconds_per_event", 0.0)) ** 2) * int(r.get("num_events", 0))
            for r in calc_timing_rows
        )
        / max(1, total_calc_events)
    )
    variance_seconds_per_event = float(max(0.0, weighted_e2 - (avg_seconds_per_event**2)))
    log_status(
        f"[{split_name}] feature extraction done chunks={len(sequences)} "
        f"processed_events={processed_events} wall_s={wall_s:.2f}"
    )

    schema = []
    for idx, node in enumerate(selected_neurons):
        schema.append(
            {
                "feature_index": int(idx),
                "rank": int(node.get("rank", idx)),
                "layer_idx": int(node.get("layer_idx", 0)),
                "feature_idx_within_layer": int(node.get("feature_idx", node.get("neuron_idx", 0))),
                "score": float(node.get("score", 0.0)),
                "feature_label": str(node.get("feature_label", "")),
                "feature_description": str(node.get("feature_description", "")),
            }
        )
    return {
        "split_name": split_name,
        "source_path": as_relative_path(split_path),
        "event_definition": "single_step_log_i_to_log_i_plus_1",
        "feature_schema": schema,
        "probe_fingerprint": _build_probe_fingerprint(cfg, split_path, f"calc_split:{split_name}"),
        "num_chunks": len(sequences),
        "num_selected_feature_nodes": len(selected_neurons),
        "prompt_samples": prompt_samples,
        "feature_extraction_timing": {
            "timing_scope": "calc_inference_only",
            "timing_unit": "seconds_per_event",
            "num_micro_batches": int(len(calc_timing_rows)),
            "num_events_timed": int(total_calc_events),
            "total_calc_seconds": float(total_calc_seconds),
            "avg_seconds_per_event": float(avg_seconds_per_event),
            "max_seconds_per_event": float(max_seconds_per_event),
            "var_seconds_per_event": float(variance_seconds_per_event),
        },
        "sampling_config": {
            "max_normal_chunks": max_normal_chunks,
            "max_anomaly_chunks": max_anomaly_chunks,
        },
        "chunks": [
            {
                "chunk_id": str(row["chunk_id"]),
                "chunk_label": int(row.get("chunk_label", 0)),
                "events": [
                    {
                        "event_index": int(i),
                        "label": int(lbl),
                        "supervise": bool(sup),
                        "features": feat.float().tolist(),
                    }
                    for i, (feat, lbl, sup) in enumerate(
                        zip(row["features"], row["labels"], row["supervision_mask"])
                    )
                ],
            }
            for row in sequences
        ],
    }


def _save_prompt_snapshot(
    cfg: Dict[str, Any],
    *,
    selection_records: Sequence[Mapping[str, Any]],
    calc_prompt_examples: Mapping[str, Sequence[str]],
) -> None:
    path = Path(cfg["prompt_snapshot_path"])
    rows = []
    for row in list(selection_records)[:3]:
        rows.append(
            {
                "chunk_id": str(row.get("chunk_id", "")),
                "target_mode": str(row.get("target_mode", "")),
                "target_pos": int(row.get("target_pos", 0)),
                "label": int(row.get("label", 0)),
                "prompt": str(row.get("prompt", "")),
            }
        )
    payload = {
        "model_name": str(cfg.get("model_name", "")),
        "prompt_type": str(cfg.get("selection_prompt_extract_option", cfg.get("prompt_extract_option", "A"))),
        "instruction_length": str(cfg.get("circuit_prompt_instruction_length", "medium")),
        "calc_input_mode": str(cfg.get("circuit_calc_input_mode", "single_message")),
        "rnn_type": str(cfg.get("detector_recurrent_type", "lstm")),
        "selection_prompt_examples": rows,
        "calc_prompt_examples": {k: list(v) for k, v in calc_prompt_examples.items()},
    }
    save_json(payload, path)


def _validate_feature_artifact_schema(feature_blob: Mapping[str, Any], split_name: str) -> None:
    if "feature_schema" not in feature_blob or "chunks" not in feature_blob:
        raise ValueError(f"{split_name} feature artifact missing required fields.")
    k = len(feature_blob.get("feature_schema", []))
    for chunk in feature_blob.get("chunks", []):
        for event in chunk.get("events", []):
            if len(event.get("features", [])) != k:
                raise ValueError(f"{split_name} feature dim mismatch; expected {k}.")


def _val_sampling_limits(cfg: Dict[str, Any]) -> Tuple[int, int]:
    normal = cfg.get("val_normal_chunks")
    anomaly = cfg.get("val_anomaly_chunks")
    if normal is None:
        normal = cfg.get("test_val_normal_chunks", cfg.get("test_test_normal_chunks", 50))
    if anomaly is None:
        anomaly = cfg.get("test_val_anomaly_chunks", cfg.get("test_test_anomaly_chunks", 50))
    return int(normal), int(anomaly)


def _test_sampling_limits(cfg: Dict[str, Any]) -> Tuple[int, int]:
    return int(cfg.get("test_test_normal_chunks", 50)), int(cfg.get("test_test_anomaly_chunks", 50))


def _validate_calc_artifacts(
    train_blob: Mapping[str, Any],
    val_blob: Mapping[str, Any],
    test_blob: Mapping[str, Any],
) -> None:
    for name, blob in (("train", train_blob), ("val", val_blob), ("test", test_blob)):
        _validate_feature_artifact_schema(blob, name)
    train_fp = train_blob.get("probe_fingerprint", {})
    for name, blob in (("val", val_blob), ("test", test_blob)):
        fp = blob.get("probe_fingerprint", {})
        keys = ("model_name", "llm_path", "transcoder_path", "tokenizer_path", "max_length")
        for key in keys:
            if train_fp.get(key) != fp.get(key):
                raise ValueError(f"Probe fingerprint mismatch between train/{name} for key '{key}'.")


def _resolve_train_split_whitelists(cfg: Dict[str, Any], selected_blob: Mapping[str, Any]) -> Dict[str, List[Any]]:
    """Recover the exact selection train subset for standalone calc runs."""

    idx_from_selection = [int(x) for x in selected_blob.get("selection_chunk_indices", []) if int(x) >= 0]
    ids_from_selection = [str(x) for x in selected_blob.get("selected_chunk_ids", []) if str(x)]
    idx_from_bank: List[int] = []
    ids_from_bank: List[str] = []
    bank_path = Path(str(cfg.get("selection_feature_bank_path", "")))
    if bank_path.is_file():
        bank = load_json(bank_path)
        for row in bank.get("events", []):
            ci = int(row.get("chunk_index", -1))
            if ci >= 0:
                idx_from_bank.append(ci)
            cid = str(row.get("chunk_id", ""))
            if cid:
                ids_from_bank.append(cid)
    chunk_indices = sorted(set(idx_from_selection or idx_from_bank))
    chunk_ids = sorted(set(ids_from_selection or ids_from_bank))
    return {"chunk_indices": chunk_indices, "chunk_ids": chunk_ids}


def run_neuron_selection(cfg: Dict[str, Any]) -> Dict[str, Any]:
    stage_started = time.perf_counter()
    log_status(
        f"[stage] run_neuron_selection start model={cfg.get('model_name')} "
        f"prompt={cfg.get('selection_prompt_extract_option', cfg.get('prompt_extract_option', 'A'))} "
        f"len={cfg.get('circuit_prompt_instruction_length', 'medium')}"
    )
    set_seed(int(cfg.get("seed", 42)))
    dataset = EventDataset(path=cfg["train_path"], cfg=cfg)
    collected = _collect_selection_events(dataset)
    if bool(cfg.get("selection_single_batch_smoke", False)):
        target_events = int(max(1, cfg.get("selection_batch_size", 2)))
        trimmed = _trim_selection_events_to_single_batch(collected, max_events=target_events)
        if len(trimmed["prompts"]) != len(collected["prompts"]):
            log_status(
                f"[selection-smoke] trimmed selection events "
                f"{len(collected['prompts'])} -> {len(trimmed['prompts'])} "
                f"(batch_size={target_events})"
            )
        collected = trimmed
    probe = FrozenLLMProbe(cfg)
    selector = CircuitNeuronSelector(cfg=cfg, probe=probe)
    selected = selector.select(prompts=collected["prompts"], labels=collected["labels"])
    selection = selected["selection"]
    layer_activations = selected["layer_activations"]
    _validate_selected_neuron_eligibility(selection=selection, top_k=int(cfg.get("top_k_neurons", 256)))

    selection.update(
        {
            "capture_layers": [int(x) for x in cfg.get("capture_layers", [])],
            "event_feature_position": str(cfg.get("event_feature_position", "last_prefix_token")),
            "model_name": str(cfg.get("model_name", "")),
            "train_split": str(cfg.get("train_path", "")),
            "selection_chunk_indices": [int(x) for x in getattr(dataset, "selected_chunk_indices", [])],
            "selected_chunk_ids": sorted({str(r.get("chunk_id", "")) for r in collected["records"]}),
            "selection_chunk_config": {
                "test_normal_chunks": int(cfg.get("test_normal_chunks", 100)),
                "test_anomaly_chunks": int(cfg.get("test_anomaly_chunks", 100)),
            },
            "selection_summary": {
                "num_events": len(collected["records"]),
                "num_normal_events": int(sum(1 for x in collected["labels"] if int(x) == 0)),
                "num_anomaly_events": int(sum(1 for x in collected["labels"] if int(x) == 1)),
            },
            "feature_bank_path": str(cfg["selection_feature_bank_path"]),
        }
    )

    selected_features_export = _build_selected_features_export(
        selection=selection,
        layer_activations=layer_activations,
        min_class_mean=float(cfg.get("selection_min_class_mean", 1e-4)),
    )
    selected_features_review = _build_selected_features_review_export(selection)
    selected_neuron_features = {
        "selected_chunk_ids": sorted({str(r.get("chunk_id", "")) for r in collected["records"]}),
        "events": [
            {
                "chunk_id": str(rec.get("chunk_id", "")),
                "event_index": int(rec.get("event_index", 0)),
                "label": int(rec.get("label", 0)),
                "selected_feature_activations": row.tolist(),
            }
            for rec, row in zip(collected["records"], slice_selected_features_from_activations(layer_activations, selection["selected_neurons"]))
        ],
    }

    _save_selection_feature_bank(
        cfg=cfg,
        records=collected["records"],
        labels=collected["labels"],
        layer_activations=layer_activations,
    )
    save_json(selection, cfg["selected_neurons_path"])
    save_json(selected_features_export, cfg["selected_features_path"])
    review_path = cfg.get("selected_features_review_path")
    if not review_path:
        review_path = str(Path(cfg["selected_features_path"]).with_name("selected_features_review.json"))
    save_json(selected_features_review, review_path)
    save_json(selected_neuron_features, cfg["selected_neuron_features_path"])

    train_blob = _extract_split_features(
        cfg=cfg,
        probe=probe,
        split_name="train",
        split_path=cfg["train_path"],
        selected_neurons=selection["selected_neurons"],
        max_normal_chunks=None,
        max_anomaly_chunks=None,
        chunk_index_whitelist=list(dataset.selected_chunk_indices),
    )
    save_json(train_blob, cfg["train_feature_json_path"])
    _validate_feature_artifact_schema(train_blob, "train")

    _save_prompt_snapshot(
        cfg=cfg,
        selection_records=collected["records"],
        calc_prompt_examples={"train": train_blob.get("prompt_samples", [])},
    )
    wall_s = time.perf_counter() - stage_started
    log_status(f"[stage] run_neuron_selection done wall_s={wall_s:.2f}")
    return {
        "selection": selection,
        "train_feature_json_path": str(cfg["train_feature_json_path"]),
        "selection_feature_bank_path": str(cfg["selection_feature_bank_path"]),
    }


def run_feature_calculation(cfg: Dict[str, Any]) -> Dict[str, Any]:
    stage_started = time.perf_counter()
    log_status(
        f"[stage] run_feature_calculation start model={cfg.get('model_name')} "
        f"prompt={cfg.get('calc_prompt_extract_option', cfg.get('prompt_extract_option', 'A'))} "
        f"len={cfg.get('circuit_prompt_instruction_length', 'medium')} calc={cfg.get('circuit_calc_input_mode', 'single_message')}"
    )
    set_seed(int(cfg.get("seed", 42)))
    selected_blob = load_json(cfg["selected_neurons_path"])
    selected_neurons = selected_blob.get("selected_neurons") or selected_blob.get("selected_feature_nodes") or []
    if not selected_neurons:
        raise ValueError("selected_neurons artifact is empty.")
    probe = FrozenLLMProbe(cfg)

    # Always rebuild train calc features from the exact selection subset.
    # This keeps standalone calc_features runs aligned with selection whitelist.
    whitelist = _resolve_train_split_whitelists(cfg, selected_blob)
    train_blob = _extract_split_features(
        cfg=cfg,
        probe=probe,
        split_name="train",
        split_path=cfg["train_path"],
        selected_neurons=selected_neurons,
        max_normal_chunks=None if whitelist["chunk_indices"] or whitelist["chunk_ids"] else int(cfg.get("test_normal_chunks", 100)),
        max_anomaly_chunks=None if whitelist["chunk_indices"] or whitelist["chunk_ids"] else int(cfg.get("test_anomaly_chunks", 100)),
        chunk_index_whitelist=whitelist["chunk_indices"] or None,
        chunk_id_whitelist=whitelist["chunk_ids"] or None,
    )
    save_json(train_blob, cfg["train_feature_json_path"])

    val_normal, val_anomaly = _val_sampling_limits(cfg)
    val_blob = _extract_split_features(
        cfg=cfg,
        probe=probe,
        split_name="val",
        split_path=cfg["val_path"],
        selected_neurons=selected_neurons,
        max_normal_chunks=val_normal,
        max_anomaly_chunks=val_anomaly,
    )
    save_json(val_blob, cfg["val_feature_json_path"])

    test_normal, test_anomaly = _test_sampling_limits(cfg)
    test_blob = _extract_split_features(
        cfg=cfg,
        probe=probe,
        split_name="test",
        split_path=cfg["test_path"],
        selected_neurons=selected_neurons,
        max_normal_chunks=test_normal,
        max_anomaly_chunks=test_anomaly,
    )
    save_json(test_blob, cfg["test_feature_json_path"])

    _validate_calc_artifacts(train_blob, val_blob, test_blob)

    manifest = {
        "event_definition": "single_step_log_i_to_log_i_plus_1",
        "mode": "full_pipeline",
        "prompt_extract_option": str(cfg.get("prompt_extract_option", "A")),
        "selection_prompt_extract_option": str(cfg.get("selection_prompt_extract_option", cfg.get("prompt_extract_option", "A"))),
        "calc_prompt_extract_option": str(cfg.get("calc_prompt_extract_option", cfg.get("prompt_extract_option", "A"))),
        "circuit_prompt_instruction_length": str(cfg.get("circuit_prompt_instruction_length", "medium")),
        "circuit_calc_input_mode": str(cfg.get("circuit_calc_input_mode", "single_message")),
        "selected_feature_nodes_path": "artifacts/selected_neurons.json",
        "train_feature_json_path": "calc_features/train.json",
        "val_feature_json_path": "calc_features/val.json",
        "test_feature_json_path": "calc_features/test.json",
        "selection_feature_bank_path": "artifacts/selection_feature_bank.json",
        "num_selected_feature_nodes": len(selected_neurons),
        "sampling_plan": {
            "train": {
                "source": "selection whitelist",
            },
            "val": {
                "max_normal_chunks": val_normal,
                "max_anomaly_chunks": val_anomaly,
            },
            "test": {
                "max_normal_chunks": test_normal,
                "max_anomaly_chunks": test_anomaly,
            },
        },
        "split_counts": {
            "train": int(train_blob.get("num_chunks", 0)),
            "val": int(val_blob.get("num_chunks", 0)),
            "test": int(test_blob.get("num_chunks", 0)),
        },
        "probe_fingerprint_by_split": {
            "train": train_blob.get("probe_fingerprint", {}),
            "val": val_blob.get("probe_fingerprint", {}),
            "test": test_blob.get("probe_fingerprint", {}),
        },
    }
    save_json(manifest, cfg["calc_manifest_path"])
    _save_prompt_snapshot(
        cfg=cfg,
        selection_records=load_json(cfg["selection_feature_bank_path"]).get("events", []),
        calc_prompt_examples={
            "train": train_blob.get("prompt_samples", []),
            "val": val_blob.get("prompt_samples", []),
            "test": test_blob.get("prompt_samples", []),
        },
    )
    wall_s = time.perf_counter() - stage_started
    log_status(
        f"[stage] run_feature_calculation done train_chunks={int(train_blob.get('num_chunks', 0))} "
        f"val_chunks={int(val_blob.get('num_chunks', 0))} "
        f"test_chunks={int(test_blob.get('num_chunks', 0))} wall_s={wall_s:.2f}"
    )
    return {
        "train_blob": train_blob,
        "val_blob": val_blob,
        "test_blob": test_blob,
        "calc_manifest_path": str(cfg["calc_manifest_path"]),
    }


def _infer_chunk_label_from_events(chunk: Mapping[str, Any]) -> int:
    """Fallback chunk label when legacy feature JSON lacks chunk_label."""

    if "chunk_label" in chunk:
        return int(chunk["chunk_label"])
    events = chunk.get("events", [])
    if not events:
        return 0
    supervised = [e for e in events if e.get("supervise")]
    if supervised:
        return int(max(int(e.get("label", 0)) for e in supervised))
    labels = [int(e.get("label", 0)) for e in events]
    return int(max(labels))


def _load_feature_sequences(path: str) -> List[Dict[str, Any]]:
    blob = load_json(path)
    seqs: List[Dict[str, Any]] = []
    for chunk in blob.get("chunks", []):
        events = chunk.get("events", [])
        features = torch.tensor([e.get("features", []) for e in events], dtype=torch.float32)
        labels = torch.tensor([int(e.get("label", 0)) for e in events], dtype=torch.long)
        supervision_mask = torch.tensor([bool(e.get("supervise", True)) for e in events], dtype=torch.bool)
        chunk_label = _infer_chunk_label_from_events(chunk)
        seqs.append(
            {
                "chunk_id": str(chunk.get("chunk_id", "")),
                "features": features,
                "labels": labels,
                "supervision_mask": supervision_mask,
                "chunk_label": chunk_label,
            }
        )
    return seqs


def _training_paradigm(cfg: Dict[str, Any]) -> str:
    return resolve_training_paradigm(cfg)


def _session_aggregation_method(cfg: Dict[str, Any]) -> str:
    return normalize_session_aggregation_method(str(cfg.get("session_aggregation_method", "max")))


def _session_aggregation_topk(cfg: Dict[str, Any]) -> int:
    return int(cfg.get("session_aggregation_topk", 3))


def _early_stopping_score(cfg: Dict[str, Any], metrics: Mapping[str, Any]) -> float:
    if _training_paradigm(cfg) == "session_aggregate":
        return float(metrics.get("chunk_metrics", {}).get("f1") or 0.0)
    return float(metrics.get("event_metrics", {}).get("f1") or 0.0)


def _compute_detector_loss(
    *,
    cfg: Dict[str, Any],
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    supervision_mask: torch.Tensor,
    chunk_labels: torch.Tensor,
    pos_weight: float,
    label_smoothing: float,
) -> torch.Tensor:
    if _training_paradigm(cfg) == "session_aggregate":
        return session_aggregated_bce_loss(
            logits=logits,
            chunk_labels=chunk_labels,
            mask=mask,
            method=_session_aggregation_method(cfg),
            topk=_session_aggregation_topk(cfg),
            pos_weight=pos_weight,
            label_smoothing=label_smoothing,
        )
    return masked_event_cross_entropy(
        logits=logits,
        labels=labels,
        mask=(mask & supervision_mask),
        pos_weight=pos_weight,
        label_smoothing=label_smoothing,
    )


def _training_metadata(cfg: Dict[str, Any]) -> Dict[str, Any]:
    meta = {"training_paradigm": _training_paradigm(cfg)}
    if _training_paradigm(cfg) == "session_aggregate":
        meta["session_aggregation_method"] = _session_aggregation_method(cfg)
        meta["session_aggregation_topk"] = _session_aggregation_topk(cfg)
    return meta


def _evaluate_detector(
    model: SequentialEventDetector,
    loader: DataLoader,
    *,
    cfg: Dict[str, Any],
    pos_weight: float,
    label_smoothing: float,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    paradigm = _training_paradigm(cfg)
    agg_method = _session_aggregation_method(cfg)
    agg_topk = _session_aggregation_topk(cfg)
    losses: List[float] = []
    flat_labels: List[int] = []
    flat_preds: List[int] = []
    flat_probs: List[float] = []
    flat_chunk_ids: List[str] = []
    chunk_true: List[int] = []
    chunk_pred: List[int] = []
    chunk_score: List[float] = []

    with torch.no_grad():
        for batch in loader:
            feats = batch["features"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)
            sup = batch["supervision_mask"].to(device)
            chunk_labels_batch = batch["chunk_labels"].to(device)
            logits = model(feats)
            loss = _compute_detector_loss(
                cfg=cfg,
                logits=logits,
                labels=labels,
                mask=mask,
                supervision_mask=sup,
                chunk_labels=chunk_labels_batch,
                pos_weight=pos_weight,
                label_smoothing=label_smoothing,
            )
            losses.append(float(loss.item()))
            dec = decode_event_predictions(logits)
            probs = dec["probabilities"]

            if paradigm == "session_aggregate":
                scores = aggregate_transition_probabilities(probs, mask, agg_method, topk=agg_topk)
                preds = (scores >= 0.5).long()
                chunk_true.extend(int(x) for x in chunk_labels_batch.detach().cpu().tolist())
                chunk_pred.extend(int(x) for x in preds.detach().cpu().tolist())
                chunk_score.extend(float(x) for x in scores.detach().cpu().tolist())
                for row_idx, row_mask in enumerate(mask.cpu()):
                    row_probs = probs[row_idx, row_mask]
                    flat_probs.extend(float(x) for x in row_probs.detach().cpu().tolist())
            else:
                flat = flatten_masked_tensors(labels, dec["predictions"], dec["probabilities"], mask & sup)
                flat_labels.extend(int(x) for x in flat["labels"])
                flat_preds.extend(int(x) for x in flat["predictions"])
                flat_probs.extend(float(x) for x in flat["probabilities"])
                for cid, row_mask in zip(batch["chunk_ids"], (mask & sup).cpu()):
                    count = int(row_mask.sum().item())
                    flat_chunk_ids.extend([str(cid)] * count)

    if paradigm == "session_aggregate":
        chunk_metrics = compute_binary_classification_metrics(chunk_true, chunk_pred, chunk_score)
        event_metrics = {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "auroc": None,
            "note": "no per-transition ground truth for session-level datasets",
            "num_transition_probs": len(flat_probs),
        }
        return {
            "loss": float(sum(losses) / max(1, len(losses))),
            "event_metrics": event_metrics,
            "chunk_metrics": chunk_metrics,
            "num_events": len(flat_probs),
            "num_chunks": len(chunk_true),
            **_training_metadata(cfg),
        }

    event_metrics = compute_binary_classification_metrics(flat_labels, flat_preds, flat_probs)
    chunk_aggr = compute_chunk_labels_from_events(flat_chunk_ids, flat_labels, flat_preds, flat_probs)
    chunk_metrics = compute_binary_classification_metrics(
        chunk_aggr["chunk_true"],
        chunk_aggr["chunk_pred"],
        chunk_aggr["chunk_score"],
    )
    return {
        "loss": float(sum(losses) / max(1, len(losses))),
        "event_metrics": event_metrics,
        "chunk_metrics": chunk_metrics,
        "num_events": len(flat_labels),
        "num_chunks": len(chunk_aggr["chunk_true"]),
        **_training_metadata(cfg),
    }


def _train_detector_trial(
    *,
    cfg: Dict[str, Any],
    trial_index: int,
    trial_params: Dict[str, Any],
    train_sequences: Sequence[Dict[str, Any]],
    val_sequences: Sequence[Dict[str, Any]],
) -> DetectorTrialResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = FeatureChunkDataset(train_sequences)
    val_ds = FeatureChunkDataset(val_sequences)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(trial_params["detector_batch_size"]),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=collate_feature_chunk_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(trial_params["detector_batch_size"]),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=collate_feature_chunk_batch,
    )
    input_dim = infer_input_dim(train_sequences)
    model = SequentialEventDetector(
        input_dim=input_dim,
        hidden_dim=int(trial_params["detector_hidden_dim"]),
        dropout=float(trial_params["detector_dropout"]),
        mlp_layers=int(trial_params["detector_mlp_layers"]),
        mlp_expansion=int(trial_params["detector_mlp_expansion"]),
        fusion_with_current=bool(trial_params["detector_fusion_with_current"]),
        recurrent_type=str(cfg.get("detector_recurrent_type", "gru")),
        head_layers=int(trial_params["detector_head_layers"]),
        head_activation=str(trial_params["detector_head_activation"]),
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(trial_params["detector_lr"]),
        weight_decay=float(trial_params["weight_decay"]),
    )
    max_epochs = int(cfg.get("detector_max_epochs", 60))
    patience = int(cfg.get("detector_early_stopping_patience", 8))
    label_smoothing = float(trial_params["detector_label_smoothing"])
    pos_weight = float(trial_params["detector_pos_weight"])
    grad_clip = float(trial_params["detector_gradient_clip"])
    start = time.perf_counter()
    best_epoch = -1
    best_score = -1.0
    best_metrics: Dict[str, Any] = {}
    best_state_dict: Dict[str, torch.Tensor] = {}
    history: List[Dict[str, Any]] = []
    no_improve = 0
    for epoch in range(max_epochs):
        model.train()
        train_losses: List[float] = []
        for batch in train_loader:
            feats = batch["features"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)
            sup = batch["supervision_mask"].to(device)
            chunk_labels_batch = batch["chunk_labels"].to(device)
            logits = model(feats)
            loss = _compute_detector_loss(
                cfg=cfg,
                logits=logits,
                labels=labels,
                mask=mask,
                supervision_mask=sup,
                chunk_labels=chunk_labels_batch,
                pos_weight=pos_weight,
                label_smoothing=label_smoothing,
            )
            opt.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            opt.step()
            train_losses.append(float(loss.item()))
        val_metrics = _evaluate_detector(
            model=model,
            loader=val_loader,
            cfg=cfg,
            pos_weight=pos_weight,
            label_smoothing=label_smoothing,
            device=device,
        )
        score = _early_stopping_score(cfg, val_metrics)
        row = {
            "epoch": int(epoch),
            "train_loss": float(sum(train_losses) / max(1, len(train_losses))),
            "val_loss": float(val_metrics["loss"]),
            "val_event_f1": float(val_metrics["event_metrics"].get("f1") or 0.0),
            "val_chunk_f1": float(val_metrics["chunk_metrics"].get("f1") or 0.0),
            "early_stopping_score": score,
        }
        history.append(row)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = val_metrics
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    wall = time.perf_counter() - start
    if bool(cfg.get("optuna_trial_compact_logging", True)):
        log_status(
            f"[optuna-trial] idx={trial_index} recurrent={cfg.get('detector_recurrent_type','gru')} "
            f"paradigm={_training_paradigm(cfg)} agg={cfg.get('session_aggregation_method','')} "
            f"best_epoch={best_epoch} best_val_score={best_score:.6f} "
            f"best_val_chunk_f1={float(best_metrics.get('chunk_metrics',{}).get('f1') or 0.0):.6f} wall_s={wall:.2f}"
        )
    return DetectorTrialResult(
        trial_index=trial_index,
        params=trial_params,
        best_epoch=best_epoch,
        best_score=best_score,
        best_val_metrics=best_metrics,
        best_state_dict=best_state_dict,
        wall_seconds=wall,
        history=history,
    )


def _default_trial_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "detector_hidden_dim": int(cfg.get("detector_hidden_dim", 256)),
        "detector_dropout": float(cfg.get("detector_dropout", 0.1)),
        "detector_mlp_expansion": int(cfg.get("detector_mlp_expansion", 4)),
        "detector_mlp_layers": int(cfg.get("detector_mlp_layers", 4)),
        "detector_fusion_with_current": bool(cfg.get("detector_fusion_with_current", True)),
        "detector_head_layers": int(cfg.get("detector_head_layers", 4)),
        "detector_head_activation": str(cfg.get("detector_head_activation", "gelu")),
        "detector_lr": float(cfg.get("detector_lr", 1e-3)),
        "weight_decay": float(cfg.get("weight_decay", 1e-4)),
        "detector_batch_size": int(cfg.get("detector_batch_size", 16)),
        "detector_gradient_clip": float(cfg.get("detector_gradient_clip", 0.5)),
        "detector_label_smoothing": float(cfg.get("detector_label_smoothing", 0.02)),
        "detector_pos_weight": float(cfg.get("detector_pos_weight", 2.0)),
    }


def _sample_trial_params(cfg: Dict[str, Any], trial_index: int) -> Dict[str, Any]:
    import random

    rng = random.Random(int(cfg.get("seed", 42)) + int(trial_index))

    def pick(name: str, default: Sequence[Any]) -> Any:
        options = cfg.get(name, default)
        if not isinstance(options, (list, tuple)) or not options:
            options = default
        return options[rng.randrange(len(options))]

    return {
        "detector_hidden_dim": int(pick("optuna_detector_hidden_dim_choices", [192, 256, 320])),
        "detector_dropout": float(pick("optuna_detector_dropout_choices", [0.05, 0.1, 0.15])),
        "detector_mlp_expansion": int(pick("optuna_detector_mlp_expansion_choices", [3, 4, 5])),
        "detector_mlp_layers": int(pick("optuna_detector_mlp_layers_choices", [4, 6])),
        "detector_fusion_with_current": bool(pick("optuna_detector_fusion_with_current_choices", [True])),
        "detector_head_layers": int(pick("optuna_detector_head_layers_choices", [3, 4])),
        "detector_head_activation": str(pick("optuna_detector_head_activation_choices", ["gelu", "silu"])),
        "detector_lr": float(pick("optuna_detector_lr_choices", [5e-4, 1e-3, 1.5e-3, 2e-3])),
        "weight_decay": float(pick("optuna_weight_decay_choices", [1e-5, 1e-4, 5e-4, 1e-3])),
        "detector_batch_size": int(pick("optuna_detector_batch_size_choices", [16, 32])),
        "detector_gradient_clip": float(pick("optuna_detector_gradient_clip_choices", [0.0, 0.5, 1.0])),
        "detector_label_smoothing": float(pick("optuna_detector_label_smoothing_choices", [0.0, 0.02, 0.05])),
        "detector_pos_weight": float(pick("optuna_detector_pos_weight_choices", [1.0, 1.5, 2.0, 3.0])),
    }


def _evaluate_best_on_split(
    *,
    cfg: Dict[str, Any],
    state_dict: Dict[str, torch.Tensor],
    trial_params: Dict[str, Any],
    sequences: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        FeatureChunkDataset(sequences),
        batch_size=int(trial_params["detector_batch_size"]),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=collate_feature_chunk_batch,
    )
    model = SequentialEventDetector(
        input_dim=infer_input_dim(sequences),
        hidden_dim=int(trial_params["detector_hidden_dim"]),
        dropout=float(trial_params["detector_dropout"]),
        mlp_layers=int(trial_params["detector_mlp_layers"]),
        mlp_expansion=int(trial_params["detector_mlp_expansion"]),
        fusion_with_current=bool(trial_params["detector_fusion_with_current"]),
        recurrent_type=str(cfg.get("detector_recurrent_type", "gru")),
        head_layers=int(trial_params["detector_head_layers"]),
        head_activation=str(trial_params["detector_head_activation"]),
    ).to(device)
    model.load_state_dict(state_dict)
    return _evaluate_detector(
        model=model,
        loader=loader,
        cfg=cfg,
        pos_weight=float(trial_params["detector_pos_weight"]),
        label_smoothing=float(trial_params["detector_label_smoothing"]),
        device=device,
    )


def run_detector_training(cfg: Dict[str, Any]) -> Dict[str, Any]:
    stage_started = time.perf_counter()
    log_status(
        f"[stage] run_detector_training start model={cfg.get('model_name')} "
        f"rnn={cfg.get('detector_recurrent_type', 'gru')} "
        f"paradigm={_training_paradigm(cfg)} agg={cfg.get('session_aggregation_method', '')} "
        f"trials={int(cfg.get('optuna_max_trials', 48))}"
    )
    set_seed(int(cfg.get("seed", 42)))
    train_sequences = _load_feature_sequences(cfg["train_feature_json_path"])
    val_sequences = _load_feature_sequences(cfg["val_feature_json_path"])
    test_sequences = _load_feature_sequences(cfg["test_feature_json_path"])
    test_feature_blob = load_json(cfg["test_feature_json_path"])
    calc_inference_timing = test_feature_blob.get("feature_extraction_timing", {})
    if not train_sequences or not val_sequences or not test_sequences:
        raise ValueError("Empty feature sequences; run_feature_calculation first.")

    max_trials = int(cfg.get("optuna_max_trials", 48))
    auto_tune = bool(cfg.get("auto_tune_detector", True))
    if not auto_tune:
        max_trials = 1

    trial_results: List[DetectorTrialResult] = []
    for trial_idx in range(max_trials):
        params = _sample_trial_params(cfg, trial_idx) if auto_tune else _default_trial_params(cfg)
        result = _train_detector_trial(
            cfg=cfg,
            trial_index=trial_idx,
            trial_params=params,
            train_sequences=train_sequences,
            val_sequences=val_sequences,
        )
        trial_results.append(result)

    best = max(trial_results, key=lambda r: r.best_score)
    best_val_metrics_payload = dict(best.best_val_metrics)
    best_val_metrics_payload.update(_training_metadata(cfg))

    test_metrics_payload = _evaluate_best_on_split(
        cfg=cfg,
        state_dict=best.best_state_dict,
        trial_params=best.params,
        sequences=test_sequences,
    )
    test_metrics_payload.update(_training_metadata(cfg))
    if isinstance(calc_inference_timing, dict) and calc_inference_timing:
        test_metrics_payload["calc_inference_timing"] = calc_inference_timing

    save_torch(
        {
            "state_dict": best.best_state_dict,
            "trial_index": int(best.trial_index),
            "params": best.params,
            "best_epoch": int(best.best_epoch),
            "val_metrics": best_val_metrics_payload,
            "test_metrics": test_metrics_payload,
            "model_name": cfg.get("model_name"),
        },
        cfg["detector_ckpt_path"],
    )
    save_torch(
        {
            "state_dict": best.best_state_dict,
            "trial_index": int(best.trial_index),
            "params": best.params,
            "best_epoch": int(best.best_epoch),
            "val_metrics": best_val_metrics_payload,
        },
        Path(cfg["detector_ckpt_path"]).with_name("detector_best.pt"),
    )

    ensure_dir(Path(cfg["detector_ckpt_path"]).with_name("detector_best_package"))
    save_json(best.params, Path(cfg["detector_ckpt_path"]).with_name("detector_best_package") / "config.json")
    save_json({"history": best.history}, Path(cfg["detector_ckpt_path"]).with_name("detector_best_package") / "training_log.json")
    save_json({"history": best.history}, Path(cfg["metrics_dir"]) / "train_history.json")
    save_json(best_val_metrics_payload, Path(cfg["metrics_dir"]) / "val_metrics.json")
    save_json(test_metrics_payload, Path(cfg["metrics_dir"]) / "test_metrics.json")

    tuning = {
        "auto_tune_detector": auto_tune,
        "best_trial": best.params,
        "best_score": float(best.best_score),
        "best_val_summary": best_val_metrics_payload,
        "best_trial_index": int(best.trial_index),
        "best_epoch": int(best.best_epoch),
        "num_trials": int(len(trial_results)),
        "trials": [
            {
                **r.params,
                "trial_index": int(r.trial_index),
                "best_epoch": int(r.best_epoch),
                "best_score": float(r.best_score),
                "trial_wall_seconds": float(r.wall_seconds),
                "best_val_event_f1": float(r.best_val_metrics.get("event_metrics", {}).get("f1") or 0.0),
                "best_val_chunk_f1": float(r.best_val_metrics.get("chunk_metrics", {}).get("f1") or 0.0),
                "best_val_summary": r.best_val_metrics,
            }
            for r in trial_results
        ],
        "optuna_trial_stats": {
            "n_trials_requested": int(max_trials),
            "n_trials_in_study": int(len(trial_results)),
            "n_complete": int(len(trial_results)),
            "n_pruned": 0,
        },
    }
    save_json(tuning, Path(cfg["metrics_dir"]) / "detector_tuning.json")
    early_metric = "val_chunk_f1" if _training_paradigm(cfg) == "session_aggregate" else "val_event_f1"
    save_json(
        {
            "best_trial_index": int(best.trial_index),
            "best_epoch": int(best.best_epoch),
            "early_stopping_metric": early_metric,
            "best_val_event_f1": float(best.best_val_metrics.get("event_metrics", {}).get("f1") or 0.0),
            "best_val_chunk_f1": float(best.best_val_metrics.get("chunk_metrics", {}).get("f1") or 0.0),
            "test_event_f1": float(test_metrics_payload.get("event_metrics", {}).get("f1") or 0.0),
            "test_chunk_f1": float(test_metrics_payload.get("chunk_metrics", {}).get("f1") or 0.0),
            "detector_recurrent_type": str(cfg.get("detector_recurrent_type", "gru")),
            "best_trial_wall_seconds": float(best.wall_seconds),
            "optuna_trial_stats": tuning["optuna_trial_stats"],
            **_training_metadata(cfg),
        },
        Path(cfg["detector_ckpt_path"]).with_name("detector_best_meta.json"),
    )
    save_json(
        {
            "mode": "full_pipeline",
            "model_name": str(cfg.get("model_name", "")),
            "num_trials": int(len(trial_results)),
            "best_trial_index": int(best.trial_index),
            "best_score": float(best.best_score),
            "detector": {"best_trial_config": best.params},
            **_training_metadata(cfg),
        },
        Path(cfg["metrics_dir"]) / "run_profile.json",
    )
    wall_s = time.perf_counter() - stage_started
    log_status(
        f"[stage] run_detector_training done best_trial={int(best.trial_index)} "
        f"best_val_score={float(best.best_score):.6f} wall_s={wall_s:.2f}"
    )
    return {
        "best_trial_index": best.trial_index,
        "best_score": best.best_score,
        "test_metrics": test_metrics_payload,
    }


def run_full_pipeline(cfg: Dict[str, Any]) -> Dict[str, Any]:
    stage_started = time.perf_counter()
    log_status(
        f"[stage] run_full_pipeline start detector_only={bool(cfg.get('circuit_detector_only', False))}"
    )
    if not bool(cfg.get("circuit_detector_only", False)):
        log_status("[stage] full_pipeline step=selection")
        run_neuron_selection(cfg)
        log_status("[stage] full_pipeline step=calc")
        run_feature_calculation(cfg)
    else:
        for path_key in ("selected_neurons_path", "train_feature_json_path", "val_feature_json_path", "test_feature_json_path", "calc_manifest_path"):
            if not resolve_path(cfg[path_key]).is_file():
                raise FileNotFoundError(f"detector-only reuse missing artifact: {cfg[path_key]}")
    log_status("[stage] full_pipeline step=detector")
    out = run_detector_training(cfg)
    wall_s = time.perf_counter() - stage_started
    log_status(f"[stage] run_full_pipeline done wall_s={wall_s:.2f}")
    return out


def run_evaluation(cfg: Dict[str, Any]) -> Dict[str, Any]:
    test_blob = load_json(cfg["test_feature_json_path"])
    _validate_feature_artifact_schema(test_blob, "test")
    metrics_path = Path(cfg["metrics_dir"]) / "test_metrics.json"
    if metrics_path.is_file():
        return load_json(metrics_path)
    return {"status": "no_metrics"}

