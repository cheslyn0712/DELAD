"""Data loading and event construction for circuit probing."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from delad.utils import log_status, resolve_path

from delad.session_aggregation import SESSION_AGGREGATION_METHODS


DEFAULT_LOG_KEYS = ("text", "log", "content", "message")
DEFAULT_LABEL_KEYS = ("label", "labels", "target", "y")
SEQUENCE_LEVEL_DATASETS = {"hdfs", "hadoop"}
LINE_LEVEL_DATASETS = {"bgl", "thunderbird"}
SESSION_AGGREGATION_METHODS = ("max", "topk_mean", "noisy_or")


def is_sequence_level_dataset(cfg: Dict[str, Any], path: Optional[str] = None) -> bool:
    """Return True for HDFS/Hadoop and other sequence-level datasets."""

    return infer_dataset_name(cfg, path) in SEQUENCE_LEVEL_DATASETS


def resolve_training_paradigm(cfg: Dict[str, Any], path: Optional[str] = None) -> str:
    """Return ``session_aggregate`` for HDFS/Hadoop or ``event_supervised`` for BGL/Thunderbird."""

    dataset_name = infer_dataset_name(cfg, path)
    label_granularity = str(cfg.get("_preview_label_granularity", "")).strip().lower()
    mode = resolve_detector_event_mode(dataset_name, label_granularity or "event")
    if mode == "session_aggregate":
        return "session_aggregate"
    return "event_supervised"


def resolve_chunk_session_label(labels: Sequence[int], label_granularity: str) -> int:
    """Resolve the session/chunk label Y_c from chunk labels."""

    if label_granularity == "chunk":
        return int(labels[0])
    return int(max(int(x) for x in labels))


@dataclass
class EventRecord:
    """A single prediction event built from a log chunk."""

    chunk_id: str
    event_index: int
    prefix_logs: List[str]
    target_log: str
    target_label: int
    prompt: str
    supervise: bool = True


@dataclass
class SelectionSample:
    """One chunk-level branch-point tracing sample for feature selection."""

    prompt: str
    label: int
    chunk_index: int
    chunk_id: str
    dataset_name: str
    target_mode: str
    target_pos: int
    target_log: str
    prefix_logs: List[str]


def build_prefix_prompt(
    prefix_logs: Sequence[str],
    prompt_prefix: str = "",
    prompt_suffix: str = "\n\nNext log:",
) -> str:
    """Build the fixed prompt used as the frozen LLM input interface."""

    if not prefix_logs:
        prefix_block = "[EMPTY PREFIX]"
    else:
        prefix_block = "\n".join(f"Log_{idx + 1}: {log}" for idx, log in enumerate(prefix_logs))
    return f"{prompt_prefix}{prefix_block}{prompt_suffix}"


def normalize_circuit_instruction_length(raw: Any) -> str:
    """Return ``short`` | ``medium`` | ``long`` for prompt instruction tiers."""

    s = str(raw or "medium").strip().lower()
    if s in ("s", "short"):
        return "short"
    if s in ("m", "med", "medium", "mid"):
        return "medium"
    if s in ("l", "long"):
        return "long"
    raise ValueError(
        "circuit_prompt_instruction_length must be short, medium, or long; "
        f"got {raw!r}"
    )


def normalize_circuit_calc_input_mode(raw: Any) -> str:
    """Return ``single_message`` | ``log_block`` for calc/extract prompting."""

    s = str(raw or "single_message").strip().lower()
    if s in ("single_message", "message", "msg", "single", "single_log"):
        return "single_message"
    if s in ("log_block", "block", "blk", "prefix_block"):
        return "log_block"
    raise ValueError(
        "circuit_calc_input_mode must be single_message or log_block; "
        f"got {raw!r}"
    )


def circuit_instruction_length_slug(length: str) -> str:
    """Stable path segment for output dirs: lenS | lenM | lenL."""

    m = {"short": "lenS", "medium": "lenM", "long": "lenL"}
    key = normalize_circuit_instruction_length(length)
    return m[key]


def circuit_calc_input_mode_slug(mode: str) -> str:
    """Stable path segment: calcmsg | calcblk."""

    mkey = normalize_circuit_calc_input_mode(mode)
    return "calcmsg" if mkey == "single_message" else "calcblk"


def circuit_prompt_variant_dir_name(
    prompt_key: str,
    instruction_length: str,
    calc_input_mode: str,
    *,
    use_extended_path_layout: bool,
) -> str:
    """Directory name under ``circuit_dataset_run_base`` for probe artifacts."""

    key = str(prompt_key).strip().upper()
    if key not in ("A", "B"):
        raise ValueError(f"circuit_prompt_variant_dir_name: expected A or B, got {prompt_key!r}")
    base = f"prompt_{key}"
    if not use_extended_path_layout:
        return base
    ins = circuit_instruction_length_slug(instruction_length)
    cx = circuit_calc_input_mode_slug(calc_input_mode)
    return f"{base}_{ins}_{cx}"


def circuit_prompt_feature_dir_segment(cfg: Dict[str, Any], prompt_key: str) -> str:
    """Resolve feature-root directory segment for circuit sweeps / ablations."""

    key = str(prompt_key).strip().upper()
    if key not in ("A", "B"):
        raise ValueError(f"circuit_prompt_feature_dir_segment: expected A or B, got {prompt_key!r}")
    ins = normalize_circuit_instruction_length(cfg.get("circuit_prompt_instruction_length"))
    cm = normalize_circuit_calc_input_mode(cfg.get("circuit_calc_input_mode"))
    use_ext = bool(
        cfg.get("circuit_instruction_length_sweep", cfg.get("circuit_prompt_ablation_sweep", False))
    ) or (ins != "medium" or cm != "single_message")
    return circuit_prompt_variant_dir_name(key, ins, cm, use_extended_path_layout=use_ext)


def _tier_lead_selection_a(length: str) -> str:
    if length == "short":
        return "Task: use the chronological log prefix to prepare for next-log continuation.\n\n"
    if length == "medium":
        return (
            "You are given the ordered prefix of a system-log sequence, ending "
            "immediately before a hidden target line. Use the full prefix as "
            "context for next-log continuation, preserving event order, component "
            "names, severity tokens, identifiers, and recurring message patterns.\n\n"
        )
    return (
        "You are performing prefix-aware next-log continuation for OS and "
        "datacenter telemetry. The visible lines are strictly chronological and "
        "stop exactly at the branch boundary before an unseen target log. Build a "
        "high-fidelity continuation context from temporal order, component and node "
        "identity, severity transitions, numeric/code fields, repeated templates, "
        "resource/fault progression, and subtle anomaly cues, while focusing only "
        "on what the immediately following log line should look like.\n\n"
    )


def _tier_lead_calc_a_single(length: str) -> str:
    if length == "short":
        return "Task: continue from the immediately previous log line only.\n\n"
    if length == "medium":
        return (
            "You are given only the immediately previous system-log entry, not the "
            "full prefix. Infer the local next-log continuation from this line's "
            "component, severity, identifiers, template structure, and message cues.\n\n"
        )
    return (
        "You are in local-context inference mode for next-log continuation. Only "
        "one prior log line is visible, so do not assume any unseen earlier prefix. "
        "Use the line's component path, severity marker, event identifiers, numeric "
        "fields, delimiters, and semantic fault/status cues to form a compact "
        "representation of the most plausible immediately succeeding log line.\n\n"
    )


def _tier_lead_calc_ab_block(length: str) -> str:
    """Shared lead when calc uses numbered prefix block (A next-log tasks)."""

    if length == "short":
        return "Task: continue after the numbered chronological log context.\n\n"
    if length == "medium":
        return (
            "You are given an ordered block of system logs that ends immediately "
            "before the hidden next line. Model the next-log continuation from the "
            "visible temporal pattern, components, severities, identifiers, and "
            "message templates.\n\n"
        )
    return (
        "You are performing context-block next-log continuation for streamed "
        "system telemetry. The numbered logs form a chronological window ending "
        "exactly before an unseen target line. Track inter-line dependencies, "
        "state transitions, repeated templates, component handoffs, fault/resource "
        "progression, and local anomaly cues, then prepare the continuation context "
        "for the immediately following log only.\n\n"
    )


def _tier_footer_a_next_log() -> str:
    return "Next log:\n\n"


def _tier_lead_selection_b(length: str) -> str:
    if length == "short":
        return (
            "Task: use the chronological log prefix to label the hidden next log "
            "as Normal or Anomalous.\n\n"
        )
    if length == "medium":
        return (
            "You are given the ordered prefix of a system-log sequence, ending "
            "immediately before a hidden target line. Use the full prefix to assess "
            "event-level risk at that boundary, and label only the next log line as "
            "Normal or Anomalous rather than summarizing the whole chunk.\n\n"
        )
    return (
        "You are performing prefix-aware event-level anomaly assessment for OS "
        "and datacenter telemetry. The visible logs are strictly chronological and "
        "stop immediately before one unseen target line. Integrate evidence from "
        "fault progression, severity shifts, component/node identity, recurring "
        "templates, timing/order dependencies, resource failures, recovery signals, "
        "and benign background chatter, then judge only whether the immediately "
        "following log should be labeled Normal or Anomalous.\n\n"
    )


def _tier_lead_calc_b_single(length: str) -> str:
    if length == "short":
        return (
            "Task: using only the immediately previous log, label the hidden next "
            "log as Normal or Anomalous.\n\n"
        )
    if length == "medium":
        return (
            "You are given only the immediately previous system-log entry before a "
            "hidden successor. Do not use any unseen prefix; infer local event-level "
            "risk from this line's severity, component, identifiers, template, and "
            "message semantics, then label the next log as Normal or Anomalous.\n\n"
        )
    return (
        "You are in local-context inference mode for event-level anomaly labeling. "
        "Exactly one prior log line is visible, and the full prefix is unavailable. "
        "Extract risk evidence from component identity, severity markers, error/status "
        "phrasing, numeric/code fields, delimiters, and abnormal transition hints, "
        "then decide whether the immediately succeeding hidden log should be Normal "
        "or Anomalous.\n\n"
    )


def _tier_lead_calc_b_block(length: str) -> str:
    if length == "short":
        return (
            "Task: use the numbered chronological context to label only the hidden "
            "next log as Normal or Anomalous.\n\n"
        )
    if length == "medium":
        return (
            "You are given an ordered block of system logs that ends immediately "
            "before a hidden successor. Correlate the visible lines to estimate "
            "event-level risk at that boundary, and label only the next log as "
            "Normal or Anomalous.\n\n"
        )
    return (
        "You are performing context-block anomaly assessment for streamed system "
        "telemetry. The numbered logs form a chronological window ending exactly "
        "before one hidden target line. Track severity transitions, component "
        "handoffs, repeated templates, fault escalation, recovery patterns, resource "
        "signals, and deceptive normal-looking chatter, then judge only whether the "
        "immediately following log should be Normal or Anomalous.\n\n"
    )


def _tier_footer_b_classify() -> str:
    return "Next-log label (Normal or Anomalous):\n\n"


def _normalize_prompt_extract_option(raw_value: Any) -> str:
    """Normalize a raw option value into ``legacy`` / ``A`` / ``B``."""

    raw = str(raw_value or "legacy").strip()
    u = raw.upper()
    if u in ("", "LEGACY", "DEFAULT", "0"):
        return "legacy"
    if u in ("A", "OPTION_A", "OPTIONA", "NEXT_LOG", "1"):
        return "A"
    if u in ("B", "OPTION_B", "OPTIONB", "FINAL_LOG", "CLASSIFY", "2"):
        return "B"
    raise ValueError(
        "Invalid prompt_extract_option: expected 'A' (next-template / Option A), "
        "'B' (final-template classification / Option B), or 'legacy' (current "
        f"prefix + Log_i block + suffix). Got {raw!r}."
    )


def resolve_prompt_extract_option(cfg: Dict[str, Any]) -> str:
    """Return the global fallback prompt option."""

    return _normalize_prompt_extract_option(cfg.get("prompt_extract_option", "legacy"))


def resolve_prompt_extract_option_for_stage(cfg: Dict[str, Any], stage: str) -> str:
    """Resolve prompt option for ``selection`` or ``calc`` stage."""

    stage_key = f"{str(stage).strip().lower()}_prompt_extract_option"
    stage_raw = cfg.get(stage_key, None)
    if stage_raw is not None and str(stage_raw).strip() != "":
        return _normalize_prompt_extract_option(stage_raw)
    return resolve_prompt_extract_option(cfg)


def _format_numbered_log_block(
    logs: Sequence[str], empty_placeholder: str = "[EMPTY LOG HISTORY]"
) -> str:
    """Format a prefix history block as numbered log lines."""

    if not logs:
        return empty_placeholder
    return "\n".join(f"Log_{idx + 1}: {log}" for idx, log in enumerate(logs))


def build_option_a_selection_prompt(cfg: Dict[str, Any], *, log_history: Sequence[str]) -> str:
    """Option A selection: full prefix history for sensitive-neuron discovery."""

    length = normalize_circuit_instruction_length(cfg.get("circuit_prompt_instruction_length"))
    history_block = _format_numbered_log_block(log_history)
    lead = _tier_lead_selection_a(length)
    return f"{lead}Prefix log history (chronological):\n{history_block}\n\n{_tier_footer_a_next_log()}"


def build_option_a_calc_prompt_message(
    cfg: Dict[str, Any], *, previous_log: str
) -> str:
    """Option A calc: narrow single previous log (message mode)."""

    length = normalize_circuit_instruction_length(cfg.get("circuit_prompt_instruction_length"))
    prev = previous_log.strip() if previous_log else ""
    block = prev if prev else "[EMPTY PREVIOUS LOG]"
    lead = _tier_lead_calc_a_single(length)
    return f"{lead}Immediate previous log:\n{block}\n\n{_tier_footer_a_next_log()}"


def build_option_a_calc_prompt_block(
    cfg: Dict[str, Any], *, prefix_logs: Sequence[str]
) -> str:
    """Option A calc: numbered prefix block equal to logs before branch (log_block mode)."""

    length = normalize_circuit_instruction_length(cfg.get("circuit_prompt_instruction_length"))
    history_block = _format_numbered_log_block(prefix_logs)
    lead = _tier_lead_calc_ab_block(length)
    return f"{lead}Local log context (chronological):\n{history_block}\n\n{_tier_footer_a_next_log()}"


def build_option_b_selection_prompt(cfg: Dict[str, Any], *, log_history: Sequence[str]) -> str:
    """Option B selection: full prefix history for sensitive-neuron discovery."""

    length = normalize_circuit_instruction_length(cfg.get("circuit_prompt_instruction_length"))
    history_block = _format_numbered_log_block(log_history)
    lead = _tier_lead_selection_b(length)
    return f"{lead}Prefix log history (chronological):\n{history_block}\n\n{_tier_footer_b_classify()}"


def build_option_b_calc_prompt_message(
    cfg: Dict[str, Any], *, previous_log: str
) -> str:
    """Option B calc: narrow single previous log."""

    length = normalize_circuit_instruction_length(cfg.get("circuit_prompt_instruction_length"))
    prev = previous_log.strip() if previous_log else ""
    block = prev if prev else "[EMPTY PREVIOUS LOG]"
    lead = _tier_lead_calc_b_single(length)
    return f"{lead}Immediate previous log:\n{block}\n\n{_tier_footer_b_classify()}"


def build_option_b_calc_prompt_block(
    cfg: Dict[str, Any], *, prefix_logs: Sequence[str]
) -> str:
    """Option B calc: numbered prefix block equal to logs before branch (log_block mode)."""

    length = normalize_circuit_instruction_length(cfg.get("circuit_prompt_instruction_length"))
    history_block = _format_numbered_log_block(prefix_logs)
    lead = _tier_lead_calc_b_block(length)
    return f"{lead}Local log context (chronological):\n{history_block}\n\n{_tier_footer_b_classify()}"


def build_circuit_extraction_prompt(
    cfg: Dict[str, Any],
    *,
    full_chunk_logs: List[str],
    target_idx: int,
    for_calc_stage: bool,
    chunk_legacy_narrow_context: bool = False,
    rng_salt: str = "",
) -> str:
    """Assemble the frozen-LLM prompt for feature extraction (selection or calc)."""

    if target_idx < 1 or target_idx >= len(full_chunk_logs):
        raise ValueError("build_circuit_extraction_prompt: invalid target_idx for chunk")

    stage = "calc" if for_calc_stage else "selection"
    opt = resolve_prompt_extract_option_for_stage(cfg, stage)
    if opt == "legacy":
        if chunk_legacy_narrow_context:
            prefix_logs = [full_chunk_logs[target_idx - 1]]
        else:
            prefix_logs = full_chunk_logs[:target_idx]
        if for_calc_stage:
            prompt_prefix = str(cfg.get("calc_prompt_prefix", ""))
            prompt_suffix = str(cfg.get("calc_prompt_suffix", "\n\nNext log:"))
        else:
            prompt_prefix = str(cfg.get("selection_prompt_prefix", cfg.get("prompt_prefix", "")))
            prompt_suffix = str(cfg.get("selection_prompt_suffix", cfg.get("prompt_suffix", "\n\nNext log:")))
        return build_prefix_prompt(prefix_logs, prompt_prefix, prompt_suffix)

    prefix_logs = full_chunk_logs[:target_idx]
    prev_one = full_chunk_logs[target_idx - 1]
    calc_mode = normalize_circuit_calc_input_mode(cfg.get("circuit_calc_input_mode"))
    if opt == "A":
        if for_calc_stage:
            if calc_mode == "single_message":
                return build_option_a_calc_prompt_message(cfg, previous_log=prev_one)
            return build_option_a_calc_prompt_block(cfg, prefix_logs=prefix_logs)
        return build_option_a_selection_prompt(cfg, log_history=prefix_logs)
    if opt == "B":
        if for_calc_stage:
            if calc_mode == "single_message":
                return build_option_b_calc_prompt_message(cfg, previous_log=prev_one)
            return build_option_b_calc_prompt_block(cfg, prefix_logs=prefix_logs)
        return build_option_b_selection_prompt(cfg, log_history=prefix_logs)
    raise RuntimeError(f"Unhandled prompt extract option: {opt}")


def load_data_file(path: str) -> List[Dict[str, Any]]:
    """Load a JSON or line-delimited JSON split file into chunk dictionaries."""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file does not exist: {file_path}")

    with file_path.open("r", encoding="utf-8") as handle:
        first_nonempty = ""
        for line in handle:
            if line.strip():
                first_nonempty = line.lstrip()
                break

    if not first_nonempty:
        return []

    if first_nonempty.startswith("{"):
        records = _load_json_lines_file(file_path)
        if records is not None:
            return records

    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "samples", "chunks", "records"):
            value = data.get(key)
            if isinstance(value, list):
                return value

    raise ValueError(f"Unsupported JSON structure in {file_path}.")


def is_jsonl_chunk_file(path: str) -> bool:
    """Return True when the split file is line-delimited JSON objects."""

    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return line.lstrip().startswith("{")
    return False


def scan_chunk_index_buckets(path: str, cfg: Dict[str, Any]) -> Tuple[List[int], List[int], int]:
    """Scan a JSONL split and bucket eligible chunk line indices without full-file RAM."""

    normal_indices: List[int] = []
    anomalous_indices: List[int] = []
    line_idx = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
                if not isinstance(chunk, dict):
                    line_idx += 1
                    continue
                logs = _extract_chunk_logs(chunk, cfg)
                if len(logs) < 2:
                    line_idx += 1
                    continue
                labels, _ = _extract_chunk_labels(chunk, cfg, len(logs))
                if _is_anomalous_chunk(labels):
                    anomalous_indices.append(line_idx)
                else:
                    normal_indices.append(line_idx)
            except Exception:
                pass
            line_idx += 1
    return normal_indices, anomalous_indices, line_idx


def load_chunks_by_line_indices(path: str, indices: Sequence[int]) -> Dict[int, Dict[str, Any]]:
    """Load only the requested JSONL line indices."""

    wanted = {int(i) for i in indices}
    if not wanted:
        return {}
    max_wanted = max(wanted)
    out: Dict[int, Dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            if line_idx > max_wanted and len(out) == len(wanted):
                break
            if line_idx not in wanted:
                continue
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            if isinstance(chunk, dict):
                out[line_idx] = chunk
    missing = sorted(wanted - set(out.keys()))
    if missing:
        raise ValueError(f"Missing chunk line indices in {path}: {missing[:8]}{'...' if len(missing) > 8 else ''}")
    return out


def load_chunks_by_id_whitelist(
    path: str,
    cfg: Dict[str, Any],
    chunk_id_whitelist: Sequence[str],
) -> Dict[int, Dict[str, Any]]:
    """Stream JSONL and load chunks whose ids appear in ``chunk_id_whitelist``."""

    wanted_ids = {str(x) for x in chunk_id_whitelist}
    out: Dict[int, Dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
                if not isinstance(chunk, dict):
                    continue
                chunk_id = _extract_chunk_id(chunk, cfg, line_idx)
                if chunk_id in wanted_ids:
                    out[line_idx] = chunk
            except Exception:
                continue
    return out


def sample_normal_anomaly_line_indices(
    normal_indices: Sequence[int],
    anomalous_indices: Sequence[int],
    normal_target: int,
    anomaly_target: int,
    seed: int,
) -> List[int]:
    """Sample sorted line indices for balanced selection."""

    rng = random.Random(int(seed))
    chosen_anomalous = (
        rng.sample(list(anomalous_indices), min(int(anomaly_target), len(anomalous_indices)))
        if int(anomaly_target) > 0
        else []
    )
    chosen_normal = (
        rng.sample(list(normal_indices), min(int(normal_target), len(normal_indices)))
        if int(normal_target) > 0
        else []
    )
    return sorted(chosen_anomalous + chosen_normal)


def inspect_data_file(path: str) -> Dict[str, Any]:
    """Read only lightweight metadata needed for routing/debug output."""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file does not exist: {file_path}")

    first_chunk: Optional[Dict[str, Any]] = None
    num_chunks: int | str = "unknown"

    with file_path.open("r", encoding="utf-8") as handle:
        first_nonempty = ""
        for line in handle:
            if line.strip():
                first_nonempty = line.lstrip()
                break

    if not first_nonempty:
        return {"path": path, "num_chunks": 0, "first_chunk": None}

    if first_nonempty.startswith("{"):
        count = 0
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parsed = json.loads(line)
                if not isinstance(parsed, dict):
                    raise ValueError(f"Unsupported JSONL record in {file_path}.")
                if first_chunk is None:
                    first_chunk = parsed
                count += 1
        num_chunks = count
    else:
        with file_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            first_chunk = data[0] if data else None
            num_chunks = len(data)
        elif isinstance(data, dict):
            for key in ("data", "samples", "chunks", "records"):
                value = data.get(key)
                if isinstance(value, list):
                    first_chunk = value[0] if value else None
                    num_chunks = len(value)
                    break
        else:
            raise ValueError(f"Unsupported JSON structure in {file_path}.")

    return {"path": path, "num_chunks": num_chunks, "first_chunk": first_chunk}


def _load_json_lines_file(file_path: Path) -> Optional[List[Dict[str, Any]]]:
    """Try to parse one JSON object per line, even when the suffix is .json."""

    records: List[Dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                return None
            if not isinstance(parsed, dict):
                return None
            records.append(parsed)
    return records


def infer_dataset_name(cfg: Dict[str, Any], path: Optional[str] = None) -> str:
    """Infer dataset name from config or input path."""

    configured = str(cfg.get("dataset_name", "auto")).strip().lower()
    if configured and configured != "auto":
        return configured

    candidates = [
        str(path or ""),
        str(cfg.get("train_path", "")),
        str(cfg.get("val_path", "")),
        str(cfg.get("test_path", "")),
        str(cfg.get("dataset_dir", "")),
    ]
    lowered = " ".join(item.lower() for item in candidates if item)
    for name in ("thunderbird", "hdfs", "hadoop", "bgl"):
        if name in lowered:
            return name
    return "unknown"


def _normalize_label(value: Any, positive_value: Any = 1) -> int:
    """Convert common label formats into binary 0/1 values."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value == positive_value or value > 0)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "anomaly", "anomalous", "abnormal", "yes"}:
            return 1
        if lowered in {"0", "false", "normal", "benign", "no"}:
            return 0
    return int(value == positive_value)


def _extract_log_text(log_item: Any, cfg: Dict[str, Any]) -> str:
    """Resolve a single log entry into plain text."""

    if isinstance(log_item, str):
        return log_item
    if isinstance(log_item, dict):
        preferred_key = cfg.get("log_text_key", "text")
        candidate_keys = (preferred_key,) + tuple(key for key in DEFAULT_LOG_KEYS if key != preferred_key)
        for key in candidate_keys:
            value = log_item.get(key)
            if value is not None:
                return str(value)
    raise ValueError(f"Unable to resolve log text from item: {log_item}")


def _extract_chunk_logs(chunk: Dict[str, Any], cfg: Dict[str, Any]) -> List[str]:
    """Resolve the log list from a chunk object."""

    preferred_key = cfg.get("chunk_logs_key", "logs")
    candidate_keys = (preferred_key, "logs", "events", "messages", "items", "sequence")
    raw_logs: Optional[Sequence[Any]] = None
    for key in candidate_keys:
        value = chunk.get(key)
        if isinstance(value, list):
            raw_logs = value
            break
    if raw_logs is None:
        raise ValueError(f"Chunk does not contain a log list: {chunk.keys()}")
    return [_extract_log_text(item, cfg) for item in raw_logs]


def _extract_chunk_labels(
    chunk: Dict[str, Any], cfg: Dict[str, Any], num_logs: int
) -> Tuple[List[int], str]:
    """Resolve labels for a chunk and return their granularity."""

    preferred_key = cfg.get("chunk_labels_key", "labels")
    positive_value = cfg.get("label_positive_value", 1)
    candidate_keys = (preferred_key, "labels", "targets", "y")
    for key in candidate_keys:
        value = chunk.get(key)
        if isinstance(value, list):
            labels = [_normalize_label(item, positive_value) for item in value]
            if len(labels) == num_logs:
                return labels, "event"
            if len(labels) == 1:
                return labels, "chunk"
            raise ValueError(
                f"Label length mismatch in chunk {chunk.get(cfg.get('chunk_id_key', 'chunk_id'))}."
            )

    label_key = cfg.get("log_label_key", "label")
    raw_logs = None
    for key in (
        cfg.get("chunk_logs_key", "logs"),
        "logs",
        "events",
        "messages",
        "items",
        "sequence",
    ):
        value = chunk.get(key)
        if isinstance(value, list):
            raw_logs = value
            break
    if isinstance(raw_logs, list) and raw_logs and isinstance(raw_logs[0], dict):
        labels = []
        for item in raw_logs:
            if label_key not in item:
                break
            labels.append(_normalize_label(item[label_key], positive_value))
        if len(labels) == num_logs:
            return labels, "event"

    raise ValueError("Unable to resolve per-log labels from chunk.")


def _extract_chunk_id(chunk: Dict[str, Any], cfg: Dict[str, Any], index: int) -> str:
    """Resolve a stable chunk identifier."""

    key = cfg.get("chunk_id_key", "chunk_id")
    value = chunk.get(key, chunk.get("id", chunk.get("uuid")))
    if value is None:
        return f"chunk_{index}"
    return str(value)


def _last_anomaly_position_after_midpoint(labels: Sequence[int]) -> int | None:
    """Return the last anomaly position that appears in the latter half."""

    midpoint = len(labels) / 2.0
    candidate: int | None = None
    for index, label in enumerate(labels):
        if int(label) == 1 and index > midpoint:
            candidate = index
    return candidate


def _is_anomalous_chunk(labels: Sequence[int]) -> bool:
    """Treat a chunk as anomalous if any event/log label is anomalous."""

    return any(int(label) == 1 for label in labels)


def build_selection_samples(
    chunks: Sequence[Dict[str, Any]],
    cfg: Dict[str, Any],
    selected_indices: Sequence[int],
    path: str,
) -> Tuple[List[SelectionSample], Dict[str, Any]]:
    """Build branch-oriented selection samples from a sampled chunk subset."""

    dataset_name = infer_dataset_name(cfg, path)

    samples: List[SelectionSample] = []
    skipped_reasons: Dict[str, int] = {}
    strategy_counts = {
        "hdfs_last_log": 0,
        "hadoop_last_log": 0,
        "bgl_normal_last_log": 0,
        "bgl_anomaly_last_late_anomaly": 0,
        "thunderbird_normal_last_log": 0,
        "thunderbird_anomaly_last_late_anomaly": 0,
    }

    def _skip(reason: str) -> None:
        skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1

    for index in selected_indices:
        if isinstance(chunks, dict):
            chunk = chunks.get(int(index))
            if chunk is None:
                _skip("missing_chunk_index")
                continue
        else:
            chunk = chunks[index]
        try:
            logs = _extract_chunk_logs(chunk, cfg)
        except Exception:
            _skip("invalid_logs")
            continue

        if len(logs) < 2:
            _skip("chunk_too_short")
            continue

        try:
            labels, label_granularity = _extract_chunk_labels(chunk, cfg, len(logs))
        except Exception:
            _skip("invalid_labels")
            continue

        chunk_id = _extract_chunk_id(chunk, cfg, index)
        target_mode = "last_log"
        target_pos = len(logs) - 1
        label = labels[0] if label_granularity == "chunk" else labels[target_pos]

        if dataset_name in SEQUENCE_LEVEL_DATASETS or label_granularity == "chunk":
            if dataset_name == "hdfs":
                strategy_counts["hdfs_last_log"] += 1
            else:
                strategy_counts["hadoop_last_log"] += 1
            label = labels[0]
        elif dataset_name in LINE_LEVEL_DATASETS:
            if _is_anomalous_chunk(labels):
                last_late_anomaly_pos = _last_anomaly_position_after_midpoint(labels)
                if last_late_anomaly_pos is None:
                    _skip("missing_late_anomaly_after_midpoint")
                    continue
                if last_late_anomaly_pos <= 0:
                    _skip("empty_prefix_last_late_anomaly")
                    continue
                target_mode = "last_anomaly_log_after_midpoint"
                target_pos = last_late_anomaly_pos
                label = 1
                if dataset_name == "bgl":
                    strategy_counts["bgl_anomaly_last_late_anomaly"] += 1
                else:
                    strategy_counts["thunderbird_anomaly_last_late_anomaly"] += 1
            else:
                if dataset_name == "bgl":
                    strategy_counts["bgl_normal_last_log"] += 1
                else:
                    strategy_counts["thunderbird_normal_last_log"] += 1
                label = 0
        else:
            label = labels[0] if label_granularity == "chunk" else labels[target_pos]

        prefix_logs = logs[:target_pos]
        if not prefix_logs:
            _skip("empty_prefix")
            continue

        prompt = build_circuit_extraction_prompt(
            cfg,
            full_chunk_logs=logs,
            target_idx=int(target_pos),
            for_calc_stage=False,
            chunk_legacy_narrow_context=False,
            rng_salt=f"{chunk_id}\t{target_pos}",
        )
        samples.append(
            SelectionSample(
                prompt=prompt,
                label=int(label),
                chunk_index=int(index),
                chunk_id=chunk_id,
                dataset_name=dataset_name,
                target_mode=target_mode,
                target_pos=int(target_pos),
                target_log=logs[target_pos],
                prefix_logs=prefix_logs,
            )
        )

    summary = {
        "raw_chunk_count": len(selected_indices),
        "num_samples": len(samples),
        "strategy_counts": strategy_counts,
        "skipped_reasons": skipped_reasons,
        "num_skipped": sum(skipped_reasons.values()),
    }
    return samples, summary


def chunk_to_events(chunk: Dict[str, Any], cfg: Dict[str, Any], index: int) -> List[EventRecord]:
    """Convert one chunk into neuron-selection events."""

    logs = _extract_chunk_logs(chunk, cfg)
    labels, label_granularity = _extract_chunk_labels(chunk, cfg, len(logs))
    chunk_id = _extract_chunk_id(chunk, cfg, index)
    dataset_name = infer_dataset_name(cfg, cfg.get("active_split_path"))

    events: List[EventRecord] = []
    if len(logs) < 2:
        return events

    event_mode = resolve_selection_event_mode(dataset_name, label_granularity)
    if event_mode == "last_only":
        target_idx = len(logs) - 1
        prefix_logs = logs[:target_idx]
        prompt = build_circuit_extraction_prompt(
            cfg,
            full_chunk_logs=logs,
            target_idx=int(target_idx),
            for_calc_stage=True,
            chunk_legacy_narrow_context=False,
            rng_salt=f"{chunk_id}\t{target_idx}",
        )
        events.append(
            EventRecord(
                chunk_id=chunk_id,
                event_index=target_idx,
                prefix_logs=prefix_logs,
                target_log=logs[target_idx],
                target_label=labels[0],
                prompt=prompt,
            )
        )
        return events

    for target_idx in range(1, len(logs)):
        prefix_logs = logs[:target_idx]
        prompt = build_circuit_extraction_prompt(
            cfg,
            full_chunk_logs=logs,
            target_idx=int(target_idx),
            for_calc_stage=True,
            chunk_legacy_narrow_context=False,
            rng_salt=f"{chunk_id}\t{target_idx}",
        )
        events.append(
            EventRecord(
                chunk_id=chunk_id,
                event_index=target_idx,
                prefix_logs=prefix_logs,
                target_log=logs[target_idx],
                target_label=labels[target_idx],
                prompt=prompt,
            )
        )
    return events


def build_chunk_event_sequence(
    chunk: Dict[str, Any],
    cfg: Dict[str, Any],
    index: int,
) -> List[EventRecord]:
    """Convert one chunk into single-step detector events with dataset-specific supervision."""

    logs = _extract_chunk_logs(chunk, cfg)
    labels, label_granularity = _extract_chunk_labels(chunk, cfg, len(logs))
    chunk_id = _extract_chunk_id(chunk, cfg, index)
    prompt_prefix = cfg.get("prompt_prefix", "")
    prompt_suffix = cfg.get("prompt_suffix", "\n\nNext log:")
    dataset_name = infer_dataset_name(cfg, cfg.get("active_split_path"))

    events: List[EventRecord] = []
    if len(logs) < 2:
        return events

    detector_mode = resolve_detector_event_mode(dataset_name, label_granularity)
    for target_idx in range(1, len(logs)):
        prefix_logs = [logs[target_idx - 1]]
        prompt = build_prefix_prompt(prefix_logs, prompt_prefix, prompt_suffix)
        if detector_mode == "session_aggregate":
            target_label = 0
            supervise = False
        else:
            target_label = labels[target_idx] if label_granularity == "event" else labels[0]
            supervise = detector_mode == "autoregressive_all"
        events.append(
            EventRecord(
                chunk_id=chunk_id,
                event_index=target_idx,
                prefix_logs=prefix_logs,
                target_log=logs[target_idx],
                target_label=target_label,
                prompt=prompt,
                supervise=supervise,
            )
        )
    return events


def resolve_selection_event_mode(dataset_name: str, label_granularity: str) -> str:
    """Choose neuron-selection event construction for a dataset."""

    dataset_name = dataset_name.lower()
    if dataset_name in SEQUENCE_LEVEL_DATASETS:
        return "last_only"
    if dataset_name in LINE_LEVEL_DATASETS:
        return "branch_point"
    if label_granularity == "chunk":
        return "last_only"
    return "autoregressive"


def resolve_detector_event_mode(dataset_name: str, label_granularity: str) -> str:
    """Choose detector supervision mode for a dataset."""

    dataset_name = dataset_name.lower()
    if dataset_name in SEQUENCE_LEVEL_DATASETS:
        return "session_aggregate"
    if label_granularity == "chunk":
        return "session_aggregate"
    return "autoregressive_all"


class EventDataset(Dataset):
    """Selection-sample dataset used for branch-point probing."""

    def __init__(self, path: str, cfg: Dict[str, Any]) -> None:
        self.cfg = dict(cfg)
        self.cfg["active_split_path"] = path
        self.path = path
        log_status(f"Loading event chunks from {path}")
        if is_jsonl_chunk_file(path):
            log_status(f"Streaming JSONL index scan for selection: {path}")
            normal_indices, anomalous_indices, total = scan_chunk_index_buckets(path, self.cfg)
            log_status(
                f"Scanned chunk line indices: total={total} "
                f"normal_eligible={len(normal_indices)} anomaly_eligible={len(anomalous_indices)}"
            )
            anomaly_target = int(self.cfg.get("test_anomaly_chunks", 100))
            normal_target = int(self.cfg.get("test_normal_chunks", 100))
            selected_indices = sample_normal_anomaly_line_indices(
                normal_indices=normal_indices,
                anomalous_indices=anomalous_indices,
                normal_target=normal_target,
                anomaly_target=anomaly_target,
                seed=int(self.cfg.get("seed", 42)),
            )
            log_status(
                "Selection chunk sampling: "
                f"requested_anomaly={anomaly_target}, available_anomaly={len(anomalous_indices)}, "
                f"chosen_anomaly={sum(1 for i in selected_indices if i in set(anomalous_indices))}, "
                f"requested_normal={normal_target}, available_normal={len(normal_indices)}, "
                f"chosen_normal={sum(1 for i in selected_indices if i in set(normal_indices))}"
            )
            self.chunks = load_chunks_by_line_indices(path, selected_indices)
            self.selected_chunk_indices = list(selected_indices)
        else:
            self.chunks = load_data_file(path)
            log_status(f"Loaded event chunks: {len(self.chunks)}")
            selected_indices = self._resolve_selected_chunk_indices()
            self.selected_chunk_indices = list(selected_indices)
        log_status(
            f"Selection chunk subset ready: selected_chunks={len(self.selected_chunk_indices)} "
            f"from streaming={'yes' if isinstance(self.chunks, dict) else 'no'}"
        )
        self.selection_samples, self.selection_summary = build_selection_samples(
            chunks=self.chunks,
            cfg=self.cfg,
            selected_indices=self.selected_chunk_indices,
            path=self.path,
        )
        strategy_counts = self.selection_summary["strategy_counts"]
        log_status(
            "Selection sample summary: "
            f"raw_chunks={self.selection_summary['raw_chunk_count']}, "
            f"built_samples={self.selection_summary['num_samples']}, "
            f"hdfs_samples={strategy_counts['hdfs_last_log']}, "
            f"hadoop_samples={strategy_counts['hadoop_last_log']}, "
            f"bgl_normal_samples={strategy_counts['bgl_normal_last_log']}, "
            f"bgl_anomaly_samples={strategy_counts['bgl_anomaly_last_late_anomaly']}, "
            f"thunderbird_normal_samples={strategy_counts['thunderbird_normal_last_log']}, "
            f"thunderbird_anomaly_samples={strategy_counts['thunderbird_anomaly_last_late_anomaly']}, "
            f"skipped={self.selection_summary['num_skipped']}, "
            f"skip_reasons={self.selection_summary['skipped_reasons']}"
        )

    def _resolve_selected_chunk_indices(self) -> List[int]:
        """Sample a small normal/anomalous chunk subset for neuron selection."""

        anomaly_target = int(self.cfg.get("test_anomaly_chunks", 100))
        normal_target = int(self.cfg.get("test_normal_chunks", 100))
        if anomaly_target < 0 or normal_target < 0:
            raise ValueError("test_normal_chunks / test_anomaly_chunks must be non-negative.")

        anomalous_indices: List[int] = []
        normal_indices: List[int] = []
        for index, chunk in enumerate(self.chunks):
            logs = _extract_chunk_logs(chunk, self.cfg)
            if len(logs) < 2:
                continue
            labels, _ = _extract_chunk_labels(chunk, self.cfg, len(logs))
            if _is_anomalous_chunk(labels):
                anomalous_indices.append(index)
            else:
                normal_indices.append(index)

        rng = random.Random(int(self.cfg.get("seed", 42)))
        chosen_anomalous = (
            rng.sample(anomalous_indices, min(anomaly_target, len(anomalous_indices)))
            if anomaly_target > 0
            else []
        )
        chosen_normal = (
            rng.sample(normal_indices, min(normal_target, len(normal_indices)))
            if normal_target > 0
            else []
        )

        log_status(
            "Selection chunk sampling: "
            f"requested_anomaly={anomaly_target}, available_anomaly={len(anomalous_indices)}, "
            f"chosen_anomaly={len(chosen_anomalous)}, requested_normal={normal_target}, "
            f"available_normal={len(normal_indices)}, chosen_normal={len(chosen_normal)}"
        )
        return sorted(chosen_anomalous + chosen_normal)

    def __len__(self) -> int:
        return len(self.selection_samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.selection_samples[index]
        return {
            "chunk_id": str(sample.chunk_id),
            "chunk_index": int(sample.chunk_index),
            "event_index": int(sample.target_pos),
            "prefix_logs": list(sample.prefix_logs),
            "target_log": str(sample.target_log),
            "label": int(sample.label),
            "prompt": str(sample.prompt),
            "dataset_name": str(sample.dataset_name),
            "target_mode": str(sample.target_mode),
            "target_pos": int(sample.target_pos),
        }


class ChunkDataset(Dataset):
    """Chunk-level dataset that preserves the event sequence structure."""

    def __init__(
        self,
        path: str,
        cfg: Dict[str, Any],
        max_normal_chunks: Optional[int] = None,
        max_anomaly_chunks: Optional[int] = None,
        chunk_index_whitelist: Optional[Sequence[int]] = None,
        chunk_id_whitelist: Optional[Sequence[str]] = None,
    ) -> None:
        self.cfg = dict(cfg)
        self.path = str(resolve_path(path))
        self.cfg["active_split_path"] = self.path
        log_status(f"Loading chunk dataset from {path}")
        candidate_indices: List[Tuple[int, bool, str]] = []
        rng = random.Random(int(self.cfg.get("seed", 42)))

        if is_jsonl_chunk_file(path):
            if chunk_index_whitelist is not None:
                whitelist_indices = sorted({int(i) for i in chunk_index_whitelist})
                self.chunks = load_chunks_by_line_indices(path, whitelist_indices)
                log_status(f"ChunkDataset loaded {len(self.chunks)} whitelisted JSONL chunks by line index")
            elif chunk_id_whitelist is not None:
                self.chunks = load_chunks_by_id_whitelist(path, self.cfg, chunk_id_whitelist)
                log_status(f"ChunkDataset loaded {len(self.chunks)} whitelisted JSONL chunks by chunk_id")
            elif max_normal_chunks is not None or max_anomaly_chunks is not None:
                log_status(f"Streaming JSONL index scan for chunk sampling: {path}")
                normal_indices, anomalous_indices, total = scan_chunk_index_buckets(path, self.cfg)
                log_status(
                    f"Scanned chunk line indices: total={total} "
                    f"normal_eligible={len(normal_indices)} anomaly_eligible={len(anomalous_indices)}"
                )
                n_norm = max_normal_chunks if max_normal_chunks is not None else len(normal_indices)
                n_anom = max_anomaly_chunks if max_anomaly_chunks is not None else len(anomalous_indices)
                if max_normal_chunks is not None and len(normal_indices) < int(max_normal_chunks):
                    log_status(
                        "WARN: requested "
                        f"{int(max_normal_chunks)} normal chunks but only "
                        f"{len(normal_indices)} eligible; using all available."
                    )
                if max_anomaly_chunks is not None and len(anomalous_indices) < int(max_anomaly_chunks):
                    log_status(
                        "WARN: requested "
                        f"{int(max_anomaly_chunks)} anomalous chunks but only "
                        f"{len(anomalous_indices)} eligible; using all available."
                    )
                selected_indices = sample_normal_anomaly_line_indices(
                    normal_indices=normal_indices,
                    anomalous_indices=anomalous_indices,
                    normal_target=int(n_norm),
                    anomaly_target=int(n_anom),
                    seed=int(self.cfg.get("seed", 42)),
                )
                self.chunks = load_chunks_by_line_indices(path, selected_indices)
                log_status(
                    f"ChunkDataset loaded sampled JSONL chunks: {len(self.chunks)} "
                    f"(requested normal={n_norm}, anomaly={n_anom})"
                )
            else:
                log_status(
                    "WARN: loading full JSONL chunk split into memory; "
                    "set max_normal_chunks / max_anomaly_chunks for large datasets."
                )
                self.chunks = load_data_file(path)
            log_status(f"Loaded chunk dataset records in memory: {len(self.chunks)}")
            chunk_items = (
                sorted(self.chunks.items())
                if isinstance(self.chunks, dict)
                else [(index, chunk) for index, chunk in enumerate(self.chunks)]
            )
            for index, chunk in chunk_items:
                try:
                    logs = _extract_chunk_logs(chunk, self.cfg)
                    if len(logs) < 2:
                        continue
                    labels, _ = _extract_chunk_labels(chunk, self.cfg, len(logs))
                    is_anomalous = _is_anomalous_chunk(labels)
                    chunk_id = _extract_chunk_id(chunk, self.cfg, index)
                    candidate_indices.append((index, is_anomalous, chunk_id))
                except Exception:
                    continue
        else:
            self.chunks = load_data_file(path)
            log_status(f"Loaded chunk dataset: {len(self.chunks)}")
            for index, chunk in enumerate(self.chunks):
                try:
                    logs = _extract_chunk_logs(chunk, self.cfg)
                    if len(logs) < 2:
                        continue
                    labels, _ = _extract_chunk_labels(chunk, self.cfg, len(logs))
                    is_anomalous = _is_anomalous_chunk(labels)
                    chunk_id = _extract_chunk_id(chunk, self.cfg, index)
                    candidate_indices.append((index, is_anomalous, chunk_id))
                except Exception:
                    continue

            if chunk_index_whitelist is not None:
                whitelist = set(int(i) for i in chunk_index_whitelist)
                candidate_indices = [(i, a, cid) for i, a, cid in candidate_indices if i in whitelist]
                log_status(f"ChunkDataset restricted to whitelist: {len(candidate_indices)} chunks")
            elif chunk_id_whitelist is not None:
                whitelist = set(str(x) for x in chunk_id_whitelist)
                candidate_indices = [(i, a, cid) for i, a, cid in candidate_indices if cid in whitelist]
                log_status(f"ChunkDataset restricted to chunk_id whitelist: {len(candidate_indices)} chunks")
            elif max_normal_chunks is not None or max_anomaly_chunks is not None:
                normal_indices = [i for i, is_anom, _ in candidate_indices if not is_anom]
                anomaly_indices = [i for i, is_anom, _ in candidate_indices if is_anom]
                n_norm = max_normal_chunks if max_normal_chunks is not None else len(normal_indices)
                n_anom = max_anomaly_chunks if max_anomaly_chunks is not None else len(anomaly_indices)
                if max_normal_chunks is not None and len(normal_indices) < int(max_normal_chunks):
                    log_status(
                        "WARN: requested "
                        f"{int(max_normal_chunks)} normal chunks but only "
                        f"{len(normal_indices)} eligible; using all available."
                    )
                if max_anomaly_chunks is not None and len(anomaly_indices) < int(max_anomaly_chunks):
                    log_status(
                        "WARN: requested "
                        f"{int(max_anomaly_chunks)} anomalous chunks but only "
                        f"{len(anomaly_indices)} eligible; using all available."
                    )
                chosen_normal = rng.sample(normal_indices, min(n_norm, len(normal_indices)))
                chosen_anomaly = rng.sample(anomaly_indices, min(n_anom, len(anomaly_indices)))
                selected_indices = set(chosen_normal + chosen_anomaly)
                candidate_indices = [(i, a, cid) for i, a, cid in candidate_indices if i in selected_indices]
                log_status(
                    f"ChunkDataset limited to {len(candidate_indices)} chunks "
                    f"({len(chosen_normal)} normal, {len(chosen_anomaly)} anomalous)"
                )

        self.records: List[Dict[str, Any]] = []
        max_events_per_chunk = int(self.cfg.get("chunk_max_events_per_chunk", 0) or 0)
        for index, _, _ in candidate_indices:
            chunk = self.chunks[index]
            logs = _extract_chunk_logs(chunk, self.cfg)
            if len(logs) < 2:
                continue
            labels, label_granularity = _extract_chunk_labels(chunk, self.cfg, len(logs))
            chunk_id = _extract_chunk_id(chunk, self.cfg, index)
            dataset_name = infer_dataset_name(self.cfg, self.path)
            detector_mode = resolve_detector_event_mode(dataset_name, label_granularity)
            chunk_label = resolve_chunk_session_label(labels, label_granularity)
            if detector_mode == "session_aggregate":
                event_rows = [
                    {
                        "event_index": target_idx,
                        "label": 0,
                        "supervise": False,
                    }
                    for target_idx in range(1, len(logs))
                ]
            else:
                event_rows = [
                    {
                        "event_index": target_idx,
                        "label": labels[target_idx] if label_granularity == "event" else labels[0],
                        "supervise": True,
                    }
                    for target_idx in range(1, len(logs))
                ]
            if max_events_per_chunk > 0 and len(event_rows) > max_events_per_chunk:
                event_rows = event_rows[:max_events_per_chunk]
            self.records.append(
                {
                    "chunk_id": chunk_id,
                    "chunk_index": index,
                    "chunk_label": chunk_label,
                    "events": event_rows,
                }
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        chunk = self.chunks[int(record["chunk_index"])]
        logs = _extract_chunk_logs(chunk, self.cfg)
        events = []
        narrow = resolve_prompt_extract_option_for_stage(self.cfg, "calc") == "legacy"
        for event_ref in record["events"]:
            target_idx = int(event_ref["event_index"])
            prompt = build_circuit_extraction_prompt(
                self.cfg,
                full_chunk_logs=logs,
                target_idx=target_idx,
                for_calc_stage=True,
                chunk_legacy_narrow_context=narrow,
                rng_salt=f"{record['chunk_id']}\t{target_idx}",
            )
            events.append(
                {
                    "event_index": target_idx,
                    "prompt": prompt,
                    "label": int(event_ref["label"]),
                    "target_log": logs[target_idx],
                    "supervise": bool(event_ref["supervise"]),
                }
            )
        return {
            "chunk_id": str(record["chunk_id"]),
            "chunk_label": int(record["chunk_label"]),
            "events": events,
        }


def collate_event_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a batch of event records without tokenizing yet."""

    return {
        "chunk_ids": [item["chunk_id"] for item in batch],
        "event_indices": [item["event_index"] for item in batch],
        "prompts": [item["prompt"] for item in batch],
        "target_logs": [item["target_log"] for item in batch],
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
    }


def collate_chunk_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Flatten chunk event sequences and keep reconstruction metadata."""

    chunk_ids: List[str] = []
    sequence_lengths: List[int] = []
    flat_prompts: List[str] = []
    flat_labels: List[int] = []
    flat_supervision_mask: List[bool] = []
    flat_event_indices: List[int] = []
    flat_chunk_ids: List[str] = []
    target_logs: List[str] = []
    chunk_labels: List[int] = []

    for chunk in batch:
        events = chunk["events"]
        chunk_ids.append(chunk["chunk_id"])
        chunk_labels.append(int(chunk.get("chunk_label", 0)))
        sequence_lengths.append(len(events))
        for event in events:
            flat_prompts.append(event["prompt"])
            flat_labels.append(int(event["label"]))
            flat_supervision_mask.append(bool(event["supervise"]))
            flat_event_indices.append(int(event["event_index"]))
            flat_chunk_ids.append(str(chunk["chunk_id"]))
            target_logs.append(event["target_log"])

    return {
        "chunk_ids": chunk_ids,
        "sequence_lengths": sequence_lengths,
        "flat_prompts": flat_prompts,
        "flat_labels": torch.tensor(flat_labels, dtype=torch.long),
        "flat_supervision_mask": torch.tensor(flat_supervision_mask, dtype=torch.bool),
        "flat_event_indices": flat_event_indices,
        "flat_chunk_ids": flat_chunk_ids,
        "target_logs": target_logs,
        "chunk_labels": torch.tensor(chunk_labels, dtype=torch.long),
    }


def infer_input_dim(feature_sequences: Sequence[Dict[str, Any]]) -> int:
    """Infer the detector input dimension from precomputed features."""

    for item in feature_sequences:
        features = item.get("features")
        if isinstance(features, torch.Tensor) and features.numel() > 0:
            return int(features.shape[-1])
    raise ValueError("Unable to infer feature dimension from empty feature sequences.")


class FeatureChunkDataset(Dataset):
    """Dataset backed by precomputed chunk-level feature sequences."""

    def __init__(self, sequences: Sequence[Dict[str, Any]]) -> None:
        self.sequences = list(sequences)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.sequences[index]


def collate_feature_chunk_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad variable-length chunk feature sequences for detector training."""

    if not batch:
        raise ValueError("Empty batch is not supported.")

    batch_size = len(batch)
    max_len = max(int(item["features"].shape[0]) for item in batch)
    feature_dim = int(batch[0]["features"].shape[-1])

    features = torch.zeros(batch_size, max_len, feature_dim, dtype=torch.float32)
    labels = torch.zeros(batch_size, max_len, dtype=torch.long)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    supervision_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    chunk_ids: List[str] = []
    chunk_labels = torch.zeros(batch_size, dtype=torch.long)

    for row, item in enumerate(batch):
        seq_len = int(item["features"].shape[0])
        features[row, :seq_len] = item["features"].float()
        labels[row, :seq_len] = item["labels"].long()
        mask[row, :seq_len] = True
        supervision_mask[row, :seq_len] = item["supervision_mask"].bool()
        chunk_ids.append(str(item["chunk_id"]))
        chunk_labels[row] = int(item.get("chunk_label", 0))

    return {
        "chunk_ids": chunk_ids,
        "features": features,
        "labels": labels,
        "mask": mask,
        "supervision_mask": supervision_mask,
        "chunk_labels": chunk_labels,
    }


def split_flat_features_by_chunk(
    flat_features: torch.Tensor,
    flat_labels: torch.Tensor,
    flat_supervision_mask: torch.Tensor,
    chunk_ids: Sequence[str],
    sequence_lengths: Sequence[int],
    chunk_labels: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    """Rebuild flattened event features into chunk-level sequences."""

    sequences: List[Dict[str, Any]] = []
    cursor = 0
    for idx, (chunk_id, seq_len) in enumerate(zip(chunk_ids, sequence_lengths)):
        next_cursor = cursor + int(seq_len)
        row: Dict[str, Any] = {
            "chunk_id": chunk_id,
            "features": flat_features[cursor:next_cursor].cpu(),
            "labels": flat_labels[cursor:next_cursor].cpu(),
            "supervision_mask": flat_supervision_mask[cursor:next_cursor].cpu(),
        }
        if chunk_labels is not None:
            row["chunk_label"] = int(chunk_labels[idx])
        sequences.append(row)
        cursor = next_cursor
    return sequences


def preview_dataset_schema(path: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a lightweight summary that helps inspect a split quickly."""

    inspection = inspect_data_file(path)
    first_chunk = inspection["first_chunk"]
    if not first_chunk:
        return {"path": path, "num_chunks": 0}

    logs = _extract_chunk_logs(first_chunk, cfg)
    labels, granularity = _extract_chunk_labels(first_chunk, cfg, len(logs))
    return {
        "path": path,
        "dataset_name": infer_dataset_name(cfg, path),
        "num_chunks": inspection["num_chunks"],
        "first_chunk_keys": sorted(first_chunk.keys()),
        "first_chunk_num_logs": len(logs),
        "first_chunk_num_labels": len(labels),
        "label_granularity": granularity,
        "selection_event_mode": resolve_selection_event_mode(infer_dataset_name(cfg, path), granularity),
        "detector_event_mode": resolve_detector_event_mode(infer_dataset_name(cfg, path), granularity),
    }


def detect_split_strategy(path: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize how a split will be interpreted by the data pipeline."""

    summary = preview_dataset_schema(path, cfg)
    return {
        "path": summary["path"],
        "dataset_name": summary["dataset_name"],
        "label_granularity": summary["label_granularity"],
        "selection_event_mode": summary["selection_event_mode"],
        "detector_event_mode": summary["detector_event_mode"],
        "num_chunks": summary["num_chunks"],
    }
