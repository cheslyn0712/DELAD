"""Offline transcoder feature metadata (peak tokens, logits, contexts)."""

from __future__ import annotations

import gzip
import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from delad.utils import resolve_path


@dataclass
class FeatureSummary:
    """Human-readable summary for one transcoder feature."""

    layer_idx: int
    feature_idx: int
    act_min: float = 0.0
    act_max: float = 0.0
    peak_tokens: List[str] = field(default_factory=list)
    top_logits: List[str] = field(default_factory=list)
    bottom_logits: List[str] = field(default_factory=list)
    top_contexts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_idx": int(self.layer_idx),
            "feature_idx": int(self.feature_idx),
            "act_min": float(self.act_min),
            "act_max": float(self.act_max),
            "peak_tokens": list(self.peak_tokens),
            "top_logits": list(self.top_logits),
            "bottom_logits": list(self.bottom_logits),
            "top_contexts": list(self.top_contexts),
        }


def _is_top_quantile_bucket(name: str) -> bool:
    """Return True for Gemma ``Top 1%`` and Qwen ``Top`` buckets."""

    normalized = str(name or "").strip().lower()
    if normalized == "top":
        return True
    return normalized.startswith("top ")


def _clean_token_text(raw: Any) -> str:
    text = str(raw or "").strip()
    return text.replace("\n", "⏎")


def _extract_peak_tokens_from_example(example: Mapping[str, Any], limit: int = 10) -> List[str]:
    tokens = example.get("tokens")
    if isinstance(tokens, list) and tokens:
        return [_clean_token_text(t) for t in tokens[:limit] if str(t).strip()]

    tokens_acts = example.get("tokens_acts_list")
    if isinstance(tokens_acts, list) and tokens_acts:
        out: List[str] = []
        for item in tokens_acts[:limit]:
            if isinstance(item, dict):
                tok = item.get("token") or item.get("text")
                if tok is not None:
                    out.append(_clean_token_text(tok))
            elif isinstance(item, (list, tuple)) and item:
                out.append(_clean_token_text(item[0]))
        return out
    return []


def _format_context_snippet(tokens: Sequence[str], peak_idx: int, window: int = 16) -> str:
    if not tokens:
        return ""
    lo = max(0, int(peak_idx) - window)
    hi = min(len(tokens), int(peak_idx) + window + 1)
    parts: List[str] = []
    for idx in range(lo, hi):
        tok = _clean_token_text(tokens[idx])
        if idx == peak_idx:
            parts.append(f"<<{tok}>>")
        else:
            parts.append(tok)
    snippet = "".join(parts)
    if len(snippet) > 180:
        return snippet[:177] + "..."
    return snippet


def _extract_top_contexts(examples: Sequence[Mapping[str, Any]], limit: int = 3) -> List[str]:
    contexts: List[str] = []
    for example in examples:
        tokens = example.get("tokens")
        if not isinstance(tokens, list) or not tokens:
            continue
        peak_idx = int(example.get("train_token_ind", len(tokens) // 2))
        snippet = _format_context_snippet(tokens, peak_idx)
        if snippet and snippet not in contexts:
            contexts.append(snippet)
        if len(contexts) >= limit:
            break
    return contexts


class FeatureExampleStore:
    """Read CircuitTracer ``features/`` archives (index + layer bin files)."""

    def __init__(self, feature_root: str | Path) -> None:
        self.feature_root = resolve_path(feature_root)
        self._index: Dict[str, Any] | None = None

    def _load_index(self) -> Dict[str, Any]:
        if self._index is not None:
            return self._index
        gz_path = self.feature_root / "index.json.gz"
        json_path = self.feature_root / "index.json"
        if gz_path.is_file():
            with gzip.open(gz_path, "rt", encoding="utf-8") as handle:
                self._index = json.load(handle)
        elif json_path.is_file():
            with json_path.open("r", encoding="utf-8") as handle:
                self._index = json.load(handle)
        else:
            raise FileNotFoundError(f"Missing feature index under {self.feature_root}")
        return self._index

    def _read_feature_record(self, layer_idx: int, feature_idx: int) -> Dict[str, Any] | None:
        index = self._load_index()
        layer_key = str(int(layer_idx))
        layer_meta = index.get(layer_key)
        if not isinstance(layer_meta, dict):
            return None
        filename = layer_meta.get("filename")
        offsets = layer_meta.get("offsets")
        if not filename or not isinstance(offsets, list):
            return None
        if feature_idx < 0 or feature_idx >= len(offsets):
            return None

        bin_path = self.feature_root / str(filename)
        if not bin_path.is_file():
            bin_path = Path(str(filename))
        if not bin_path.is_file():
            return None

        with bin_path.open("rb") as handle:
            handle.seek(int(offsets[feature_idx]))
            raw_len = struct.unpack("<I", handle.read(4))[0]
            payload = gzip.decompress(handle.read(raw_len))
        try:
            record = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return record if isinstance(record, dict) else None

    def get_summary(self, layer_idx: int, feature_idx: int) -> FeatureSummary:
        record = self._read_feature_record(layer_idx, feature_idx)
        if record is None:
            return FeatureSummary(layer_idx=int(layer_idx), feature_idx=int(feature_idx))

        quantiles = record.get("examples_quantiles") or record.get("quantiles") or []
        peak_tokens: List[str] = []
        top_contexts: List[str] = []
        top_examples: List[Mapping[str, Any]] = []

        if isinstance(quantiles, list):
            for quantile in quantiles:
                if not isinstance(quantile, dict):
                    continue
                quantile_name = str(quantile.get("quantile_name", "")).strip()
                examples = quantile.get("examples") or []
                if not isinstance(examples, list):
                    continue
                is_top_bucket = _is_top_quantile_bucket(quantile_name)
                if is_top_bucket and examples:
                    top_examples = [ex for ex in examples if isinstance(ex, dict)]
                    for example in top_examples[:5]:
                        peak_tokens.extend(_extract_peak_tokens_from_example(example, limit=10))
                    peak_tokens = list(dict.fromkeys(peak_tokens))[:10]
                    top_contexts = _extract_top_contexts(top_examples, limit=3)
                    break

        top_logits = [
            _clean_token_text(x)
            for x in (record.get("top_logits") or [])[:5]
            if str(x).strip()
        ]
        bottom_logits = [
            _clean_token_text(x)
            for x in (record.get("bottom_logits") or [])[:5]
            if str(x).strip()
        ]

        return FeatureSummary(
            layer_idx=int(layer_idx),
            feature_idx=int(feature_idx),
            act_min=float(record.get("act_min", 0.0) or 0.0),
            act_max=float(record.get("act_max", 0.0) or 0.0),
            peak_tokens=peak_tokens,
            top_logits=top_logits,
            bottom_logits=bottom_logits,
            top_contexts=top_contexts,
        )


def build_feature_description(summary: FeatureSummary) -> str:
    """Build a short offline description for dashboards / GPT prompts."""

    parts: List[str] = []
    if summary.peak_tokens:
        quoted = ", ".join(f"'{tok}'" for tok in summary.peak_tokens[:6])
        parts.append(f"peaks on {quoted}")
    if summary.top_logits:
        quoted = ", ".join(repr(tok) for tok in summary.top_logits[:5])
        parts.append(f"promotes {quoted}")
    if summary.top_contexts:
        parts.append(f"{len(summary.top_contexts)} top contexts available")
    if not parts and summary.top_logits:
        parts.append("no peak tokens; using promoted logits as fallback")
    return "; ".join(parts) if parts else "no offline metadata"


def format_feature_for_prompt(summary: FeatureSummary) -> str:
    """Compact one-line label used in selection exports."""

    layer_idx = int(summary.layer_idx)
    feature_idx = int(summary.feature_idx)
    if summary.peak_tokens:
        quoted = ",".join(f"'{tok}'" for tok in summary.peak_tokens[:3])
        return f"L{layer_idx}#{feature_idx} peaks {quoted}"
    if summary.top_logits:
        quoted = ",".join(repr(tok) for tok in summary.top_logits[:3])
        return f"L{layer_idx}#{feature_idx} logits {quoted}"
    return f"L{layer_idx}#{feature_idx}"
