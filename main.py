"""DELAD entry point."""

from __future__ import annotations

import argparse
import copy
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from delad.data import (
    circuit_prompt_feature_dir_segment,
    infer_dataset_name,
    is_sequence_level_dataset,
    resolve_training_paradigm,
)
from delad.trainer import (
    run_detector_training,
    run_evaluation,
    run_feature_calculation,
    run_full_pipeline,
    run_neuron_selection,
)
from delad.utils import (
    configure_run_logging,
    ensure_dir,
    log_status,
    normalize_cfg_paths,
    pretty_print_config,
    repo_root,
    save_json,
)


def _default_assets_root() -> Path:
    env = os.environ.get("DELAD_ROOT", "").strip()
    if env:
        return Path(env)
    return repo_root() / "assets"


def _load_json_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")
    return payload


def _load_models_registry(assets_root: Path) -> Dict[str, Dict[str, Any]]:
    registry_path = repo_root() / "config" / "models.json"
    if registry_path.is_file():
        return _load_json_file(registry_path)
    return {
        "gemma-2b": {"llm_rel": "llm/gemma-2b", "trans_rel": "transcoders/gemma-2b-plt", "num_capture_layers": 26},
        "qwen3-8b": {"llm_rel": "llm/qwen3-8b", "trans_rel": "transcoders/qwen3-8b", "num_capture_layers": 36},
        "qwen3-14b": {"llm_rel": "llm/qwen3-14b", "trans_rel": "transcoders/qwen3-14b", "num_capture_layers": 36},
    }


def _merge_config(base: Dict[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if value is None:
            continue
        out[key] = value
    return out


def _base_cfg() -> Dict[str, Any]:
    defaults_path = repo_root() / "config" / "defaults.json"
    if defaults_path.is_file():
        return _load_json_file(defaults_path)
    return {}


def resolve_model_config(cfg: Dict[str, Any], models: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(cfg)
    key = str(out.get("model_name", "")).strip().lower()
    if key not in models:
        raise ValueError(f"Unknown model_name={out.get('model_name')!r}; expected one of {sorted(models)}")
    assets_root = Path(str(out["assets_root"]))
    mp = models[key]
    out["llm_path"] = str(assets_root / mp["llm_rel"])
    out["tokenizer_path"] = out["llm_path"]
    out["transcoder_path"] = str(assets_root / mp["trans_rel"])
    out["circuit_feature_dir"] = str(Path(out["transcoder_path"]) / "features")
    num_layers = int(mp.get("num_capture_layers", 36))
    requested = out.get("capture_layers")
    if not requested:
        out["capture_layers"] = list(range(num_layers))
    else:
        clipped = [int(x) for x in requested if int(x) < num_layers]
        if not clipped:
            raise ValueError(f"capture_layers has no valid indices for {key} (num_capture_layers={num_layers}).")
        out["capture_layers"] = clipped
    return out


def apply_run_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    prompt_key = str(out.get("selection_prompt_extract_option", out.get("prompt_extract_option", "A"))).upper()
    seg = circuit_prompt_feature_dir_segment(out, prompt_key=prompt_key)
    rnn = str(out.get("detector_recurrent_type", "gru")).lower()
    run_base = Path(out["output_dir"]) / "circuit" / out["model_name"] / out["dataset_name"] / seg
    feature_root = ensure_dir(run_base / "artifacts")
    rnn_dir = ensure_dir(run_base / f"rnn_{rnn}")
    paradigm = resolve_training_paradigm(out)
    agg_method = str(out.get("session_aggregation_method", "noisy_or")).strip().lower()
    if paradigm == "session_aggregate":
        run_dir = ensure_dir(rnn_dir / f"agg_{agg_method}")
    else:
        run_dir = rnn_dir
    calc_dir = ensure_dir(run_base / "calc_features")
    out["run_output_dir"] = str(run_dir)
    out["metrics_dir"] = str(ensure_dir(run_dir / "metrics"))
    out["feature_artifacts_dir"] = str(feature_root)
    out["selected_neurons_path"] = str(feature_root / "selected_neurons.json")
    out["selected_features_path"] = str(feature_root / "selected_features.json")
    out["selected_neuron_features_path"] = str(feature_root / "selected_neuron_features.json")
    out["selection_feature_bank_path"] = str(feature_root / "selection_feature_bank.json")
    out["calc_feature_dir"] = str(calc_dir)
    out["train_feature_json_path"] = str(calc_dir / "train.json")
    out["val_feature_json_path"] = str(calc_dir / "val.json")
    out["test_feature_json_path"] = str(calc_dir / "test.json")
    out["calc_manifest_path"] = str(feature_root / "calc_manifest.json")
    out["detector_ckpt_path"] = str(ensure_dir(run_dir / "artifacts") / "detector.pt")
    out["prompt_snapshot_path"] = str(Path(out["detector_ckpt_path"]).with_name("prompt_snapshot.json"))
    out["run_manifest_path"] = str(run_dir / "run_manifest.json")
    return out


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _base_cfg()
    if args.config:
        cfg = _merge_config(cfg, _load_json_file(Path(args.config)))
    cli_overlay = {k: v for k, v in vars(args).items() if v is not None and k != "config"}
    cfg = _merge_config(cfg, cli_overlay)

    assets_root = Path(str(cfg.get("assets_root") or _default_assets_root()))
    cfg["assets_root"] = str(assets_root)
    cfg["shared_root"] = str(assets_root)

    dataset_key = str(cfg.get("dataset_key", "") or "").strip().lower()
    if not cfg.get("dataset_dir"):
        if dataset_key:
            cfg["dataset_dir"] = str(assets_root / "datasets" / dataset_key)
        else:
            cfg["dataset_dir"] = str(assets_root / "datasets" / "bgl")

    dataset_dir = Path(str(cfg["dataset_dir"]))
    cfg["dataset_dir"] = str(dataset_dir)
    cfg["dataset_name"] = infer_dataset_name(cfg, str(dataset_dir))
    cfg["train_path"] = str(dataset_dir / "train.json")
    cfg["val_path"] = str(dataset_dir / "val.json")
    cfg["test_path"] = str(dataset_dir / "test.json")

    if not cfg.get("output_dir"):
        cfg["output_dir"] = "outputs"

    if is_sequence_level_dataset(cfg) and not cfg.get("session_aggregation_method"):
        cfg["session_aggregation_method"] = "noisy_or"

    models = _load_models_registry(assets_root)
    cfg = resolve_model_config(cfg, models)
    cfg = apply_run_paths(cfg)
    return normalize_cfg_paths(cfg)


def _ts_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_run_manifest(cfg: Dict[str, Any], *, status: str, error: Optional[str] = None) -> None:
    payload = {
        "status": status,
        "error": error,
        "updated_at": _ts_now(),
        "mode": str(cfg.get("mode", "full_pipeline")),
        "model_name": str(cfg.get("model_name", "")),
        "dataset_name": str(cfg.get("dataset_name", "")),
        "training_paradigm": resolve_training_paradigm(cfg),
        "detector_recurrent_type": str(cfg.get("detector_recurrent_type", "gru")),
        "session_aggregation_method": str(cfg.get("session_aggregation_method", "")),
    }
    save_json(payload, cfg["run_manifest_path"])


def run_delad(cfg: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(cfg.get("mode", "full_pipeline")).strip().lower()
    if mode == "select_neurons":
        return run_neuron_selection(cfg)
    if mode == "calc_features":
        return run_feature_calculation(cfg)
    if mode == "detector":
        return run_detector_training(cfg)
    if mode == "evaluation":
        return run_evaluation(cfg)
    return run_full_pipeline(cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DELAD experiment runner")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config overlay")
    parser.add_argument("--mode", type=str, default=None, choices=["full_pipeline", "select_neurons", "calc_features", "detector", "evaluation"])
    parser.add_argument("--assets-root", dest="assets_root", type=str, default=None)
    parser.add_argument("--dataset-key", dest="dataset_key", type=str, default=None, choices=["bgl", "hdfs", "thunderbird"])
    parser.add_argument("--dataset-dir", dest="dataset_dir", type=str, default=None)
    parser.add_argument("--output-dir", dest="output_dir", type=str, default=None)
    parser.add_argument("--model-name", dest="model_name", type=str, default=None, choices=["gemma-2b", "qwen3-8b", "qwen3-14b"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--top-k-neurons", dest="top_k_neurons", type=int, default=None)
    parser.add_argument("--detector-recurrent-type", dest="detector_recurrent_type", choices=["gru", "lstm"], default=None)
    parser.add_argument("--session-aggregation-method", dest="session_aggregation_method", choices=["max", "topk_mean", "noisy_or"], default=None)
    parser.add_argument("--optuna-max-trials", dest="optuna_max_trials", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = build_config(args)
    pretty_print_config(cfg)
    ensure_dir(Path(cfg["run_manifest_path"]).parent)
    log_path = ensure_dir(Path(cfg["run_output_dir"]) / "logs") / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    configure_run_logging(log_path)
    _write_run_manifest(cfg, status="started")
    try:
        result = run_delad(cfg)
        _write_run_manifest(cfg, status="completed")
        log_status(f"run completed mode={cfg.get('mode')}")
        return 0
    except Exception as exc:
        _write_run_manifest(cfg, status="failed", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1
    finally:
        configure_run_logging(None)


if __name__ == "__main__":
    raise SystemExit(main())
