#!/usr/bin/env python3
"""Check local asset layout."""

from __future__ import annotations

import argparse
from pathlib import Path

from delad.utils import as_relative_path, resolve_path


DATASETS = ("bgl", "hdfs", "thunderbird")
MODELS = {
    "gemma-2b": ("llm/gemma-2b", "transcoders/gemma-2b-plt"),
    "qwen3-8b": ("llm/qwen3-8b", "transcoders/qwen3-8b"),
    "qwen3-14b": ("llm/qwen3-14b", "transcoders/qwen3-14b"),
}
SPLITS = ("train.json", "val.json", "test.json")


def _assets_root(explicit: str | None) -> Path:
    if explicit:
        return resolve_path(explicit)
    import os

    env = os.environ.get("DELAD_ROOT", "").strip()
    if env:
        return resolve_path(env)
    return resolve_path("assets")


def _display_path(path: Path) -> str:
    return as_relative_path(path)


def _check_dir(path: Path, label: str, errors: list[str]) -> bool:
    if path.is_dir():
        print(f"[ok] {label}: {_display_path(path)}")
        return True
    errors.append(f"Missing directory: {label}")
    print(f"[missing] {label}: {_display_path(path)}")
    return False


def _check_file(path: Path, label: str, errors: list[str]) -> bool:
    if path.is_file():
        print(f"[ok] {label}")
        return True
    errors.append(f"Missing file: {label}")
    print(f"[missing] {label}: {_display_path(path)}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Check DELAD asset layout")
    parser.add_argument("--assets-root", default=None)
    parser.add_argument("--model", default=None, choices=list(MODELS))
    parser.add_argument("--dataset", default=None, choices=DATASETS)
    args = parser.parse_args()

    root = _assets_root(args.assets_root)
    errors: list[str] = []
    print(f"Assets root: {_display_path(root)}")

    datasets = [args.dataset] if args.dataset else DATASETS
    for ds in datasets:
        ds_dir = root / "datasets" / ds
        if _check_dir(ds_dir, f"dataset/{ds}", errors):
            for split in SPLITS:
                _check_file(ds_dir / split, f"dataset/{ds}/{split}", errors)

    models = [args.model] if args.model else MODELS
    for model, (llm_rel, trans_rel) in ((m, MODELS[m]) for m in models):
        llm_dir = root / llm_rel
        trans_dir = root / trans_rel
        if _check_dir(llm_dir, f"llm/{model}", errors):
            _check_file(llm_dir / "config.json", f"llm/{model}/config.json", errors)
        if _check_dir(trans_dir, f"transcoder/{model}", errors):
            features_dir = trans_dir / "features"
            _check_dir(features_dir, f"transcoder/{model}/features", errors)

    if errors:
        print("\nSetup incomplete. See assets/README.md for download instructions.")
        return 1
    print("\nAll checked assets are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
