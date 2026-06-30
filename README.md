# DELAD

Event-level log anomaly detection with frozen LLM transcoder features and a GRU–MLP detector.


## Pipeline

1. **Feature selection** — probe frozen LLM on labeled train transitions; select Top-K features.
2. **Feature calculation** — extract activations for train, validation, and test.
3. **Detector training** — train on train, tune on validation, evaluate on test.

The `evaluation` mode reloads test metrics from a completed detector run; the `detector` stage already writes test metrics during training.

## Repository layout

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── main.py                 # CLI entry point
│
├── config/
│   ├── README.md           # Config schema
│   ├── defaults.json       # Parameter keys (values null in this submission)
│   └── models.json         # Model alias → relative paths under assets/
│
├── delad/                  # Core pipeline package
│   ├── __init__.py
│   ├── data.py             # Dataset loading, prompts, event construction
│   ├── detector.py         # GRU–MLP sequential event detector
│   ├── feature_labels.py   # Transcoder feature metadata & labels
│   ├── probe.py            # Frozen LLM probe & neuron selection
│   ├── session_aggregation.py  # HDFS session-level aggregation
│   ├── trainer.py          # Selection / calc / train / eval orchestration
│   └── utils.py            # Metrics, I/O, path helpers, logging
│
├── scripts/
│   └── check_setup.py      # Verify local asset layout
│
├── assets/                 # External artifacts (see each subfolder README)
│   ├── README.md
│   ├── datasets/
│   │   ├── bgl/README.md           → train.json, val.json, test.json
│   │   ├── hdfs/README.md          → train.json, val.json, test.json
│   │   └── thunderbird/README.md   → train.json, val.json, test.json
│   ├── llm/
│   │   ├── gemma-2b/README.md      → HuggingFace checkpoint files
│   │   ├── qwen3-8b/README.md      → HuggingFace checkpoint files
│   │   └── qwen3-14b/README.md     → HuggingFace checkpoint files
│   └── transcoders/
│       ├── gemma-2b-plt/README.md  → transcoder weights + features/
│       ├── qwen3-8b/README.md      → transcoder weights + features/
│       └── qwen3-14b/README.md     → transcoder weights + features/
│
├── results/                # Aggregated metrics (optional; see README)
│   └── README.md
│
└── outputs/                # Created at run time (not included in submission)
    └── circuit/<model>/<dataset>/...
        ├── artifacts/      # Selected neurons, feature banks
        ├── calc_features/  # train.json, val.json, test.json features
        └── rnn_<type>/     # Detector checkpoints & metrics/
```

Files under `assets/` marked with `→` are not included in this submission; follow the download links below or each subfolder README. Run outputs go under `outputs/`.

## Setup

Run commands from the repository root.

**Linux / macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export DELAD_ROOT=./assets
python scripts/check_setup.py
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:DELAD_ROOT = ".\assets"
python scripts/check_setup.py
```

Place datasets, LLM weights, and transcoders under `assets/` (see each subfolder README). Transcoders: [Anthropic circuit-tracing](https://www.anthropic.com/research/open-source-circuit-tracing).

## Configuration

[config/defaults.json](config/defaults.json) lists accepted parameter keys (values are `null` in this submission). See [config/README.md](config/README.md) for details.

### Artifacts

| Artifact | Location | Download |
|----------|----------|----------|
| **Model weights** | `assets/llm/` | HuggingFace: [google/gemma-2-2b](https://huggingface.co/google/gemma-2-2b), [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B). All LLM checkpoints used in the paper evaluation will be **open-sourced after acceptance**. |
| **Datasets** | `assets/datasets/` | [LogHub](https://github.com/logpai/loghub) (BGL, HDFS, Thunderbird); preprocessing notes in each dataset README. |
| **Transcoders** | `assets/transcoders/` | [Anthropic circuit-tracing](https://www.anthropic.com/research/open-source-circuit-tracing), [`circuit-tracer`](https://github.com/safety-research/circuit-tracer); see each transcoder README. |
| **Experiment config** | `config/` | Schema in-repo; full hyperparameter JSON: `path/to/released_config.json` (provided separately). |
| **Results** | `results/` | Per-run metrics under `outputs/circuit/<model>/<dataset>/.../metrics/`; optional aggregated copies in `results/`. |

Run with the experiment config file:

```bash
python main.py --config path/to/released_config.json \
  --model-name qwen3-14b --dataset-key bgl --mode full_pipeline
```

## Run

Full pipeline:

```bash
python main.py --config path/to/released_config.json \
  --model-name qwen3-14b --dataset-key bgl --mode full_pipeline --output-dir outputs
```

Stages:

```bash
python main.py --config path/to/released_config.json --model-name qwen3-14b --dataset-key bgl --mode select_neurons
python main.py --config path/to/released_config.json --model-name qwen3-14b --dataset-key bgl --mode calc_features
python main.py --config path/to/released_config.json --model-name qwen3-14b --dataset-key bgl --mode detector
python main.py --config path/to/released_config.json --model-name qwen3-14b --dataset-key bgl --mode evaluation
```

Main grid (3 models × 3 datasets):

```bash
for model in gemma-2b qwen3-8b qwen3-14b; do
  for dataset in bgl hdfs thunderbird; do
    python main.py --config path/to/released_config.json \
      --model-name "$model" --dataset-key "$dataset" --mode full_pipeline --output-dir outputs
  done
done
```

## Outputs

```text
outputs/circuit/<model>/<dataset>/.../metrics/test_metrics.json
outputs/circuit/<model>/<dataset>/.../metrics/val_metrics.json
```

Summaries may be copied to [results/](results/README.md). Event-level metrics: `event_metrics`; session-level (HDFS): `chunk_metrics`.
