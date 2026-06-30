# Thunderbird dataset

Place the Thunderbird benchmark files in **this directory**.

## Required files

| File | Description |
|------|-------------|
| `train.json` | Training split (chunked log sequences with labels) |
| `val.json` | Validation split |
| `test.json` | Test split |

## Source

Download from [LogHub](https://github.com/logpai/loghub) (Thunderbird benchmark). Preprocess into non-overlapping 128-message blocks with temporal train/val/test splits.

## Layout after download

```text
assets/datasets/thunderbird/
├── README.md
├── train.json
├── val.json
└── test.json
```
