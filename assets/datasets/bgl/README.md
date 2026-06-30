# BGL dataset

Place the BGL benchmark files in **this directory**.

## Required files

| File | Description |
|------|-------------|
| `train.json` | Training split (chunked log sequences with labels) |
| `val.json` | Validation split |
| `test.json` | Test split |

## Source

Download from [LogHub](https://github.com/logpai/loghub) (BGL benchmark). Preprocess into the three JSON splits above with a temporal train/validation/test partition.

## Layout after download

```text
assets/datasets/bgl/
├── README.md      (this file)
├── train.json
├── val.json
└── test.json
```
