# HDFS dataset

Place the HDFS benchmark files in **this directory**.

## Required files

| File | Description |
|------|-------------|
| `train.json` | Training split (session-level chunks) |
| `val.json` | Validation split |
| `test.json` | Test split |

## Source

Download from [LogHub](https://github.com/logpai/loghub) (HDFS benchmark). Partition into session-level chunks with a temporal train/validation/test split. Each JSON file contains chunked log sequences with labels in the format expected by `delad/data.py`.

## Layout after download

```text
assets/datasets/hdfs/
├── README.md
├── train.json
├── val.json
└── test.json
```
