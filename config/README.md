# Configuration

## Files in this directory

| File | Purpose |
|------|---------|
| [defaults.json](defaults.json) | Parameter keys (values intentionally `null` in this submission) |
| [models.json](models.json) | Model alias → relative paths under `assets/` |

Full hyperparameter values are supplied in a separate experiment config JSON (`path/to/released_config.json`).

## Usage

```bash
python main.py --config path/to/released_config.json --model-name qwen3-14b --dataset-key bgl
```

Paths in `models.json` are relative only:

```text
assets/llm/<model>/
assets/transcoders/<name>/
```

## Related downloads

| Item | Link |
|------|------|
| Model weights (HuggingFace) | [gemma-2-2b](https://huggingface.co/google/gemma-2-2b), [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) |
| Datasets | [LogHub](https://github.com/logpai/loghub) |
| Transcoders | [Anthropic circuit-tracing](https://www.anthropic.com/research/open-source-circuit-tracing), [`circuit-tracer`](https://github.com/safety-research/circuit-tracer) |

All LLM checkpoints used in the paper evaluation will be open-sourced after acceptance.
