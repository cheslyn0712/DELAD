# Assets

Set `DELAD_ROOT=./assets` (default). Each subdirectory has a README with download instructions.

| Path | Content | Download |
|------|---------|----------|
| [datasets/bgl/](datasets/bgl/README.md) | BGL JSON splits | [LogHub — BGL](https://github.com/logpai/loghub) |
| [datasets/hdfs/](datasets/hdfs/README.md) | HDFS JSON splits | [LogHub — HDFS](https://github.com/logpai/loghub) |
| [datasets/thunderbird/](datasets/thunderbird/README.md) | Thunderbird JSON splits | [LogHub — Thunderbird](https://github.com/logpai/loghub) |
| [llm/gemma-2b/](llm/gemma-2b/README.md) | Gemma-2B weights | [HuggingFace — google/gemma-2-2b](https://huggingface.co/google/gemma-2-2b) |
| [llm/qwen3-8b/](llm/qwen3-8b/README.md) | Qwen3-8B weights | [HuggingFace — Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| [llm/qwen3-14b/](llm/qwen3-14b/README.md) | Qwen3-14B weights | [HuggingFace — Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) |
| [transcoders/gemma-2b-plt/](transcoders/gemma-2b-plt/README.md) | Gemma transcoder | [circuit-tracer](https://github.com/safety-research/circuit-tracer) |
| [transcoders/qwen3-8b/](transcoders/qwen3-8b/README.md) | Qwen3-8B transcoder | [circuit-tracer](https://github.com/safety-research/circuit-tracer) |
| [transcoders/qwen3-14b/](transcoders/qwen3-14b/README.md) | Qwen3-14B transcoder | [circuit-tracer](https://github.com/safety-research/circuit-tracer) |

```bash
python scripts/check_setup.py
```

All LLM checkpoints used in the paper evaluation will be open-sourced after acceptance.
