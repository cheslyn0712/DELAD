# Qwen3-8B weights

Place the **Qwen3-8B** HuggingFace model checkpoint in **this directory**.

## Required contents

At minimum:

- `config.json`
- Model weight files (`.safetensors` and/or `.bin`)
- Tokenizer files

## Suggested source

- HuggingFace: `Qwen/Qwen3-8B`

## Layout after download

```text
assets/llm/qwen3-8b/
├── README.md
├── config.json
├── model*.safetensors
└── tokenizer*
```

Official checkpoints used in the paper evaluation will be open-sourced after acceptance. Until then, use the HuggingFace links above or see [config/README.md](../../config/README.md).
