# Gemma-2B weights

Place the **Gemma-2B** HuggingFace model checkpoint in **this directory**.

## Required contents

At minimum:

- `config.json`
- Model weight files (`.safetensors` and/or `.bin`)
- Tokenizer files (`tokenizer.json`, `tokenizer_config.json`, etc.)

## Suggested source

- HuggingFace: `google/gemma-2-2b` (or the exact 2B checkpoint used in the evaluation)

## Layout after download

```text
assets/llm/gemma-2b/
├── README.md
├── config.json
├── model*.safetensors
└── tokenizer*
```

Official checkpoints used in the paper evaluation will be open-sourced after acceptance. Until then, use the HuggingFace links above or see [config/README.md](../../config/README.md).
