# Gemma-2B transcoder

Place the **Gemma-2B** transcoder artifacts in **this directory**.

## Required contents

- Transcoder weight files compatible with [`circuit-tracer`](https://github.com/safety-research/circuit-tracer)
- A `features/` subdirectory with precomputed feature metadata used during probing

## Source

Build or download transcoders following [Anthropic's open-source circuit-tracing tools](https://www.anthropic.com/research/open-source-circuit-tracing) and the `circuit-tracer` documentation for Gemma-family models.

## Layout after download

```text
assets/transcoders/gemma-2b-plt/
├── README.md
├── features/
└── (transcoder checkpoint files)
```
