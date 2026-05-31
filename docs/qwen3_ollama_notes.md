# qwen3:4b + Ollama Notes

Default model:

```yaml
default_model: qwen3:4b
fallback_model: qwen3:4b
```

The default branch uses one model only: `qwen3:4b` through Ollama. No second specialist model is used by default.

## Why the staged design matters

`qwen3:4b` is small enough that unrestricted multi-file coding is fragile. The runtime therefore constrains the task:

- one function or one class per generation call;
- deterministic runtime-supplied IO contract;
- plan manifest before code generation;
- JSON draft first;
- validation in temporary staging;
- deterministic domain fallback when the first draft misses the contract;
- repair from compact failure feedback;
- materialization only after checks pass.

## 6 GB VRAM profile

The RTX 3060 6 GB profile is intentionally conservative:

```yaml
num_ctx: 4096
think: false
keep_alive: 30s
```

Role-specific `num_predict` values are short: 768 for planning/review, 1280 for code-unit generation, and 1024 for repair. This reduces KV-cache pressure and avoids long thinking-only outputs that are hard to parse into strict JSON.

Recommended manual Ollama launch:

```bash
OLLAMA_NUM_PARALLEL=1 \
OLLAMA_MAX_LOADED_MODELS=1 \
OLLAMA_MAX_QUEUE=16 \
OLLAMA_FLASH_ATTENTION=1 \
OLLAMA_KV_CACHE_TYPE=q8_0 \
ollama serve
```

When the CLI starts Ollama itself, it applies the same conservative environment defaults.

## Structured outputs

Ollama structured output is used through the `/api/chat` `format` field with a JSON schema. The same schema is injected into the prompt to ground the model. Pydantic validates every response.

## PyTorch requests

For requests such as “write Q-learning code in Python PyTorch”, the first stable baseline is a standard-library-only, PyTorch-compatible scaffold. It reports compatibility in `artifacts` and `diagnostics` but does not import `torch`. This prevents failures in environments where torch is not installed or where type checking cannot resolve torch symbols.
