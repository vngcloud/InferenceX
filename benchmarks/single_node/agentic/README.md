# Agentic single-node benchmarks

**MVP / experimental.** Nothing in this directory is an official InferenceX
benchmark. Results are not published on https://inferencex.com and are not
intended to be cited.

These launchers exist to develop and validate the agentic-coding scenario
type before it is promoted to first-class status. The scripts themselves
are best-effort and mainly serve as a reference implementation of how the
plumbing (env vars, scenario routing, result paths) should work. Specific
models and configs may be broken at any given time — multi-node in
particular is not yet first-class.

## Index

| Path | Purpose |
|---|---|
| `datasets/README.md` | Dataset inventory, active vs archived traces, and format-specific AIPerf source notes. |
| `../qwen3-4b-v4-weka_bf16_h200_vllm.sh` | Smoke-tested MiniMax Claude Code v4 Weka launcher template. |
