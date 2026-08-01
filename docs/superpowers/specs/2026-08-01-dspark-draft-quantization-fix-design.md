# DSpark Draft Quantization Fix

[English](2026-08-01-dspark-draft-quantization-fix-design.md) | [中文](2026-08-01-dspark-draft-quantization-fix-design_zh.md)

## Problem

The GLM-5.2 W4AFP8 agentic recipe leaves the draft quantization unset. SGLang therefore inherits the target's `w4afp8` quantization for the BF16 DSpark checkpoint. The resulting draft proposals have zero acceptance under both TP8 and DP4/EP8.

## Design

Keep SGLang's general inheritance behavior unchanged. Add `--speculative-draft-model-quantization unquant` only to the GLM-5.2 DSpark recipe so the W4AFP8 target remains quantized while the draft loads in its declared BF16 dtype.

Add a recipe regression test that executes the real launcher with controlled shell stubs and asserts the generated SGLang command carries the explicit draft override. Validate the fix with the focused automated test and a local H200 DP4/EP8 CUDA-graph smoke request, reading acceptance directly from server metrics/logs.

## Success Criteria

- The regression test fails before the recipe change and passes afterward.
- The generated command keeps target `--quantization w4afp8` and adds draft `unquant`.
- A local production-path smoke test reports speculative acceptance above zero.

