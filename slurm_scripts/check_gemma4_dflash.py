#!/usr/bin/env python3
"""Gate checks for the Gemma-4-31B DFlash smoke (plan: [30-09-26] gemma4-dflash-smoke).

Subcommands (each exits non-zero on the first failed gate):

  static    CPU only. Validates the run YAMLs against the SpecForge algorithm
            contract, builds each draft on the meta device, checks it against
            the target config, and checks every smoke row's loss mask against
            the exact supervised text written by build_gemma4_dflash_smoke.py.
  features  After capture. Loads the feature files, checks keys, shapes,
            dtype, finiteness, and that input_ids / loss_mask are exactly what
            the parser produces for some smoke row.
  curve     After training. Parses ``step N: {...}`` lines from a job log and
            applies the overfit pass criteria.

Usage (cluster, from the repo root, SpecForge venv):
  python scripts/check_gemma4_dflash.py static
  python scripts/check_gemma4_dflash.py features data/gemma4-dflash-smoke/hidden_states/overfit
  python scripts/check_gemma4_dflash.py curve /shared/phucnlt2/job-logs/g4-dflash-train_123.out
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import warnings

SMOKE_DIR = "data/gemma4-dflash-smoke"
CONFIG_DIR = "configs/specforge"
RUN_CONFIGS = (
    "gemma4-dflash-smoke-overfit.yaml",
    "gemma4-dflash2-smoke-overfit.yaml",
    "gemma4-dflash-smoke-long.yaml",
)
SPLIT_MAX_LENGTH = {"overfit": 2048, "heldout": 2048, "long": 10240}
FEATURE_KEYS = {"input_ids", "loss_mask", "hidden_states"}

WALK_KEY = "dflash/hard_label/walk_accepted_length"


def _fail(message: str) -> None:
    print(f"FAIL: {message}")
    sys.exit(1)


def _ok(message: str) -> None:
    print(f"ok    {message}")


def _tokenizer(cfg):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(cfg.model.target_model_path)


def _parser(tokenizer, template_name: str):
    from specforge.data.parse import GeneralParser
    from specforge.data.template import TEMPLATE_REGISTRY

    return GeneralParser(tokenizer, TEMPLATE_REGISTRY.get(template_name))


def _parse_rows(parser, split: str):
    rows = [json.loads(line) for line in open(os.path.join(SMOKE_DIR, f"{split}.jsonl"))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for row in rows:
            input_ids, loss_mask = parser.parse(
                row["text"], max_length=SPLIT_MAX_LENGTH[split], preformatted=True
            )
            yield row, input_ids, loss_mask


def check_static(_args) -> None:
    import torch
    from specforge.application.composition import resolve_run
    from specforge.config.schema import load_config
    from specforge.modeling.auto import AutoDraftModel
    from specforge.modeling.target.target_utils import (
        load_target_config,
        target_text_config,
        target_vocab_size,
    )
    from specforge.training.model_loading import resolve_draft_config

    cfg = None
    for name in RUN_CONFIGS:
        cfg = load_config(os.path.join(CONFIG_DIR, name))
        resolved = resolve_run(cfg)
        provider = resolved.algorithm.providers.model.draft_config
        draft_config = resolve_draft_config(cfg, provider=provider)
        target = target_text_config(load_target_config(cfg.model.target_model_path))

        with torch.device("meta"):
            draft = AutoDraftModel.from_config(draft_config, torch_dtype=torch.bfloat16)
        params = sum(p.numel() for p in draft.parameters())
        layer_ids = list(draft.target_layer_ids)
        expected_fc = len(layer_ids) * int(target.hidden_size)
        checks = {
            "hidden_size == target": draft_config.hidden_size == target.hidden_size,
            "vocab_size == target": draft_config.vocab_size == target_vocab_size(target),
            "num_target_layers == target": (
                draft_config.num_target_layers == target.num_hidden_layers
            ),
            "target_layer_ids in range": max(layer_ids) < target.num_hidden_layers,
            "fc in_features": draft.fc.in_features == expected_fc,
            "no draft embedding / lm_head": not any(
                "embed_tokens" in n or "lm_head" in n for n, _ in draft.named_parameters()
            ),
        }
        bad = [k for k, v in checks.items() if not v]
        if bad:
            _fail(f"{name}: {bad}")
        _ok(
            f"{name}: {type(draft).__name__}, {params / 1e9:.2f}B params, "
            f"block {draft.block_size}, layers {layer_ids}"
        )

    tokenizer = _tokenizer(cfg)
    mask_id = cfg.model.mask_token_id
    if tokenizer.convert_ids_to_tokens(mask_id) != "<mask>":
        _fail(f"mask_token_id {mask_id} is {tokenizer.convert_ids_to_tokens(mask_id)!r}")
    _ok(f"mask_token_id {mask_id} = <mask>")

    expect = {
        json.loads(line)["id"]: json.loads(line)["supervised"]
        for line in open(os.path.join(SMOKE_DIR, "expect.jsonl"))
    }
    parser = _parser(tokenizer, cfg.data.chat_template)
    for split in SPLIT_MAX_LENGTH:
        mismatches = []
        rows = 0
        for row, input_ids, loss_mask in _parse_rows(parser, split):
            rows += 1
            if len(input_ids) >= SPLIT_MAX_LENGTH[split]:
                mismatches.append((row["id"], "truncated"))
                continue
            supervised = tokenizer.decode(input_ids[loss_mask.bool()])
            if supervised != expect[row["id"]]:
                mismatches.append((row["id"], supervised[:80]))
        if mismatches:
            _fail(f"{split}: {len(mismatches)}/{rows} loss masks differ, e.g. {mismatches[:3]}")
        _ok(f"{split}: {rows} rows, loss mask == reply + <turn|>")

    sample = next(_parse_rows(parser, "overfit"))
    _, input_ids, loss_mask = sample
    start = int(loss_mask.nonzero()[0])
    print("\nprompt tail | supervised head of one overfit row:")
    print(
        repr(tokenizer.decode(input_ids[max(0, start - 12) : start])),
        "|",
        repr(tokenizer.decode(input_ids[start : start + 12])),
    )


def check_features(args) -> None:
    import torch
    from specforge.config.schema import load_config
    from specforge.runtime.data_plane.offline_reader import list_feature_files

    cfg = load_config(os.path.join(CONFIG_DIR, RUN_CONFIGS[0]))
    draft = json.load(open(cfg.model.draft_model_config))
    num_layers = len(draft["dflash_config"]["target_layer_ids"])
    hidden = draft["hidden_size"]
    split = os.path.basename(os.path.normpath(args.path))
    if split not in SPLIT_MAX_LENGTH:
        _fail(f"feature dir must end with one of {sorted(SPLIT_MAX_LENGTH)}")

    tokenizer = _tokenizer(cfg)
    parser = _parser(tokenizer, cfg.data.chat_template)
    expected = {
        tuple(input_ids.tolist()): loss_mask
        for _, input_ids, loss_mask in _parse_rows(parser, split)
    }

    files = list_feature_files(args.path)
    if not files:
        _fail(f"no feature files under {args.path}")
    if len(files) != len(expected):
        print(f"warn  {len(files)} feature files for {len(expected)} rows")
    total_bytes = 0
    pad_tokens = 0
    total_tokens = 0
    seen = set()
    layer_rms = torch.zeros(num_layers, dtype=torch.float64)
    for path in files[: args.limit]:
        record = torch.load(path, map_location="cpu", weights_only=True)
        if set(record) != FEATURE_KEYS:
            _fail(f"{path}: keys {sorted(record)} != {sorted(FEATURE_KEYS)}")
        input_ids = record["input_ids"].reshape(-1)
        loss_mask = record["loss_mask"].reshape(-1)
        hidden_states = record["hidden_states"]
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        if hidden_states.shape != (input_ids.numel(), num_layers * hidden):
            _fail(
                f"{path}: hidden_states {tuple(hidden_states.shape)}, "
                f"expected ({input_ids.numel()}, {num_layers * hidden})"
            )
        if hidden_states.dtype != torch.bfloat16:
            _fail(f"{path}: dtype {hidden_states.dtype}")
        if not torch.isfinite(hidden_states).all():
            _fail(f"{path}: non-finite hidden states")
        # SpecForge captures the padded batch as given: rows are right-padded to the
        # batch max. Strip the trailing pad run (it must carry no loss) before matching.
        real = input_ids.numel()
        while real > 0 and input_ids[real - 1].item() == tokenizer.pad_token_id:
            real -= 1
        if loss_mask[real:].any():
            _fail(f"{path}: loss_mask is set on trailing padding")
        pad_tokens += input_ids.numel() - real
        key = tuple(input_ids[:real].tolist())
        if key not in expected:
            _fail(f"{path}: input_ids match no {split} row (template/tokenizer drift)")
        if not torch.equal(loss_mask[:real].long(), expected[key].long()):
            _fail(f"{path}: loss_mask differs from the parser's mask")
        seen.add(key)
        per_layer = hidden_states.float().reshape(-1, num_layers, hidden)
        rms = per_layer.pow(2).mean(dim=(0, 2)).sqrt().double()
        if (rms == 0).any():
            _fail(f"{path}: a captured layer is all zeros, rms {rms.tolist()}")
        layer_rms += rms
        total_bytes += hidden_states.numel() * 2
        total_tokens += input_ids.numel()
    checked = min(len(files), args.limit)
    _ok(
        f"{checked}/{len(files)} files: keys, shape [T, {num_layers}x{hidden}], bf16, "
        f"finite, ids+mask match parser"
    )
    if checked == len(files) and len(seen) != len(expected):
        _fail(f"{len(seen)} distinct rows captured, expected {len(expected)}")
    print(
        f"      {len(seen)} distinct rows; padding {pad_tokens / total_tokens:.1%} of stored tokens"
    )
    print(
        "      mean RMS per captured layer:", [round(v, 2) for v in (layer_rms / checked).tolist()]
    )
    print(f"      {total_bytes / checked / 1e6:.1f} MB/sample in memory")


def check_curve(args) -> None:
    train, evals = [], []
    for line in open(args.log):
        match = re.match(r"step (\d+): (\{.*\})\s*$", line.strip())
        if not match:
            continue
        try:
            metrics = ast.literal_eval(match.group(2))
        except (ValueError, SyntaxError):
            continue
        step = int(match.group(1))
        if any(k.startswith("eval/") for k in metrics):
            evals.append((step, metrics))
            continue
        # The local trainer logs train metrics unprefixed; tracker backends add "train/".
        metrics = {k.removeprefix("train/"): v for k, v in metrics.items()}
        if "loss" in metrics:
            train.append((step, metrics))
    if len(train) < 10:
        _fail(f"only {len(train)} train log lines in {args.log}")

    def window(points, key):
        values = [m[key] for _, m in points if key in m]
        return sum(values) / len(values) if values else float("nan")

    if not any(WALK_KEY in m for _, m in train):
        keys = sorted({k for _, m in train for k in m if "accept" in k or "length" in k})
        _fail(f"{WALK_KEY} never logged; acceptance-like keys: {keys}")
    head, tail = train[:5], train[-5:]
    loss0, loss1 = window(head, "loss"), window(tail, "loss")
    walk0, walk1 = window(head, WALK_KEY), window(tail, WALK_KEY)
    print(
        f"steps {train[0][0]}..{train[-1][0]}  loss {loss0:.3f} -> {loss1:.3f}  "
        f"walk accepted length {walk0:.2f} -> {walk1:.2f}"
    )
    for step, metrics in evals:
        shown = {
            k: round(v, 3)
            for k, v in metrics.items()
            if k.startswith("eval/") and isinstance(v, float)
        }
        print(f"eval step {step}: {shown}")
    if any(m["loss"] != m["loss"] for _, m in train):
        _fail("NaN loss")
    if not loss1 < args.max_loss_ratio * loss0:
        _fail(
            f"loss fell only to {loss1 / loss0:.0%} of its start (gate < {args.max_loss_ratio:.0%})"
        )
    if not walk1 >= args.min_walk:
        _fail(f"train walk accepted length {walk1:.2f} < {args.min_walk} on memorized data")
    _ok("overfit gate passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("static")
    features = sub.add_parser("features")
    features.add_argument("path")
    features.add_argument("--limit", type=int, default=1000)
    curve = sub.add_parser("curve")
    curve.add_argument("log")
    curve.add_argument("--max-loss-ratio", type=float, default=0.3)
    curve.add_argument("--min-walk", type=float, default=6.0)
    args = parser.parse_args()
    {"static": check_static, "features": check_features, "curve": check_curve}[args.command](args)


if __name__ == "__main__":
    main()
