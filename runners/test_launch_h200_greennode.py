#!/usr/bin/env python3
"""Every cluster:h200-greennode recipe must resolve to a bench script on disk.

launch_h200-greennode.sh derives the script name from the config's model-prefix,
precision, framework and spec-decoding method. A config whose name resolves to a
missing file fails at `docker run` with exit 127 after the image pull and model
load -- minutes into a job, with no signal at matrix-generation time. This test
is that signal.

Run: python3 runners/test_launch_h200_greennode.py
"""
import os
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Mirrors the case statement in launch_h200-greennode.sh.
SPEC_SUFFIX = {"mtp": "_mtp", "draft_model": "_specdec"}


def resolve(model_prefix, precision, framework, spec, scenario_type):
    subdir = "agentic/" if scenario_type == "agentic-coding" else "fixed_seq_len/"
    fw = "" if framework == "vllm" else f"_{framework}"
    base = f"benchmarks/single_node/{subdir}{model_prefix}_{precision}_h200{fw}"
    suffixed = f"{base}{SPEC_SUFFIX.get(spec, '')}.sh"
    # The launcher falls back to the unsuffixed name; so must this test.
    if os.path.isfile(os.path.join(REPO, suffixed)):
        return suffixed
    return f"{base}.sh"


def main():
    configs = yaml.safe_load(open(os.path.join(REPO, "configs/nvidia-master.yaml")))
    missing = []
    checked = 0
    for key, cfg in configs.items():
        if not isinstance(cfg, dict) or "h200-greennode" not in str(cfg.get("runner", "")):
            continue
        for scenario_type, entries in (cfg.get("scenarios") or {}).items():
            for entry in entries:
                for bmk in entry.get("search-space", []):
                    script = resolve(
                        cfg["model-prefix"], cfg["precision"], cfg["framework"],
                        bmk.get("spec-decoding", "none"), scenario_type,
                    )
                    checked += 1
                    if not os.path.isfile(os.path.join(REPO, script)):
                        missing.append(f"{key} -> {script}")

    assert checked, "no cluster:h200-greennode configs found -- test is not testing anything"
    assert not missing, "bench scripts missing for:\n  " + "\n  ".join(sorted(set(missing)))
    print(f"OK: {checked} h200-greennode benchmarks all resolve to an existing script")


if __name__ == "__main__":
    main()
    sys.exit(0)
