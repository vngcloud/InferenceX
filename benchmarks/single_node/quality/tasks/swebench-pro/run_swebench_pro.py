#!/usr/bin/env python3
"""Run mini-SWE-agent on SWE-bench Pro instances loaded from our generated instances.yaml.

This is a thin wrapper around minisweagent.run.extra.swebench that loads instances
from the YAML file produced by helper_code/generate_sweagent_instances.py (which
already contains the correct jefzda/sweap-images:<tag> image_name per instance)
instead of relying on the standard SWE-bench DATASET_MAPPING + default image-name
formula (which does not apply to SWE-bench Pro).

Usage:
    python tasks/swebench-pro/run_swebench_pro.py \
        --instances-path SWE-bench_Pro-os/SWE-agent/data/instances.yaml \
        --output jobs/<RUN_ID>/swebench-pro/preds \
        --config tasks/swebench-pro/swebench_pro.yaml \
        --model openai/z-ai/glm-5.2 \
        --workers 4 \
        --slice 0:3
"""

import argparse
import concurrent.futures
import json
import random
import re
import time
import traceback
from pathlib import Path

import yaml
from rich.live import Live

from minisweagent.run.extra.swebench import filter_instances, process_instance
from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager
from minisweagent.utils.log import add_file_handler, logger


def load_instances_from_yaml(path: Path) -> list[dict]:
    """Load instances from the YAML produced by generate_sweagent_instances.py.

    Each YAML entry has: image_name, problem_statement, instance_id, base_commit, repo_name.
    We re-shape into the dict shape that minisweagent.run.extra.swebench expects:
        instance_id, problem_statement, image_name, base_commit, repo.
    """
    entries = yaml.safe_load(path.read_text())
    instances = []
    for e in entries:
        instances.append(
            {
                "instance_id": e["instance_id"],
                "problem_statement": e["problem_statement"],
                "image_name": e["image_name"],
                "base_commit": e.get("base_commit", ""),
                "repo": e.get("repo_name", ""),
            }
        )
    return instances


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instances-path", required=True, help="Path to instances.yaml")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--config", required=True, help="Path to agent config yaml")
    parser.add_argument("--model", default=None, help="Model name (e.g. openai/z-ai/glm-5.2)")
    parser.add_argument("--model-class", default=None, help="Model class shortcut or import path")
    parser.add_argument("--environment-class", default=None, help="Environment class (docker/singularity)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--slice", default="", help="Slice spec e.g. 0:3")
    parser.add_argument("--filter", default="", help="Regex filter on instance_id")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--redo-existing", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent.log")

    logger.info(f"Loading instances from {args.instances_path}")
    instances = load_instances_from_yaml(Path(args.instances_path))
    logger.info(f"Loaded {len(instances)} instances")

    instances = filter_instances(
        instances, filter_spec=args.filter, slice_spec=args.slice, shuffle=args.shuffle
    )

    if not args.redo_existing and (output_path / "preds.json").exists():
        existing = list(json.loads((output_path / "preds.json").read_text()).keys())
        logger.info(f"Skipping {len(existing)} existing instances")
        instances = [i for i in instances if i["instance_id"] not in existing]

    logger.info(f"Running on {len(instances)} instances...")

    config = yaml.safe_load(Path(args.config).read_text())
    if args.environment_class is not None:
        config.setdefault("environment", {})["environment_class"] = args.environment_class
    if args.model is not None:
        config.setdefault("model", {})["model_name"] = args.model
    if args.model_class is not None:
        config.setdefault("model", {})["model_class"] = args.model_class

    progress_manager = RunBatchProgressManager(len(instances), output_path / f"exit_statuses_{time.time()}.yaml")

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                iid = futures[future]
                logger.error(f"Error in future for instance {iid}: {e}", exc_info=True)
                progress_manager.on_uncaught_exception(iid, e)

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_instance, inst, output_path, config, progress_manager): inst["instance_id"]
                for inst in instances
            }
            try:
                process_futures(futures)
            except KeyboardInterrupt:
                logger.info("Cancelling pending jobs. Press ^C again to exit immediately.")
                for f in futures:
                    if not f.running() and not f.done():
                        f.cancel()
                process_futures(futures)


if __name__ == "__main__":
    main()
