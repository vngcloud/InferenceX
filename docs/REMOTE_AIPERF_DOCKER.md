# Remote agentic-replay: running AIPerf from a pre-built image

How the remote-replay benchmark client runs, why it re-installs AIPerf on every
job today, and the future-work plan for using a pre-built AIPerf image on the
`benchmark-client` runner to skip that install.

## How the remote path runs the client

For a remote agentic-replay config (one with a `remote:` block in
`.github/configs/nvidia-master.yaml`), the model is already served on a separate
host (e.g. `http://192.168.4.13:8000`). The CI job on the `benchmark-client`
runner does **not** serve anything — it only runs the AIPerf client against those
remote endpoints.

The client is launched by [`runners/launch_remote.sh`](../runners/launch_remote.sh),
which uses the config's top-level `image:` purely as the **container that hosts
the client orchestration**:

```bash
docker run --rm --network host \
  -v "$GITHUB_WORKSPACE:/workspace" -w /workspace \
  --entrypoint bash "$IMAGE" \
  benchmarks/single_node/agentic/_remote_replay.sh
```

So in the remote path `image:` is the *client runtime*, not a server image. Today
that image is the serving image (e.g. `vllm/vllm-openai:...`). Inside it,
[`_remote_replay.sh`](../benchmarks/single_node/agentic/_remote_replay.sh) calls
`install_agentic_deps` (in [`benchmark_lib.sh`](../benchmarks/benchmark_lib.sh)),
which pip-installs AIPerf from the `utils/aiperf-mooncake` submodule on **every
job**. The slow part is the editable install of AIPerf plus its
transformers-from-git dependency.

## The `aiperf-docker-image` config option

The config schema and CI plumbing already carry an optional pre-built-image name
end to end:

- `.github/configs/nvidia-master.yaml` — `remote.aiperf-docker-image: <name:tag>`
- `utils/matrix_logic/validation.py` — `RemoteConfig.aiperf_docker_image`
- `.github/workflows/e2e-tests.yml` → `benchmark-tmpl.yml` — passed through as the
  `AIPERF_DOCKER_IMAGE` env var on the runner
- `benchmark_lib.sh` `install_agentic_deps` — when `AIPERF_DOCKER_IMAGE` is set and
  the image exists locally, it skips the pip install and marks the run to invoke
  AIPerf via `docker run <image>` instead.

**Known limitation (not yet wired end to end).** With the current
`launch_remote.sh`, the whole orchestration already runs *inside* the top-level
`image:` container, so that `docker run <aiperf-image>` would be a **nested**
docker call (Docker-in-Docker). That needs a `docker` CLI inside the serving
image and the host's `/var/run/docker.sock` mounted into the container — neither
of which `launch_remote.sh` sets up. Until that is addressed, leave
`aiperf-docker-image` unset so the runner keeps the pip-install path.

## Why not just point `image:` at the AIPerf image

This is the clean idea: since `image:` is the client runtime in the remote path,
set it to an AIPerf image that already has the client installed, and drop the
per-job install entirely. It would also avoid pulling the heavy vLLM image onto
the `benchmark-client` runner, which never serves a model in this path.

The blocker is *which* AIPerf image. `make docker` in `utils/aiperf-mooncake`
builds the default `runtime` target, which is **distroless**
([Dockerfile](../utils/aiperf-mooncake/Dockerfile) `runtime` stage): it ships only
`/bin/bash`, the AIPerf venv, and ffmpeg. It has no `mkdir`, `timeout`, `tee`,
`id`, `git`, `curl`, or `sleep`. Its `ENTRYPOINT ["/bin/bash", "-c"]` is built to
run a single `aiperf …` command string.

But `image:` has to host the **whole** orchestration, not just AIPerf:
`_remote_replay.sh` needs `mkdir`/`timeout`/`tee` and `python3` for result
aggregation and `analyze_benchmark_distributions.py`; the pre-check and pip paths
in `benchmark_lib.sh` need `curl`/`sleep`/`git`. The distroless image would fail
on the first line. It is perfect for running a single AIPerf command, and
unusable as the orchestration host.

## Future work (decided: defer)

Preferred direction, to avoid pulling the unused vLLM image on the remote client
runner:

1. Build a **full** AIPerf image instead of the distroless `runtime` target —
   e.g. base it on the Dockerfile's `test`/`local-dev` (Debian) stage, or add
   `coreutils`, `git`, and `curl` to a runtime variant. It must have AIPerf
   pre-installed plus the shell utilities and `python3` the orchestration uses.
2. Point the remote config's top-level `image:` at that full AIPerf image.
3. Add a one-line "AIPerf already installed → skip the slow editable install"
   bypass to `install_agentic_deps` (mirroring the reuse check `ensure_aiperf`
   already has). AIPerf then runs directly in the container — no nested docker,
   no Docker-in-Docker, results identical to the pip path since it is the same
   AIPerf build.

This is deferred for now. Until it lands, remote-replay configs continue to use
the serving `image:` and pip-install AIPerf per job; leave `aiperf-docker-image`
unset.
