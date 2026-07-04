# Remote agentic-replay: running AIPerf from a pre-built image

How the remote-replay benchmark client runs, and how to use a pre-built full
AIPerf image on the `benchmark-client` runner to skip the per-job install.

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

## The `aiperf-docker-image` config option (deprecated, inert)

The config schema and CI plumbing carry an optional pre-built-image name end to
end:

- `.github/configs/nvidia-master.yaml` — `remote.aiperf-docker-image: <name:tag>`
- `utils/matrix_logic/validation.py` — `RemoteConfig.aiperf_docker_image`
- `.github/workflows/e2e-tests.yml` → `benchmark-tmpl.yml` — passed through as the
  `AIPERF_DOCKER_IMAGE` env var on the runner
- `benchmark_lib.sh` `install_agentic_deps` — when `AIPERF_DOCKER_IMAGE` is set and
  the image exists locally, it skips the pip install and marks the run to invoke
  AIPerf via `docker run <image>` instead.

**This field is inert and should not be used.** `runners/launch_remote.sh` passes
an explicit allowlist of env vars into the container (`RUN_ENV`), which does not
include `AIPERF_DOCKER_IMAGE` — so the value never reaches the job. Even if it
did, the whole orchestration already runs *inside* the top-level `image:`
container, so `docker run <aiperf-image>` would be a **nested** docker call
(Docker-in-Docker), which would need a `docker` CLI inside the container and the
host's `/var/run/docker.sock` mounted in — neither of which `launch_remote.sh`
sets up. Use the full-image approach below instead.

## The full-image approach (implemented)

Since `image:` is already the client runtime in the remote path, point it
directly at a pre-built AIPerf image and drop the per-job install entirely. This
also avoids pulling the heavy vLLM image onto the `benchmark-client` runner,
which never serves a model in this path.

`make docker` in `utils/aiperf-mooncake` builds the default `runtime` target,
which is **distroless** ([Dockerfile](../utils/aiperf-mooncake/Dockerfile)
`runtime` stage). It's built around `ENTRYPOINT ["/bin/bash", "-c"]` to run a
single `aiperf …` command, but it turns out to be enough to host the whole
orchestration too: on top of `/bin/bash`, the AIPerf venv, and `python3`, the
base distroless image ships busybox, which provides `mkdir`, `timeout`, `tee`,
`sleep`, `id`, and `wget`. The only orchestration dependency it's missing is
`curl` (used only for the endpoint reachability pre-check) and `git` (used only
by the pip-install path, which the full image skips). No rebuild needed:

1. **`_probe_endpoint`** ([benchmark_lib.sh](../benchmarks/benchmark_lib.sh))
   prefers `curl` and falls back to busybox `wget` when `curl` is absent.
2. **`install_agentic_deps`** short-circuits with `command -v aiperf` (mirroring
   the reuse check `ensure_aiperf` already has) — when the image already has
   `aiperf` on `PATH`, the pip install is skipped entirely. Set
   `AIPERF_FORCE_PIP_INSTALL=true` to force the source install anyway (e.g. to
   pick up submodule changes not yet baked into the image).
3. The remote config's top-level `image:` points at the pre-built AIPerf image
   (e.g. `aiperf:0.8.0`) instead of the serving image, and
   `remote.aiperf-docker-image` is removed.

AIPerf then runs directly in the container — no nested docker, no
Docker-in-Docker, results identical to the pip path since it's the same AIPerf
build. Rebuilding/re-tagging the image is only needed to pick up
`utils/aiperf-mooncake` submodule changes, since the full image pins whatever
AIPerf version was baked in at build time.
