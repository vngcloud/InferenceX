# InferenceX Benchmark Troubleshooting

> **Append-only log.** Each entry documents a real failure encountered during a
> benchmark dispatch, its root cause, and the fix. Add new entries at the bottom.
> Format: `## [date] Title` → Symptom → Root cause → Fix → Prevention.

---

## [2026-08-15] Runner label mismatch — jobs stuck in `queued` forever

**Symptom:** All 3 agentic jobs show `status: queued` with `runner_name: null` on
GitHub Actions. The runner (`h200-greennode_03`) is online and idle, but no job
is picked up.

**Root cause:** The config in `configs/nvidia-master.yaml` sets
`runner: cluster:h200-greennode`. The GitHub Actions workflow uses
`runs-on: ${{ matrix.config.runner }}`, so jobs get the label
`cluster:h200-greennode`. But the self-hosted runner was registered with labels
`self-hosted,linux,x64,h200,h200-1x,h200-greennode_03` — it does NOT have the
`cluster:h200-greennode` label. GitHub Actions label matching is exact; no
runner ever claims the job.

**Fix:** Use `--runner-node-filter <substring>` in the generate-cli-command to
emit individual runner node names instead of the cluster label:

```
--runner-type cluster:h200-greennode --runner-node-filter greennode_03
```

This makes the matrix emit `runner: h200-greennode_03` (which the runner does
have as a label) instead of `runner: cluster:h200-greennode`.

**Alternative fix:** Re-register the runner with the `cluster:h200-greennode`
label:
```bash
sudo systemctl stop actions.runner.<org>-<repo>.<runner-name>.service
cd /mnt/runner
sudo -u <user> ./config.sh remove --token <token>
sudo -u <user> ./config.sh --url https://github.com/<org>/<repo> --token <token> --labels self-hosted,linux,x64,h200,h200-1x,h200-greennode_03,cluster:h200-greennode
sudo systemctl start actions.runner.<org>-<repo>.<runner-name>.service
```
Requires `admin:org` scope on the gh CLI token to generate the registration
token via `gh api -X POST /repos/<org>/<repo>/actions/runners/registration-token`.

**Prevention:** Before dispatching, verify that at least one runner in
`configs/runners.yaml` has a GitHub label matching the exact `runner:` value
the matrix will emit. Check runner labels via:
```bash
# On the runner host:
cat /mnt/runner/.runner  # shows agentName
# Check registration labels in diag log:
sudo grep 'Adding option.*labels' /mnt/runner/_diag/Runner_*.log
```

**Run affected:** 31826489057 (cancelled), 31856064893 (first attempt)

---

## [2026-08-15] Transient git clone failure during `uv pip install` — `transformers` fetch fails

**Symptom:** All 3 agentic jobs fail within ~2 minutes. The "Launch job script"
step exits with code 1. No `server.log` is produced (server never started).
Upload steps report "No files were found".

**Root cause:** During `install_agentic_deps()`, `uv pip install` tries to
git-clone `transformers` from `https://github.com/huggingface/transformers.git`
(pinned to a specific commit in `utils/aiperf`). The `git fetch` fails with:
```
error: RPC failed; curl 92 HTTP/2 stream 5 was not closed cleanly: CANCEL (err 8)
error: 6616 bytes of body are still expected
fetch-pack: unexpected disconnect while reading sideband packet
fatal: early EOF
```
This is a transient network error on the runner host — not a config issue.

**Fix:** Just re-dispatch the same run. No config changes needed.

**Prevention:** None needed — this is a flaky network issue. If it recurs
frequently, consider:
- Pre-warming the uv cache on the runner with a manual `uv pip install` of
  the same requirements.
- Setting `GIT_HTTP_VERSION=1.1` or `GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=http.postBuffer GIT_CONFIG_VALUE_0=524288000` in the runner environment to work around HTTP/2 stream issues.

**Run affected:** 31856064893 (all 3 jobs failed), 31857039753 (retry — succeeded)

---

## [2026-08-15] `IndexerTopkCapturer` requires `attn_tp_size == 1` — `--enable-return-indexer-topk` crashes without DP8

**Symptom:** SGLang server starts, loads weights, allocates KV cache, then
crashes during state capturer init with:
```
AssertionError: IndexerTopkCapturer now only supports DP attention
```

**Root cause:** `--enable-return-indexer-topk` instantiates
`IndexerTopkCapturer` in `sglang/srt/state_capturer/indexer_topk.py`, which
asserts `attn_tp_size == 1`. With TP8 and no DP-attention,
`attn_tp_size = 8`. With DP4, `attn_tp_size = 8/4 = 2`. Only DP8 gives
`attn_tp_size = 8/8 = 1`.

**Fix:** Use `--dp 8 --enable-dp-attention` (so `attn_tp_size = 1`) when
running with `--enable-return-indexer-topk`. Also increase
`--watchdog-timeout` to 600+ and reduce `--cuda-graph-max-bs-decode` to 64
to avoid CUDA graph capture timeout with 8 DP workers.

**Prevention:** Only use `--enable-return-indexer-topk` with DP8
(attn_tp_size=1). This flag is for topk measurement, not production.

**Context:** Not an InferenceX dispatch issue — encountered during manual
server startup for topk stability measurement on the 8×H200 box.

---
