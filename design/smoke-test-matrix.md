# Design: Post-Deploy Smoke Test Matrix

## Why

`inference-cicd` (GitOps repo, ArgoCD) deploys serving stacks onto a shared VKS
cluster. Nothing today verifies "did the last deploy still serve correctly" —
`e2e-tests.yml`/`run-sweep.yml` only exercise ephemeral servers this repo
launches itself; they never talk to an already-deployed, externally-managed
endpoint. This design covers a new, separate workflow: a fast correctness +
light-throughput check against live deployments, triggered by `inference-cicd`
on deploy.

This is deliberately *not* a benchmark, and it's new code rather than an
extension of the closest existing system
(`remote:`/`RemoteConfig` agentic-replay in `utils/matrix_logic/validation.py`).
That system solves a different problem: human-dispatched,
private-LAN-only, agentic-trace-throughput benchmarking that requires a
self-hosted `benchmark-client` runner. It explicitly rejects plain
fixed-seq-len throughput against a remote endpoint, has no deploy trigger,
and no correctness checks. None of it applies here.

## Input

The source of truth for "what's deployed and where" is `inference-cicd`'s live
`/discover` endpoint — **not** a hand-maintained catalog in this repo (this
matches the "self-report, no-catalog" design referenced in
`inference-cicd`'s `design/inferencex-integration.md`). All three deployed
stacks are registered:

```bash
$ curl -s http://116.118.91.176.nip.io/discover | jq .
{
  "stacks": [
    {
      "name": "sglang-mooncake-store",
      "base_url": "http://116.118.91.176.nip.io/sglang-mooncake-store",
      "endpoint": "/v1/chat/completions",
      "version_url": "http://116.118.91.176.nip.io/sglang-mooncake-store-version",
      "chart": "sglang-mooncake-store-0.1.0",
      "framework": "sglang",
      "image": "lmsysorg/sglang:v0.5.14",
      "model": "RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8",
      "precision": "fp8",
      "servedName": "DeepSeek-Coder-V2-Lite-Instruct-FP8",
      "tp": 1
    },
    {
      "name": "sglang-pd-disaggregation",
      "base_url": "http://116.118.91.176.nip.io/sglang-pd-disaggregation",
      "endpoint": "/v1/chat/completions",
      "version_url": "http://116.118.91.176.nip.io/sglang-pd-disaggregation-version",
      "chart": "sglang-pd-disaggregation-0.1.0",
      "disaggregation": true,
      "framework": "sglang",
      "image": "lmsysorg/sglang:v0.5.14",
      "model": "RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8",
      "precision": "fp8",
      "servedName": "DeepSeek-Coder-V2-Lite-Instruct-FP8",
      "tp": 1
    },
    {
      "name": "sglang-vanilla",
      "base_url": "http://116.118.91.176.nip.io/sglang-vanilla",
      "endpoint": "/v1/chat/completions",
      "version_url": "http://116.118.91.176.nip.io/sglang-vanilla-version",
      "chart": "sglang-vanilla-0.1.0",
      "framework": "sglang",
      "image": "lmsysorg/sglang:v0.5.14",
      "model": "RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8",
      "precision": "fp8",
      "servedName": "DeepSeek-Coder-V2-Lite-Instruct-FP8",
      "tp": 2
    }
  ]
}
```

The schema isn't fully uniform across stacks — `sglang-pd-disaggregation`
carries an extra `disaggregation: true` field the other two don't have.
Probes/config should treat unlisted fields as optional, not assume every
stack entry has the exact same key set.

Each `version_url` (per-stack self-report, no cluster credentials needed)
independently returns `200` and the same metadata directly.

Input InferenceX still needs to declare itself (not derivable from
`/discover`): which probes to run per stack, throughput concurrency levels,
and the tool-calling schema to test with. This lives in
`.github/configs/smoke-tests.yaml`, keyed by stack `name` so it can be
cross-referenced against whatever `/discover` reports at run time — the
config never hardcodes `base_url`/`model`/`framework`/`tp`, since those come
from `/discover` live and would otherwise drift out of sync with reality.

## Process

1. **Trigger**: `repository_dispatch` (event `stack-deployed`, payload
   `{"stack": "<name>"}`) fired by `inference-cicd` on push to a stack's Helm
   values; also `workflow_dispatch` (optional `stack` input) for manual runs /
   running the full matrix.
2. **Matrix build** (`get-jobs` job): call `/discover`, cross-reference each
   returned stack against `smoke-tests.yaml`'s test-params keyed by name.
   - Stack in both `/discover` and `smoke-tests.yaml` → full matrix entry.
   - Stack in `/discover` but not `smoke-tests.yaml` → still run a default
     probe set (`metadata` + `tool-calling`, skip `throughput`), logged
     explicitly — no silent skip of a live, discoverable stack.
   - Stack named in a `repository_dispatch`/`workflow_dispatch` input but
     absent from `/discover` → fail loudly (deploy claims to exist but isn't
     discoverable — that's itself a signal worth surfacing).
3. **Per-stack job** (matrix over discovered stacks), each running:
   - `metadata`: fetch `version_url`; if `smoke-tests.yaml` declares expected
     model/framework/precision/tp, diff live-reported values against it to
     catch config-vs-reality drift (not just "did it respond").
   - `tool-calling`: real chat-completion request with `tools=[...]`, assert
     a `tool_calls` response.
   - `throughput`: `aiperf`-based sweep against the live endpoint — see
     `design/throughput-test.md` for the full design. Runs on a normal
     hosted `ubuntu-latest` runner: no self-hosted `benchmark-client` runner
     needed, no cluster credentials, since it only ever talks to the public
     Ingress.
4. **Report**: `$GITHUB_STEP_SUMMARY` table, one row per stack; job fails
   (non-zero exit) if any probe fails. DB ingest (same portal as sweep
   results, tagged `run_type: live-check`) is a deliberately separate
   follow-up — needs `InferenceX-app` (a different repo) to agree on how it
   renders/filters that field before we wire `trigger-ingest` to it.

## Commands to run before writing/editing `smoke-tests.yaml`

Never hand-guess a stack's metadata. Always re-verify against the live
cluster and the live `/discover` response first:

```bash
# ground truth: what's actually running in the cluster
kubectl get pods -n inference
kubectl get ingress -n inference

# what InferenceX will actually query at run time
curl -s http://116.118.91.176.nip.io/discover | jq .

# per-stack self-report (same data as one /discover entry, fetched directly)
curl -s http://116.118.91.176.nip.io/<stack>-version | jq .
```

If a stack you want to add isn't in `/discover`'s output, adding it to
`smoke-tests.yaml` is pointless until `inference-cicd` registers it — file
that as a request to the `inference-cicd` owner, not a workaround here.

## Open items

- `sglang-pd-disaggregation`'s single flat `tp: 1` doesn't capture that
  disaggregated serving actually has separate prefill/decode parallelism —
  worth a follow-up question to `inference-cicd` on whether `/discover`
  should report `prefill_tp`/`decode_tp` for disagg stacks, but not a
  blocker for metadata/tool-calling/throughput probes, which don't need that
  breakdown.
- DB ingest tagging (`run_type: live-check`) deferred pending coordination
  with `InferenceX-app`.
- Exact `repository_dispatch` event name/payload shape needs to be agreed
  with whoever owns the `inference-cicd` side of this.
