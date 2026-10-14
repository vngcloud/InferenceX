# custom-agentic tau2: infra steps (needs node access)

Code is done on `feat/custom-agentic-tau2` (stacked on V1 `feat/serving-config-dispatch`).
These steps need a person with access to han-3 / slurm and the user's OK to push.

1. Runner token (runner `h200-greennode-slurm-1x_01`, node `hoanq3-h200-1x-han-3`, runner dir `/data/thanglq5/runner-1x_01`):
   do NOT put the workstation's write/admin HF token on the runner. Create a fine-grained READ-only HF token (read access to the Green-Lab repos),
   save it locally as `~/.secrets/greenlab_hf_read_token` (mode 600), then append it to `/data/thanglq5/runner-1x_01/.env`
   (keep that file mode 600; never paste the token in chat, logs or git). From the workstation, without echoing it:
   ```bash
   ssh slurm-client 'ssh hoanq3-h200-1x-han-3 "umask 077; cat >> /data/thanglq5/runner-1x_01/.env"' < <(printf 'GREENLAB_HF_TOKEN=%s\n' "$(cat ~/.secrets/greenlab_hf_read_token)")
   ```
2. Restart the runner so it re-reads `.env` (runner = sbatch daemon job, script `/shared/thanglq5/daemon-1x01.sh`):
   on `ssh slurm-client`: `squeue -u thanglq5` → `scancel <daemon job id>` → `sbatch /shared/thanglq5/daemon-1x01.sh`;
   then `gh api repos/vngcloud/InferenceX/actions/runners --paginate -q '.runners[]|select(.name=="h200-greennode-slurm-1x_01")|[.status,.busy]'` → `["online",false]`.
3. Push (user OK): `git push -u origin feat/custom-agentic-tau2` (includes V1 commits).
4. Smoke both scenarios from the pushed branch:
   ```bash
   utils/serving/dispatch.sh serving/qwen38-bf16-tp1.yaml
   utils/serving/dispatch.sh serving/qwen38-bf16-tp1-tau2.yaml
   ```
   Pass = both runs green; tau2 `benchmark.log` shows `unique conversations: 1576`; `results/dataset.txt` names
   `Green-Lab/customer-service-agent-traces@2c793380d8ba45bfde79f9a5c7b5991ed40ab2ea`; the recipe log shows `GREENLAB_HF_TOKEN: set`.
   A second tau2 run must not re-download the 1.1 GB file (HF cache hit).
5. Private-path check (proves a future private dataset needs only a lock entry), inside any container on `_01`, with only the runner env:
   `python -c "import os; from huggingface_hub import hf_hub_download as d; print(d('Green-Lab/speed-bench-prepared','README.md',repo_type='dataset',token=os.environ['GREENLAB_HF_TOKEN']))"`
6. Stateless-agent acceptance: a fresh agent (e.g. GreenNode GLM) with only the `serving-bench` skill, prompt:
   "benchmark agentx tau2 smoke cho config: image vllm/vllm-openai:v0.29.0, `vllm serve Qwen/Qwen3.8-27B --max-model-len 262144`, conc 8, runner h200-greennode-slurm-1x_01".
   Pass = it writes one serving YAML with `scenario: custom-agentic-tau2-smoke`, runs `dispatch.sh` once, and the run is green.
