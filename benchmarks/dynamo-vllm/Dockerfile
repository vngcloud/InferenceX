# vLLM + Dynamo (aggregated dynamo.vllm workers behind python3 -m dynamo.frontend) for serving YAMLs.
#
# vLLM stays the base image's: ai-dynamo is installed without its vllm extra, and aggregated
# serving needs no NIXL. 1.6.0.dev20261018 is the first ai-dynamo whose vLLM worker publishes
# Gemma-4's hybrid KV groups to the KV router (vLLM >= 0.31's get_kv_cache_group_metadata).
#
#   docker build -t <registry>/dynamo-stack/vllm-runtime:0.31.0-dyn1.6.0.dev20261018 benchmarks/dynamo-vllm
ARG VLLM_IMAGE=vllm/vllm-openai:v0.31.0
FROM ${VLLM_IMAGE}
ARG DYN_VERSION=1.6.0.dev20261018
RUN pip install --no-cache-dir "ai-dynamo==${DYN_VERSION}" "ai-dynamo-runtime==${DYN_VERSION}" blake3 uvloop \
 && python3 - <<'PY'
import vllm, dynamo.vllm, dynamo.frontend, dynamo._core
from vllm.v1.engine.core import EngineCore
assert hasattr(EngineCore, "get_kv_cache_group_metadata"), "vLLM lacks get_kv_cache_group_metadata"
print("vllm", vllm.__version__)
PY
