#!/usr/bin/env bash
# Adaptive arm of the glm5.3w4adpa campaign: the static recipe (boot steps 3)
# plus adaptive speculative decoding under DP attention, which stock v0.5.21
# rejects. The patch is the v0.5.21 backport of the feature branch
# feat/adaptive-spec-dpa-v0521 (python/ only).
set -euo pipefail
PATCH="$(readlink -f "$(dirname "$0")/../../patches/adaptive-spec-dpa-v0521.patch")"
ADAPTIVE_CFG="$(readlink -f "$(dirname "$0")/../../patches/adaptive-spec-dpa-glm53w4a.json")"
[ -f "$PATCH" ] && [ -f "$ADAPTIVE_CFG" ] || { echo "FATAL: adaptive patch/config missing" >&2; exit 1; }
pushd /sgl-workspace/sglang >/dev/null
if git apply --check -p1 "$PATCH" 2>/dev/null; then
  git apply -p1 "$PATCH"
  echo "adaptive-spec-dpa patch applied"
elif git apply --check -p1 -R "$PATCH" 2>/dev/null; then
  echo "adaptive-spec-dpa patch already applied, skipping"
else
  echo "FATAL: adaptive-spec-dpa patch neither applies nor is already applied" >&2
  exit 1
fi
popd >/dev/null
# DeepEP cap stays at the default 128: each tier captures only the graph bs the
# DP consensus can replay (peak 96 tokens/rank, checked at startup).
SPEC_NUM_STEPS=3 SPEC_EXTRA_ARGS="--speculative-adaptive --speculative-adaptive-config $ADAPTIVE_CFG" \
  exec "$(dirname "$0")/glm5.3w4adpa_fp4_h200_sglang_mtp.sh"
