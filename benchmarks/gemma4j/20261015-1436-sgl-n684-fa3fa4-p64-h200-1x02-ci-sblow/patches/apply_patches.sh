#!/usr/bin/env bash
# Patch a private copy of the image's sglang package and put it first on PYTHONPATH.
# The image's own package (and the node's cached enroot container, reused by other
# runs) is never modified. Both arms run this exact code; only --attention-backend differs.
# Usage: source apply_patches.sh <patch_dir> <site_dir>
_ap_run() {
    local pdir="$1" site="$2" pkg f got
    pkg=$(python3 -c "import importlib.util as u; print(u.find_spec('sglang').submodule_search_locations[0])") || return 1
    f=frozen_kv_mtp_worker_v2.py
    got=$(sha256sum "$pkg/srt/speculative/$f" | cut -d' ' -f1)
    # sglang commit 684fea8a's file; anything else means the tag moved
    [[ "$got" == 7dd3c1d0e9bea8e0ac3fe145538838245fa0be3f746cac7eff56f81a0c68f3dd ]] \
        || { echo "PATCH: $pkg/srt/speculative/$f sha256=$got is not sglang 684fea8a" >&2; return 2; }
    (cd "$pdir" && sha256sum -c --quiet SHA256SUMS) || { echo "PATCH: patch files changed" >&2; return 2; }
    rm -rf "$site"; mkdir -p "$site"; site=$(cd "$site" && pwd -P)
    cp -a "$pkg" "$site/sglang" || return 1
    # PR 43542 runner-level fix (frozen-KV MTP draft runner adopts the target KV pool)
    patch --forward --batch "$site/sglang/srt/speculative/$f" < "$pdir/runner-fix.patch" || return 1
    # sglang PR 44399 @ e536b10bca: fa3_fa4 per-layer backend (paths a/python/sglang/...)
    patch --forward --batch -p2 -d "$site" < "$pdir/hybrid.patch" || return 1
    find "$site/sglang" -name __pycache__ -prune -exec rm -rf {} +
    export PYTHONPATH="$site${PYTHONPATH:+:$PYTHONPATH}"
    python3 - "$site" <<'PY' || return 1
import importlib.util as u, os, sys
loc = os.path.realpath(u.find_spec("sglang").submodule_search_locations[0])
assert loc.startswith(os.path.realpath(sys.argv[1]) + os.sep), f"sglang resolves to {loc}, not the patched copy"
src = open(f"{loc}/srt/arg_groups/choices.py").read()
assert '"fa3_fa4"' in src, "fa3_fa4 not registered"
print("patched sglang:", loc)
PY
    echo "base_image_pkg=$pkg"
    echo "frozen_kv_mtp_worker_v2_sha256_before=$got"
    sha256sum "$pdir"/*.patch
}
_ap_run "$@"
