import subprocess
import sys
from pathlib import Path

PATCH_DIR = Path(__file__).resolve().parents[1] / "runners/vllm_pr57613"

# The v0.31.0 tail of SlidingWindowManager.reachable_block_mask, as a callable stand-in module.
FAKE_MODULE = '''
def tail_mask(start_block, end_block, alignment_tokens, block_size, need, shift, reachable_boundaries):
    mask = [False] * (end_block - start_block)
    retention_interval = 0
    if retention_interval is not None:
            for boundary_tokens in reachable_boundaries:
                aligned = boundary_tokens // alignment_tokens * alignment_tokens
                end = aligned // block_size + shift
                for j in range(max(start_block, end - need), min(end_block, end)):
                    mask[j - start_block] = True
    return mask
'''


def run(tmp_path: Path, module: str, with_patch: bool) -> subprocess.CompletedProcess:
    pkg = tmp_path / "vllm/v1/core"
    pkg.mkdir(parents=True)
    for d in (tmp_path / "vllm", tmp_path / "vllm/v1", pkg):
        (d / "__init__.py").write_text("")
    (pkg / "single_type_kv_cache_manager.py").write_text(module)
    path = f"{PATCH_DIR}:{tmp_path}" if with_patch else str(tmp_path)
    # Gemma-4 TP4 layout: alignment 32, SWA block 16, a hit needs 64 blocks; boundary N-1 = 4099.
    code = ("from vllm.v1.core.single_type_kv_cache_manager import tail_mask;"
            "m = tail_mask(0, 300, 32, 16, 64, 0, [4099]);"
            "print([i for i, k in enumerate(m) if k][0], sum(m))")
    return subprocess.run([sys.executable, "-c", code], env={"PYTHONPATH": path},
                          capture_output=True, text=True, check=False)


def test_patch_keeps_one_alignment_block_below_boundary(tmp_path):
    plain = run(tmp_path / "plain", FAKE_MODULE, with_patch=False)
    patched = run(tmp_path / "patched", FAKE_MODULE, with_patch=True)
    assert plain.stdout.split() == ["192", "64"]  # [end - need, end), end = 4096 / 16
    assert patched.stdout.split() == ["190", "66"]  # + alignment_tokens // block_size = 2 blocks
    assert "[pr57613] patched vllm.v1.core.single_type_kv_cache_manager" in patched.stderr


def test_patch_fails_import_when_target_text_is_missing(tmp_path):
    result = run(tmp_path, FAKE_MODULE.replace("end - need", "end - need - 1"), with_patch=True)
    assert result.returncode != 0
    assert "PR #57613 patch: expected 1 match" in result.stderr
