from pathlib import Path

import yaml

WF = Path(__file__).resolve().parents[1] / ".github/workflows"


def load(name: str) -> dict:
    data = yaml.safe_load((WF / name).read_text())
    data["on"] = data.pop(True, data.get("on"))
    return data


def test_no_ingest_in_e2e_tests():
    jobs = load("e2e-tests.yml")["jobs"]
    assert not [name for name in jobs if "ingest" in name.lower()]
    assert "ingest" not in yaml.safe_dump(jobs).lower()


def test_benchmark_tmpl_maps_serving_fields():
    env = load("benchmark-tmpl.yml")["env"]
    for var, key in [
        ("SERVING_ENGINE_ARGS", "engine-args"),
        ("ROUTER_IMAGE", "router-image"), ("ROUTER_CMD", "router-cmd"), ("ROUTER_ENV", "router-env"),
        ("ROUTER_PROFILE", "router-profile"), ("ROUTER_METRICS_PORT", "router-metrics-port"),
    ]:
        assert env[var] == f"${{{{ inputs.serving != '' && fromJSON(inputs.serving).{key} || '' }}}}"
