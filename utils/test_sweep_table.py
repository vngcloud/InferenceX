from utils.serving.collect_sweep_table import build_table, parse_scenario, to_markdown


def test_parse_scenario():
    assert parse_scenario("sweep-isl64k-osl1k-cache90") == (90, 65536, 1024)
    assert parse_scenario("sweep-isl128k-osl512-cache0") == (0, 131072, 512)


def test_rows_sorted_by_cache_then_isl_then_osl():
    rows = build_table(
        [
            {"scenario": "sweep-isl64k-osl1k-cache90", "state": "ok", "max_conc": 20, "tpm": 1},
            {"scenario": "sweep-isl8k-osl512-cache0", "state": "ok", "max_conc": 90, "tpm": 2},
            {"scenario": "sweep-isl8k-osl1k-cache0", "state": "ok", "max_conc": 80, "tpm": 3},
        ]
    )
    assert [r["scenario"] for r in rows] == [
        "sweep-isl8k-osl512-cache0",
        "sweep-isl8k-osl1k-cache0",
        "sweep-isl64k-osl1k-cache90",
    ]


def test_non_ok_state_rendered_not_dropped():
    md = to_markdown(
        build_table(
            [
                {
                    "scenario": "sweep-isl128k-osl512-cache0",
                    "state": "no_pass_in_range",
                    "max_conc": None,
                },
            ]
        )
    )
    assert "no_pass_in_range" in md
    assert "None" not in md
