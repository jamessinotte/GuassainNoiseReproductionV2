import itertools
import statistics
import pytest
from repro.analysis import bh_adjust, expected_maximum, measurements, summarize, archive_rows


def test_measurement_is_max_over_seeds_and_sweep_not_mean_of_seed_ratios():
    rows = [{"model":"m","benchmark":"b","measurement":m,"condition":c,"seed":s,"magnitude":x,
             "accuracy":(.2 if x==0 else .3+.1*s) if c=="prompted_sandbagging" else .8,
             "origin":"unit_fixture"}
            for m,c,s,x in itertools.product(["v1","v2"],["ordinary","prompted_sandbagging"],[1,2],[0,.001])]
    measured = measurements(rows)
    assert [r["phi"] for r in measured if r["condition"]=="prompted_sandbagging"] == [2.5,2.5]
    summary,_ = summarize(measured)
    assert all(r["measurements"]==2 for r in summary)  # not 4 seeds or 8 cells


def test_zero_baseline_ratio_is_undefined_not_infinite_or_smoothed():
    rows = [{"model":"m","benchmark":"b","measurement":"v1","condition":"ordinary","seed":1,
             "magnitude":0,"accuracy":0,"origin":"fixture"}]
    assert measurements(rows)[0]["phi"] is None


def test_bh_known_values():
    assert bh_adjust([.01,.04,.03,.2]) == pytest.approx([.04,.0533333333,.0533333333,.2])


def test_expected_max_matches_exhaustive_subsets():
    values = [.1,.2,.4,.9]
    for k in range(1,5):
        exact = statistics.mean(max(v) for v in itertools.combinations(values,k))
        assert expected_maximum(values,k) == pytest.approx(exact)


def test_archives_complete_and_no_combined_duplicates():
    rows = archive_rows()
    assert len(rows) == 59000
    assert len({(r["model"],r["benchmark"]) for r in rows}) == 16
    measured = measurements(rows)
    assert len(measured)==160
    assert all(r["noise_seeds"]==5 for r in measured)
    assert all(r["origin"]=="reference_authors" for r in measured)
