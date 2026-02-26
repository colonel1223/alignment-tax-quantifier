"""Tests for alignment tax quantifier."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from core.tax_quantifier import AlignmentTaxQuantifier, METHODS
from core.scaling import ScalingLawModel
from core.pareto import ParetoFrontier


def test_benchmark():
    q = AlignmentTaxQuantifier()
    results = q.benchmark()
    assert len(results) == len(METHODS) * 4
    for r in results:
        assert r.flops_overhead_pct >= 0
        assert r.aligned_flops >= r.baseline_flops
    print(f"  {len(results)} benchmarks valid ✓")


def test_ordering():
    q = AlignmentTaxQuantifier(scales=["1.3B"])
    results = q.benchmark()
    rlhf = next(r for r in results if r.method == "rlhf")
    filt = next(r for r in results if r.method == "filtering")
    assert rlhf.flops_overhead_pct > filt.flops_overhead_pct
    print(f"  RLHF ({rlhf.flops_overhead_pct:.0f}%) > filtering ({filt.flops_overhead_pct:.0f}%) ✓")


def test_scaling_law():
    slm = ScalingLawModel()
    params = np.array([125e6, 350e6, 1.3e9, 6.7e9])
    overhead = np.array([42, 41, 40, 39.5])  # Slight decrease with scale
    fit = slm.fit(params, overhead)
    pred_13b = slm.predict(13e9)
    assert 35 < pred_13b < 45
    print(f"  Scaling law R²={fit.r_squared:.3f}, 13B pred={pred_13b:.1f}% ✓")


def test_pareto():
    methods = ["rlhf", "dpo", "filtering", "steering"]
    overheads = np.array([40, 15, 5, 2])
    safeties = np.array([0.95, 0.90, 0.70, 0.85])
    points = ParetoFrontier.compute(methods, overheads, safeties)
    frontier = ParetoFrontier.frontier_only(points)
    names = [p.method for p in frontier]
    assert "steering" in names  # Low overhead, good safety
    print(f"  Pareto frontier: {names} ✓")


if __name__ == "__main__":
    print("alignment-tax-quantifier tests\n")
    test_benchmark()
    test_ordering()
    test_scaling_law()
    test_pareto()
    print("\n✓ All tests passed.")
