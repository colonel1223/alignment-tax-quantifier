"""Verify benchmarking framework produces consistent, reasonable results."""

import sys
sys.path.insert(0, '..')
from core.tax_quantifier import AlignmentTaxQuantifier, METHODS


def test_all_methods_produce_overhead():
    q = AlignmentTaxQuantifier()
    results = q.benchmark()
    assert len(results) == len(METHODS) * 4, f"Expected {len(METHODS)*4} results, got {len(results)}"
    for r in results:
        assert r.flops_overhead >= 0, f"Negative FLOP overhead for {r.method}"
        assert r.task_regression >= 0, f"Negative task regression for {r.method}"
    print(f"  {len(results)} benchmarks, all overhead values valid")


def test_ordering():
    """RLHF should cost more than filtering."""
    q = AlignmentTaxQuantifier(scales=[1_300_000_000])
    results = q.benchmark()
    rlhf = [r for r in results if r.method == "rlhf"][0]
    filt = [r for r in results if r.method == "filtering"][0]
    assert rlhf.flops_overhead > filt.flops_overhead
    print(f"  RLHF overhead ({rlhf.flops_overhead:.1f}%) > filtering ({filt.flops_overhead:.1f}%)")


def test_json_export():
    q = AlignmentTaxQuantifier(methods=["dpo"], scales=[125_000_000])
    q.benchmark()
    output = q.to_json()
    assert '"method": "dpo"' in output
    print("  JSON export valid")


if __name__ == "__main__":
    print("Running alignment-tax-quantifier tests...\n")
    test_all_methods_produce_overhead()
    test_ordering()
    test_json_export()
    print("\nAll tests passed.")
