"""bench_verdict 纯裁决逻辑单测:中位数 + go/no-go 判定。"""
import pytest

from mlx_streaming.mtp.bench_verdict import median, verdict_from_delta


def test_median_odd():
    assert median([3, 1, 2]) == 2


def test_median_even():
    assert median([1, 2, 3, 4]) == 2.5


def test_median_single():
    assert median([7.5]) == 7.5


def test_verdict_bug_when_not_exact():
    # exact_all=False 一票否决,无论提速多少都判 bug
    assert verdict_from_delta(0.5, exact_all=False) == "bug"


def test_verdict_go_above_margin():
    assert verdict_from_delta(0.06, exact_all=True, margin=0.05) == "go"


def test_verdict_even_within_margin():
    assert verdict_from_delta(0.02, exact_all=True, margin=0.05) == "even"
    assert verdict_from_delta(-0.02, exact_all=True, margin=0.05) == "even"


def test_verdict_nogo_below_margin():
    assert verdict_from_delta(-0.06, exact_all=True, margin=0.05) == "no-go"


def test_verdict_boundary_is_even():
    # 恰好等于 margin 不算 go(严格大于才 go)
    assert verdict_from_delta(0.05, exact_all=True, margin=0.05) == "even"
    assert verdict_from_delta(-0.05, exact_all=True, margin=0.05) == "even"
