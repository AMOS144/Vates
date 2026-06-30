import os
from mlx_streaming import config


def test_zerocopy_default_off():
    os.environ.pop("ZEROCOPY_DUAL_SOURCE", None)
    assert config.zerocopy_dual_source() is False


def test_zerocopy_on():
    os.environ["ZEROCOPY_DUAL_SOURCE"] = "1"
    try:
        assert config.zerocopy_dual_source() is True
    finally:
        os.environ.pop("ZEROCOPY_DUAL_SOURCE", None)


def test_pool_spec_slots_default_3():
    os.environ.pop("POOL_SPEC_SLOTS", None)
    assert config.pool_spec_slots() == 3
