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


def test_pool_admission_slots_defaults_to_physical_contribution(monkeypatch):
    monkeypatch.setenv("POOL_SPEC_SLOTS", "56")
    monkeypatch.delenv("POOL_ADMISSION_SLOTS", raising=False)
    assert config.pool_admission_slots() == 56
    monkeypatch.setenv("POOL_ADMISSION_SLOTS", "32")
    assert config.pool_admission_slots() == 32
