"""每层池预算 profile 的解析:默认从 {EXPERT_DIR}/pool_profile.json 自动启用,可覆盖/关闭。"""
from mlx_streaming.model_builder import load_pool_profile


def test_default_from_expert_dir(tmp_path, monkeypatch):
    # 未设环境变量 → 自动读 {EXPERT_DIR}/pool_profile.json
    monkeypatch.delenv("EXPERT_POOL_PROFILE", raising=False)
    (tmp_path / "pool_profile.json").write_text('{"layer_caps": {"0": 32, "5": 64}}')
    assert load_pool_profile(str(tmp_path)) == {0: 32, 5: 64}


def test_disabled_with_none(tmp_path, monkeypatch):
    (tmp_path / "pool_profile.json").write_text('{"layer_caps": {"0": 32}}')
    monkeypatch.setenv("EXPERT_POOL_PROFILE", "none")
    assert load_pool_profile(str(tmp_path)) is None


def test_explicit_path_overrides_default(tmp_path, monkeypatch):
    (tmp_path / "pool_profile.json").write_text('{"layer_caps": {"0": 32}}')
    custom = tmp_path / "custom.json"
    custom.write_text('{"layer_caps": {"3": 16}}')
    monkeypatch.setenv("EXPERT_POOL_PROFILE", str(custom))
    assert load_pool_profile(str(tmp_path)) == {3: 16}


def test_absent_returns_none(tmp_path, monkeypatch):
    monkeypatch.delenv("EXPERT_POOL_PROFILE", raising=False)
    assert load_pool_profile(str(tmp_path)) is None
