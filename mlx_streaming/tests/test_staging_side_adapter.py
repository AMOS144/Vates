from mlx_streaming.core.prefetch.native_staging import _StagingSide


class _FakeStg:
    def sideregion_contents(self, layer, gen=0):
        return {layer * 10: 1}


def test_staging_side_adapter_delegates_contents():
    side = _StagingSide(_FakeStg())
    assert side.contents(2) == {20: 1}


def test_staging_side_adapter_passes_gen():
    class _FakeStgGen:
        def __init__(self):
            self.calls = []
        def sideregion_contents(self, layer, gen=0):
            self.calls.append((layer, gen))
            return {}
    f = _FakeStgGen()
    side = _StagingSide(f, gen=1)
    side.contents(2)
    assert f.calls == [(2, 1)]
