from __future__ import annotations

import pytest

from tile2net.core.cfg.cfg import Cfg


@pytest.fixture(autouse=True)
def clean_state():
    """Ensure clean state between tests."""
    cfg = Cfg._default
    backup_data = dict(cfg.data)
    backup_context = Cfg._context
    yield
    cfg.data.clear()
    cfg.data.update(backup_data)
    Cfg._context = backup_context


@pytest.fixture
def cfg():
    return Cfg._default


class TestDefaults:
    """Test that default values are accessible via the global cfg."""

    def test_postprocess_is_gac(self, cfg):
        assert cfg.segmentation.postprocess == 'gac'

    def test_static_is_true(self, cfg):
        assert cfg.static is True

    def test_inference_is_true(self, cfg):
        assert cfg.inference is True

    def test_log_level(self, cfg):
        assert cfg.log_level == 'DEBUG'

    def test_download_only_is_false(self, cfg):
        assert cfg.download.only is False

    def test_train_batch_size(self, cfg):
        assert cfg.train.batch_size == 2

    def test_loss_ocr_alpha(self, cfg):
        assert cfg.loss.ocr_alpha == 0.4


class TestLocalOverride:
    """Test setting values on a local Cfg instance."""

    def test_local_override_does_not_affect_default(self, cfg):
        local = Cfg()
        local['segmentation.postprocess'] = 'hysteresis'
        assert local['segmentation.postprocess'] == 'hysteresis'
        assert cfg.segmentation.postprocess == 'gac'

    def test_local_falls_back_to_default(self):
        local = Cfg()
        assert local.segmentation.postprocess == 'gac'

    def test_local_attribute_access(self):
        local = Cfg()
        local['segmentation.postprocess'] = 'gac'
        assert local.segmentation.postprocess == 'gac'

    def test_separate_instances_are_independent(self):
        a = Cfg()
        b = Cfg()
        a['segmentation.postprocess'] = 'gac'
        b['segmentation.postprocess'] = 'hysteresis'
        assert a['segmentation.postprocess'] == 'gac'
        assert b['segmentation.postprocess'] == 'hysteresis'


class TestContextManagerDefault:
    """Test context manager on the global default cfg."""

    def test_temporary_override(self, cfg):
        assert cfg.segmentation.postprocess == 'gac'
        with cfg:
            cfg.segmentation.postprocess = 'hysteresis'
            assert cfg.segmentation.postprocess == 'hysteresis'
        assert cfg.segmentation.postprocess == 'gac'

    def test_multiple_overrides(self, cfg):
        with cfg:
            cfg.segmentation.postprocess = 'hysteresis'
            cfg.train.batch_size = 16
            assert cfg.segmentation.postprocess == 'hysteresis'
            assert cfg.train.batch_size == 16
        assert cfg.segmentation.postprocess == 'gac'
        assert cfg.train.batch_size == 2

    def test_restore_on_exception(self, cfg):
        with pytest.raises(ValueError):
            with cfg:
                cfg.segmentation.postprocess = 'hysteresis'
                raise ValueError('test')
        assert cfg.segmentation.postprocess == 'gac'

    def test_nested_with_blocks(self, cfg):
        """Test nested context managers on the same cfg."""
        with cfg:
            cfg.segmentation.postprocess = 'hysteresis'
            assert cfg.segmentation.postprocess == 'hysteresis'
            with cfg:
                cfg.segmentation.postprocess = 'gmb'
                assert cfg.segmentation.postprocess == 'gmb'
            # inner exit restores cfg to pre-inner state
            assert cfg.segmentation.postprocess == 'hysteresis'
        assert cfg.segmentation.postprocess == 'gac'


class TestJsonRoundtrip:
    """Test JSON serialization and deserialization."""

    def test_roundtrip_preserves_override(self, cfg, tmp_path):
        path = tmp_path / 'cfg.json'
        with cfg:
            cfg.segmentation.postprocess = 'hysteresis'
            cfg.to_json(path)
        loaded = Cfg.from_json(path)
        assert loaded.segmentation.postprocess == 'hysteresis'

    def test_loaded_does_not_affect_default(self, cfg, tmp_path):
        path = tmp_path / 'cfg.json'
        with cfg:
            cfg.segmentation.postprocess = 'hysteresis'
            cfg.to_json(path)
        loaded = Cfg.from_json(path)
        assert cfg.segmentation.postprocess == 'gac'
        assert loaded.segmentation.postprocess == 'hysteresis'

    def test_delete_loaded_no_side_effect(self, cfg, tmp_path):
        path = tmp_path / 'cfg.json'
        with cfg:
            cfg.segmentation.postprocess = 'hysteresis'
            cfg.to_json(path)
        loaded = Cfg.from_json(path)
        del loaded
        assert cfg.segmentation.postprocess == 'gac'

    def test_loaded_falls_back_to_default(self, cfg, tmp_path):
        """Non-overridden values still resolve via default."""
        path = tmp_path / 'cfg.json'
        with cfg:
            cfg.segmentation.postprocess = 'hysteresis'
            cfg.to_json(path)
        loaded = Cfg.from_json(path)
        # batch_size was never overridden
        assert loaded.train.batch_size == 2

    def test_double_roundtrip(self, cfg, tmp_path):
        """Save, load, re-save, reload preserves values."""
        p1 = tmp_path / 'cfg1.json'
        p2 = tmp_path / 'cfg2.json'
        with cfg:
            cfg.segmentation.postprocess = 'hysteresis'
            cfg.to_json(p1)
        first = Cfg.from_json(p1)
        first.to_json(p2)
        second = Cfg.from_json(p2)
        assert second.segmentation.postprocess == 'hysteresis'


class TestLoadedContextManager:
    """Test using a loaded cfg as a context manager."""

    def test_loaded_pushes_to_global(self, cfg, tmp_path):
        path = tmp_path / 'cfg.json'
        with cfg:
            cfg.segmentation.postprocess = 'hysteresis'
            cfg.to_json(path)
        loaded = Cfg.from_json(path)

        assert cfg.segmentation.postprocess == 'gac'
        assert loaded.segmentation.postprocess == 'hysteresis'
        with loaded:
            assert cfg.segmentation.postprocess == 'hysteresis'
            assert loaded.segmentation.postprocess == 'hysteresis'
        assert cfg.segmentation.postprocess == 'gac'
        assert loaded.segmentation.postprocess == 'hysteresis'

    def test_loaded_context_restores_global(self, cfg, tmp_path):
        path = tmp_path / 'cfg.json'
        with cfg:
            cfg.segmentation.postprocess = 'hysteresis'
            cfg.to_json(path)
        loaded = Cfg.from_json(path)

        with loaded:
            pass
        assert cfg.segmentation.postprocess == 'gac'

    def test_assignment_in_loaded_context_visible_globally(self, cfg, tmp_path):
        """Assigning to the loaded cfg within its context is visible via the global."""
        p1 = tmp_path / 'cfg1.json'
        p2 = tmp_path / 'cfg2.json'
        with cfg:
            cfg.segmentation.postprocess = 'hysteresis'
            cfg.to_json(p1)
        loaded = Cfg.from_json(p1)
        loaded.to_json(p2)
        del loaded

        second = Cfg.from_json(p2)
        assert cfg.segmentation.postprocess == 'gac'
        with second:
            second.segmentation.postprocess = 'gmb'
            assert cfg.segmentation.postprocess == 'gmb'
        assert cfg.segmentation.postprocess == 'gac'

    def test_loaded_context_exception_restores(self, cfg, tmp_path):
        path = tmp_path / 'cfg.json'
        with cfg:
            cfg.segmentation.postprocess = 'hysteresis'
            cfg.to_json(path)
        loaded = Cfg.from_json(path)

        with pytest.raises(RuntimeError):
            with loaded:
                loaded.segmentation.postprocess = 'gmb'
                raise RuntimeError('boom')
        # loaded's data restored to pre-enter snapshot
        assert loaded.segmentation.postprocess == 'hysteresis'
        assert cfg.segmentation.postprocess == 'gac'


class TestFlatten:
    """Test flatten method."""

    def test_flatten_includes_all_defaults(self, cfg):
        flat = cfg.flatten()
        defaults = Cfg.from_defaults()
        for key in defaults:
            assert key in flat

    def test_flatten_with_override(self, cfg):
        with cfg:
            cfg.segmentation.postprocess = 'hysteresis'
            flat = cfg.flatten()
            assert flat['segmentation.postprocess'] == 'hysteresis'


class TestHash:
    """Test hash method."""

    def test_hash_changes_with_config(self, cfg):
        h1 = cfg.hash()
        with cfg:
            cfg.segmentation.postprocess = 'hysteresis'
            h2 = cfg.hash()
        assert h1 != h2

    def test_hash_deterministic(self, cfg):
        h1 = cfg.hash()
        h2 = cfg.hash()
        assert h1 == h2


class TestDeletion:
    """Test deleting overrides restores fallback to default."""

    def test_delete_restores_default(self):
        local = Cfg()
        local['segmentation.postprocess'] = 'hysteresis'
        assert local.segmentation.postprocess == 'hysteresis'
        del local['segmentation.postprocess']
        assert local.segmentation.postprocess == 'gac'


class TestTraceKeys:
    """Test that trace keys are correctly formed."""

    def test_top_level_trace(self, cfg):
        assert 'static' in cfg
        assert 'inference' in cfg

    def test_nested_trace(self, cfg):
        assert 'segmentation.postprocess' in cfg
        assert 'train.batch_size' in cfg
        assert 'loss.ocr_alpha' in cfg

    def test_deep_nested_trace(self, cfg):
        assert 'model.ocr.mid_channels' in cfg


if __name__ == '__main__':
    import sys

    cfg = Cfg._default
    backup_data = dict(cfg.data)
    backup_context = Cfg._context

    def reset():
        cfg.data.clear()
        cfg.data.update(backup_data)
        Cfg._context = backup_context

    tests = [
        (TestDefaults, 'test_postprocess_is_gac', 'Default postprocess is gac'),
        (TestDefaults, 'test_static_is_true', 'Default static is True'),
        (TestLocalOverride, 'test_local_override_does_not_affect_default', 'Local override isolated'),
        (TestLocalOverride, 'test_local_falls_back_to_default', 'Local falls back to default'),
        (TestContextManagerDefault, 'test_temporary_override', 'Context manager temporary override'),
        (TestContextManagerDefault, 'test_restore_on_exception', 'Context manager restores on exception'),
        (TestContextManagerDefault, 'test_nested_with_blocks', 'Nested context managers'),
        (TestDeletion, 'test_delete_restores_default', 'Delete restores default'),
        (TestTraceKeys, 'test_top_level_trace', 'Top-level trace keys'),
        (TestTraceKeys, 'test_nested_trace', 'Nested trace keys'),
        (TestTraceKeys, 'test_deep_nested_trace', 'Deep nested trace keys'),
    ]

    all_passed = True
    for cls, method_name, description in tests:
        reset()
        obj = cls()
        method = getattr(obj, method_name)
        # inject cfg fixture for methods that need it
        import inspect
        sig = inspect.signature(method)
        kwargs = {}
        if 'cfg' in sig.parameters:
            kwargs['cfg'] = cfg
        method(**kwargs)
        print(f'  \u2713 {description}')

    sys.exit(0)
