from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

import pytest

generate_module = import_module("tile2net.raster.generate.generate")


def test_generate_prepares_weights_before_imagery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guarantee that the published generation command prepares checkpoints."""
    events: list[str] = []

    class FakeRaster:
        def generate(self, stitch_step: int) -> None:
            assert stitch_step == 4
            events.append("imagery")

    monkeypatch.setattr(
        generate_module,
        "_raster_from_info",
        lambda _: FakeRaster(),
    )
    monkeypatch.setattr(
        generate_module,
        "ensure_weights",
        lambda: events.append("weights"),
    )
    args = SimpleNamespace(stitch_step=4)

    generate_module.generate(args)

    assert events == ["weights", "imagery"]
