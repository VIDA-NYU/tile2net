from __future__ import annotations

import subprocess
import sys


def test_tile_import_does_not_load_raster_or_tileseg() -> None:
    code = """
import sys
import tile2net.raster.tile

assert 'tile2net.raster.raster' not in sys.modules
assert not any(name.startswith('tile2net.tileseg') for name in sys.modules)
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
