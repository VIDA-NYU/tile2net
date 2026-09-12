from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from tile2net.raster.raster import Raster
from tile2net.raster.tile_utils.genutils import deg2num


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the source-tree CLI in a clean child interpreter."""
    environment = os.environ.copy()
    source = Path(__file__).parents[1] / "src"
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(source), environment.get("PYTHONPATH")))
    )
    return subprocess.run(
        [sys.executable, "-m", "tile2net", *arguments],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_top_level_help_does_not_import_ml_dependencies() -> None:
    result = run_cli("--help")

    assert result.returncode == 0, result.stderr
    assert "{generate,inference}" in result.stdout


def test_generate_help_does_not_import_ml_dependencies() -> None:
    result = run_cli("generate", "--help")

    assert result.returncode == 0, result.stderr
    assert "--location" in result.stdout
    assert "--name" in result.stdout


def test_inference_help_imports_installed_ml_stack() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("cv2")
    pytest.importorskip("tabulate")

    result = run_cli("inference", "--help")

    assert result.returncode == 0, result.stderr
    assert "--city_info" in result.stdout
    assert "--active_tile_ids" not in result.stdout
    assert "--eval {test,folder}" in result.stdout
    assert "--vector-format {parquet,shapefile}" in result.stdout
    assert "--dump_percent DUMP_PERCENT" in result.stdout
    assert "100 means all and 0 means none" in result.stdout


def test_inference_mode_normalization() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("cv2")
    pytest.importorskip("tabulate")
    from tile2net.tileseg.inference.inference import normalize_inference_eval_mode

    assert normalize_inference_eval_mode(None) == "test"
    assert normalize_inference_eval_mode("") == "test"
    assert normalize_inference_eval_mode(" TEST ") == "test"
    with pytest.raises(ValueError, match="Unsupported inference mode 'validation'"):
        normalize_inference_eval_mode("validation")


def test_unknown_command_fails_without_traceback() -> None:
    result = run_cli("unknown")

    assert result.returncode == 2
    assert "{generate,inference}" in result.stdout
    assert "Traceback" not in result.stderr


def test_generate_command_emits_valid_pipeline_metadata(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    xtile, ytile = deg2num(40.7500, -74.0000, 19)
    Image.new("RGB", (256, 256), color=(10, 20, 30)).save(
        input_dir / f"{xtile}_{ytile}.png"
    )

    result = run_cli(
        "generate",
        "--location",
        "40.7490,-74.0000,40.7500,-73.9990",
        "--output",
        str(tmp_path / "output"),
        "--name",
        "cli-smoke",
        "--input",
        str(input_dir / "x_y.png"),
        "--zoom",
        "19",
        "--stitch_step",
        "1",
    )

    assert result.returncode == 0, result.stderr
    project = json.loads(result.stdout)
    metadata_path = Path(project["tiles"]["info"])
    stitched_path = Path(project["tiles"]["stitched"])

    assert metadata_path.is_file()
    assert len(tuple(stitched_path.glob("*.png"))) == 9


def test_namespace_accepts_legacy_plural_datasets_key(tmp_path: Path) -> None:
    fake_torch = tmp_path / "torch.py"
    fake_torch.write_text(
        """__version__ = "2.0.0"
class cuda:
    @staticmethod
    def device_count():
        return 0
""",
        encoding="utf-8",
    )
    source = Path(__file__).parents[1] / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(source)))
    code = """
from tile2net.tileseg.config import cfg

cfg["DATASETS"] = cfg.pop("DATASET")
from tile2net.namespace import Namespace

assert cfg.DATASET is cfg.DATASETS
assert Namespace.datasets is Namespace.dataset
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr


def test_raster_inference_uses_active_interpreter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raster = Raster(
        name="interpreter-test",
        location=[40.7490, -74.0000, 40.7500, -73.9990],
        input_dir=tmp_path / "input" / "x_y.png",
        output_dir=tmp_path / "output",
        zoom=19,
        padding=False,
    )
    recorded: dict[str, object] = {}

    def fake_run(arguments: list[str], **kwargs: object) -> None:
        recorded["arguments"] = arguments
        recorded["kwargs"] = kwargs

    monkeypatch.setattr(subprocess, "run", fake_run)

    raster.inference("--local")

    arguments = recorded["arguments"]
    assert isinstance(arguments, list)
    assert arguments[:4] == [sys.executable, "-m", "tile2net", "inference"]
