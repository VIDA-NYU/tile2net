from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

LOCATION = "42.35555189953313,-71.07168915322092,42.35364837213307,-71.06437423368418"


def _validate_stitched_images(project_dir: Path) -> tuple[Path, ...]:
    """Validate stitched imagery using the path recorded in project metadata."""
    metadata_files = tuple(project_dir.glob("tiles/*_info.json"))
    assert len(metadata_files) == 1

    metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
    stitched_dir = Path(metadata["project"]["tiles"]["stitched"])
    stitched_images = tuple(path for path in stitched_dir.iterdir() if path.is_file())
    assert stitched_images

    for stitched_image in stitched_images:
        with Image.open(stitched_image) as image:
            image.verify()
    return stitched_images


def test_stitched_artifact_validation_is_extension_agnostic(tmp_path: Path) -> None:
    """Accept valid provider imagery without assuming PNG output."""
    project_dir = tmp_path / "project"
    stitched_dir = project_dir / "tiles" / "stitched" / "256_19_4"
    stitched_dir.mkdir(parents=True)
    jpeg_path = stitched_dir / "0_0_0.jpg"
    Image.new("RGB", (4, 4)).save(jpeg_path)
    metadata_path = project_dir / "tiles" / "project_256_info.json"
    metadata_path.write_text(
        json.dumps({"project": {"tiles": {"stitched": str(stitched_dir)}}}),
        encoding="utf-8",
    )

    assert _validate_stitched_images(project_dir) == (jpeg_path,)


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.remote
def test_published_pipeline_runs_on_cuda(tmp_path: Path) -> None:
    """Run imagery generation and semantic inference on a real CUDA device."""
    try:
        import torch
    except ModuleNotFoundError:
        pytest.fail(
            "PyTorch is not installed in the GPU test environment.", pytrace=False
        )

    assert torch.cuda.is_available(), "CUDA is unavailable to PyTorch."
    assert torch.cuda.device_count() >= 1, "No CUDA device is visible."

    environment = os.environ.copy()
    visible_devices = environment.get("CUDA_VISIBLE_DEVICES")
    environment["CUDA_VISIBLE_DEVICES"] = (
        visible_devices.split(",", maxsplit=1)[0] if visible_devices else "0"
    )
    source = Path(__file__).parents[1] / "src"
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(source), environment.get("PYTHONPATH")))
    )
    output_dir = tmp_path / "output"
    generate_command = [
        sys.executable,
        "-m",
        "tile2net",
        "generate",
        "--location",
        LOCATION,
        "--output",
        str(output_dir),
        "--name",
        "gpu-example",
    ]
    inference_command = [
        sys.executable,
        "-m",
        "tile2net",
        "inference",
        "--local",
        "--dump_percent",
        "100",
        "--eval",
        "test",
    ]
    generate_log = tmp_path / "generate.log"

    with generate_log.open("w", encoding="utf-8") as log:
        generator = subprocess.Popen(
            generate_command,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            env=environment,
        )
        assert generator.stdout is not None
        inference = subprocess.run(
            inference_command,
            stdin=generator.stdout,
            capture_output=True,
            text=True,
            env=environment,
            timeout=30 * 60,
            check=False,
        )
        generator.stdout.close()
        generate_status = generator.wait(timeout=5 * 60)

    assert generate_status == 0, generate_log.read_text(encoding="utf-8")
    assert inference.returncode == 0, inference.stderr
    assert "CUDA model placement verified" in inference.stderr
    metrics = next(
        (line for line in inference.stderr.splitlines() if "CUDA_METRICS" in line),
        None,
    )
    assert metrics is not None, inference.stderr
    print(metrics)

    project_dir = output_dir / "gpu-example"
    _validate_stitched_images(project_dir)
    segmentation_images = tuple(project_dir.glob("segmentation/**/*.png"))
    assert segmentation_images
    for segmentation_image in segmentation_images:
        with Image.open(segmentation_image) as image:
            image.verify()
    assert not tuple(project_dir.glob("segmentation/**/*.npy"))
    assert tuple(project_dir.glob("polygons/*.parquet"))
    assert tuple(project_dir.glob("network/*.parquet"))
    assert not tuple(project_dir.glob("**/*.shp"))
