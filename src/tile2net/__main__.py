from __future__ import annotations

import sys
from collections.abc import Callable, Sequence

import argh


def _load_command(name: str) -> Callable[..., object]:
    """Import only the dependency tree required by the selected command."""
    if name == "generate":
        from tile2net.raster.generate.generate import generate

        return generate
    if name == "inference":
        from tile2net.tileseg.inference.inference import inference

        return inference
    raise ValueError(f"Unknown command: {name}")


def _print_usage() -> None:
    """Print top-level help without importing raster or ML dependencies."""
    print("usage: python -m tile2net {generate,inference} [options]")


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch one command while keeping raster and ML imports isolated."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help", "help"}:
        _print_usage()
        return

    name, *command_arguments = arguments
    try:
        command = _load_command(name)
    except ValueError as error:
        _print_usage()
        raise SystemExit(2) from error

    argh.dispatch_command(command, argv=command_arguments)


if __name__ == "__main__":
    main()
