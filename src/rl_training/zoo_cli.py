from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from rl_training.resources import find_packaged_asset


def _default_manifest_path() -> Path:
    packaged = find_packaged_asset("zoo/atari/benchmark.yaml")
    if packaged is not None:
        return packaged
    return Path("zoo/atari/benchmark.yaml")


def _load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        packaged = find_packaged_asset(manifest_path)
        if packaged is not None:
            manifest_path = packaged

    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"expected YAML object in {manifest_path}, got {type(payload)!r}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="List or print benchmark commands from the RL Training zoo.")
    parser.add_argument(
        "--manifest",
        default=str(_default_manifest_path()),
        help="Path to a zoo benchmark manifest.",
    )
    parser.add_argument(
        "--format",
        choices=("table", "commands"),
        default="table",
        help="Choose plain table output or shell commands.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = _load_manifest(args.manifest)
    presets = manifest.get("presets", [])
    if not isinstance(presets, list):
        raise TypeError("manifest 'presets' must be a list")

    if args.format == "commands":
        for preset in presets:
            config_path = preset["config"]
            print(f"rl-training train --config {config_path}")
        return 0

    suite = manifest.get("suite", "unknown")
    print(f"suite={suite}")
    for preset in presets:
        print(f"{preset['name']}: {preset['config']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
