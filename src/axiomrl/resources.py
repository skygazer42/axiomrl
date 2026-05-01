from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
ASSETS_ROOT = PACKAGE_ROOT / "assets"


def find_packaged_asset(path: str | Path) -> Path | None:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None

    asset_path = (ASSETS_ROOT / candidate).resolve()
    if not asset_path.is_relative_to(ASSETS_ROOT.resolve()):
        return None
    if asset_path.exists():
        return asset_path
    return None
