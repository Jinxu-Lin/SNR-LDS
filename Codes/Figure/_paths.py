"""Portable figure inputs and outputs; never write manuscript assets."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("BALDS_DATA_ROOT", ROOT / "_Data")).expanduser().resolve()
FIGURE_ROOT = Path(os.environ.get("BALDS_FIGURE_ROOT", DATA / "results/paper/figures")).expanduser().resolve()
METADATA = FIGURE_ROOT / "metadata"
LEGACY = DATA / "results/paper_figures/legacy"
METADATA.mkdir(parents=True, exist_ok=True)

def figure_dir(name):
    path = FIGURE_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path

def metadata_input(name):
    """Prefer regenerated metadata; otherwise read the accepted archived values."""
    fresh = METADATA / name
    return fresh if fresh.is_file() else LEGACY / name

def data_path(value):
    """Rebase historical _Data references strictly onto the selected data root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        parts = path.parts
        return DATA.joinpath(*parts[1:]) if parts and parts[0] == "_Data" else DATA / path
    if path.is_relative_to(DATA):
        return path
    if "_Data" in path.parts:
        return DATA.joinpath(*path.parts[path.parts.index("_Data") + 1:])
    raise ValueError(f"Artifact path is outside the data root: {value}")
