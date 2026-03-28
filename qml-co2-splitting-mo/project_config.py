from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MaterialSpec:
    name: str
    metal_symbol: str
    structure: str
    miller_index: tuple[int, int, int]
    layers: int = 2
    repeat: tuple[int, int, int] = (1, 1, 1)
    vacuum: float = 10.0


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
GEOMETRY_DIR = DATA_DIR / "geometries"
RESULTS_DIR = ROOT_DIR / "results"
DFT_RESULTS_DIR = RESULTS_DIR / "dft"
TDDFT_RESULTS_DIR = RESULTS_DIR / "tddft"

MATERIALS: dict[str, MaterialSpec] = {
    "ZnO": MaterialSpec(
        name="ZnO",
        metal_symbol="Zn",
        structure="wurtzite",
        miller_index=(1, 0, 0),
    ),
    "TiO2": MaterialSpec(
        name="TiO2",
        metal_symbol="Ti",
        structure="rutile",
        miller_index=(1, 1, 0),
    ),
    "CeO2": MaterialSpec(
        name="CeO2",
        metal_symbol="Ce",
        structure="fluorite",
        miller_index=(1, 1, 1),
    ),
}

ADSORPTION_SITES: tuple[str, ...] = ("top_metal", "top_oxygen", "bridge")

CO2_SITE_HEIGHTS = {
    "top_metal": 2.20,
    "top_oxygen": 2.40,
    "bridge": 2.80,
}

# Engineering-friendly screening targets for photocatalytic CO2 splitting.
TARGET_ADSORPTION_ENERGY_EV = -0.70
TARGET_BAND_GAP_EV = 2.40
TARGET_ONSET_EV = 2.20


def ensure_directories() -> None:
    for path in (
        DATA_DIR,
        GEOMETRY_DIR,
        RESULTS_DIR,
        DFT_RESULTS_DIR,
        TDDFT_RESULTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
