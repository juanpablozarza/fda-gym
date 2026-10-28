"""The frozen 10-drug pilot in data/snapshots/2026-10-15."""

from fdagym.openfda.store import DiskSnapshot

PILOT_DRUGS = (
    "warfarin",
    "metformin",
    "semaglutide",
    "isotretinoin",
    "atorvastatin",
    "montelukast",
    "lisinopril",
    "xanomeline",
    "thioridazine",
    "adalimumab",
)


def load() -> DiskSnapshot:
    return DiskSnapshot("2026-10-15")
