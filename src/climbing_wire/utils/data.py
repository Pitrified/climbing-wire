"""Utility functions for the whole package."""

from pathlib import Path
from typing import Literal

from loguru import logger as lg


def get_package_fol(
    which_fol: Literal[
        "root",
        "sample_square",
    ]
) -> Path:
    """Get the requested folder."""
    # root_fol = Path(__file__).absolute().parent.parent.parent
    root_fol = Path(__file__).absolute().parents[3]
    lg.debug(f"{root_fol=}")

    if which_fol == "root":
        return root_fol
    elif which_fol == "sample_square":
        return root_fol / "data" / "sample_square"
