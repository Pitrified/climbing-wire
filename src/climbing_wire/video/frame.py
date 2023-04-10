"""A frame is a video frame.

With a timestamp and an index.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    """A frame is a video frame."""

    frame: np.ndarray
    usec: int
    idx: int

    def __str__(self) -> str:
        """Return the string representation of a frame."""
        return f"Frame(idx={self.idx}, usec={self.usec})"

    def __repr__(self) -> str:
        """Return a detailed string representation of a frame."""
        return str(self)
