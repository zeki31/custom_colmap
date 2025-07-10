from jaxtyping import Float, Int
from numpy.typing import NDArray


class ImageData:
    """Holds keypoints and matches for a single image frame."""

    def __init__(self, frame_id: int):
        self.frame_id = frame_id
        self.matches = {}

    @property
    def matches(self) -> dict[int, Int[NDArray, " 2"]]:
        return self.matches

    def add_kpts(
        self, kpts: Float[NDArray, "N 2"], descs: Float[NDArray, "N 128"]
    ) -> None:
        """Add keypoints to the image data."""
        self.kpts = kpts
        self.descs = descs

    def add_matches(self, curr_frame_id: int, matches: Int[NDArray, "2 N"]) -> None:
        """Add matches to the image data."""
        for t in range(curr_frame_id):
            self.matches[t]
        self.matches[curr_frame_id] = matches
