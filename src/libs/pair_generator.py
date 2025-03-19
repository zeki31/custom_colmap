import itertools
from pathlib import Path


def get_pairs_exhaustive(
    paths: list[Path],
) -> list[tuple[int, int]]:
    """Obtains all possible index pairs of a list"""
    return list(itertools.combinations(range(len(paths)), 2))
