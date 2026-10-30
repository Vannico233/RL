"""Predeclared observation controls. No policy reads a target answer."""

import hashlib
import math
import random


POLICIES = ("O0", "O1", "O2", "O3", "O4", "D0")


def fixed_regions(size):
    width, height = size
    if width >= height:
        return [[0, 0, width / 2, height], [width / 2, 0, width, height]]
    return [[0, 0, width, height / 2], [0, height / 2, width, height]]


def random_regions(size, example_id, seed):
    """Two reproducible crops: area fraction 0.2--0.5, relative aspect 0.5--2.

    Seed by question id, independent of batching/order. Relative aspect means
    crop aspect / whole-image aspect. Reject samples that do not fit, without
    clipping or silently violating the declared area bounds.
    """
    digest = hashlib.sha256(f"{seed}:{example_id}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    width, height = size
    result = []
    for _ in range(2):
        for _ in range(100):
            area = rng.uniform(0.2, 0.5)
            aspect = math.exp(rng.uniform(math.log(0.5), math.log(2)))
            w, h = width * math.sqrt(area * aspect), height * math.sqrt(area / aspect)
            if w <= width and h <= height:
                x, y = rng.uniform(0, width - w), rng.uniform(0, height - h)
                result.append([x, y, x + w, y + h])
                break
        else:
            raise RuntimeError("Could not sample a crop within the declared bounds")
    return result
