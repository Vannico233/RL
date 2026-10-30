"""Crop only the episode's original image; never inspect labels or tables."""

from dataclasses import dataclass, field
import math
from typing import Any

from PIL import Image


@dataclass
class CropResult:
    image: Image.Image | None
    event: dict


@dataclass
class CropSession:
    original: Image.Image = field(repr=False)
    reference_size: tuple[int, int]
    max_calls: int = 2
    min_side_original: int = 2
    attempts: int = field(default=0, init=False)
    events: list[dict] = field(default_factory=list, init=False)

    def __post_init__(self):
        if len(self.reference_size) != 2 or any(type(x) is not int or x < 1 for x in self.reference_size):
            raise ValueError("reference_size must contain positive integer width and height")
        if type(self.max_calls) is not int or not 0 <= self.max_calls <= 2:
            raise ValueError("This project's crop budget is between zero and two")
        if type(self.min_side_original) is not int or self.min_side_original < 1:
            raise ValueError("min_side_original must be positive")
        self.original = self.original.convert("RGB").copy()

    @property
    def remaining(self):
        return self.max_calls - self.attempts

    def execute(self, bbox: Any) -> CropResult:
        """Invalid calls consume budget; requests beyond it never crop an image.

        Coordinates refer to the actual first displayed image. Reject outside
        coordinates instead of silently clipping. Floor left/top and ceil
        right/bottom after mapping to the retained original. Log both spaces.
        """
        event = {
            "attempt": self.attempts + 1,
            "reference_size": list(self.reference_size),
            "original_size": list(self.original.size),
            "requested_bbox": None,
            "executed_bbox_original": None,
            "success": False,
            "repeated_region": False,
            "error": None,
        }
        if self.remaining == 0:
            event["error"] = "budget_exhausted"
            event["consumed_budget"] = False
        else:
            self.attempts += 1
            event["consumed_budget"] = True
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                event["error"] = "bbox_shape"
            elif any(type(x) not in (int, float) or (type(x) is float and not math.isfinite(x)) for x in bbox):
                event["error"] = "bbox_nonfinite_or_nonnumeric"
            else:
                event["requested_bbox"] = list(bbox)
                x1, y1, x2, y2 = bbox
                width, height = self.reference_size
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    event["error"] = "bbox_outside_or_degenerate"
                else:
                    original_width, original_height = self.original.size
                    actual = [
                        math.floor(x1 / width * original_width),
                        math.floor(y1 / height * original_height),
                        math.ceil(x2 / width * original_width),
                        math.ceil(y2 / height * original_height),
                    ]
                    if min(actual[2] - actual[0], actual[3] - actual[1]) < self.min_side_original:
                        event["error"] = "bbox_too_small"
                    elif max(actual[2] - actual[0], actual[3] - actual[1]) / min(actual[2] - actual[0], actual[3] - actual[1]) > 200:
                        event["error"] = "bbox_extreme_aspect"
                    else:
                        crop = self.original.crop(tuple(actual))
                        event["executed_bbox_original"] = actual
                        event["crop_size_original"] = list(crop.size)
                        event["repeated_region"] = any(
                            prior["success"] and prior["executed_bbox_original"] == actual for prior in self.events
                        )
                        event["success"] = True
                        self.events.append(event)
                        return CropResult(crop, event)
        self.events.append(event)
        return CropResult(None, event)
