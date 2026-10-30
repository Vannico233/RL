"""Explicit image budgets; all displayed dimensions are multiples of 28."""

from dataclasses import dataclass
import math

from PIL import Image


@dataclass(frozen=True)
class ImageBudget:
    max_pixels: int = 1003520
    min_pixels: int = 3136
    factor: int = 28

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.max_pixels, self.min_pixels, self.factor)):
            raise ValueError("Image budgets must be positive integers")
        if self.max_pixels < self.min_pixels or self.min_pixels < self.factor ** 2:
            raise ValueError("Image budget must satisfy factor^2 <= min_pixels <= max_pixels")


def display_image(image: Image.Image, budget: ImageBudget) -> Image.Image:
    """Resize explicitly before inference so action coordinates are unambiguous.

    Inputs already on the 28-pixel grid and inside these area limits should not
    be resized again by a Qwen2.5-VL processor configured with matching limits.
    The GPU preflight must check the processor's actual image_grid_thw.
    """
    width, height = image.size
    if min(width, height) < 1 or max(width, height) / min(width, height) > 200:
        raise ValueError("Unsupported empty or extreme-aspect image")
    factor = budget.factor
    w = max(factor, round(width / factor) * factor)
    h = max(factor, round(height / factor) * factor)
    if w * h > budget.max_pixels:
        scale = math.sqrt(budget.max_pixels / (width * height))
        w = max(factor, math.floor(width * scale / factor) * factor)
        h = max(factor, math.floor(height * scale / factor) * factor)
        if w * h > budget.max_pixels:
            if w >= h:
                w = budget.max_pixels // (h * factor) * factor
            else:
                h = budget.max_pixels // (w * factor) * factor
    if w * h < budget.min_pixels:
        scale = math.sqrt(budget.min_pixels / (width * height))
        w = max(factor, math.ceil(width * scale / factor) * factor)
        h = max(factor, math.ceil(height * scale / factor) * factor)
    if not budget.min_pixels <= w * h <= budget.max_pixels:
        raise ValueError("Image aspect and budget cannot be satisfied on the patch grid; adjust the budget")
    return image.convert("RGB").resize((w, h), resample=Image.Resampling.LANCZOS)
