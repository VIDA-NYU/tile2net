from functools import *
from typing import TypeVar

import numpy as np
import torch
from PIL import Image, ImageColor

if False:
    pass

T = TypeVar('T')


class ColorMap:
    def __init__(
            self,
            id2rgb: dict[int, tuple[int, int, int]] | None = None,
            id2name: dict[int, str] | None = None,
            color_names: dict[int, str] | None = None,
    ) -> None:
        """
        Initialize a ColorMap for converting class indices to RGB colors.

        Args:
            id2rgb: Mapping from class ID to RGB tuple (e.g., {0: (255, 0, 0)})
            id2name: Optional mapping from class ID to class name (e.g., {0: 'sidewalk'})
            color_names: Optional mapping from class ID to color name (e.g., {0: 'red'})
        """
        if id2rgb is None or callable(id2rgb):
            id2rgb = {
                0: (0, 0, 255),
                1: (0, 128, 0),
                2: (255, 0, 0),
                3: (0, 0, 0)
            }

        self.id2rgb = id2rgb
        self.id2name = id2name or {}
        self.color_names = color_names or {}

        # build flat palette for PIL compatibility
        palette = []
        for i in range(256):
            if i in id2rgb:
                palette.extend(id2rgb[i])
            else:
                palette.extend([0, 0, 0])

        self.palette = palette

    @cached_property
    def _lut_torch(self):
        """Build PyTorch lookup table for GPU-accelerated color mapping."""
        return (
            torch
            .tensor(self.palette, dtype=torch.uint8, device='cuda')
            .view(-1, 3)
        )

    @cached_property
    def _lut_np(self):
        """Build NumPy lookup table for efficient color mapping."""
        return (
            np
            .array(self.palette, dtype=np.uint8)
            .reshape(-1, 3)
        )

    @singledispatchmethod
    def __call__(self, item: object):
        """Map class indices to RGB colors for various input types."""
        raise TypeError(f"Unsupported type: {type(item).__name__}")

    @__call__.register
    def _(
            self,
            item: Image.Image,
    ) -> Image.Image:
        """Convert PIL Image class indices to RGB using palette."""
        out = item.convert("P")
        out.putpalette(self.palette)
        return out

    @__call__.register
    def _(
            self,
            item: np.ndarray,
    ) -> np.ndarray:
        """Convert NumPy array of class indices to RGB."""
        if item.ndim not in (2, 3):
            raise ValueError("Expected (H,W) or (N,H,W) class-index array.")

        item_int = item.astype(np.intp)

        # (H,W) → (H,W,3)
        if item.ndim == 2:
            return self._lut_np[item_int].copy()  # type: ignore[return-value]

        # (N,H,W) → (N,H,W,3)
        n, h, w = item.shape
        return (
            self._lut_np[item_int.reshape(-1)]
            .reshape(n, h, w, 3)
            .copy()
        )  # type: ignore[return-value]

    @__call__.register
    def _(
            self,
            item: torch.Tensor,
    ) -> torch.Tensor:
        """Convert PyTorch tensor of class indices to RGB."""
        if item.dim() not in (2, 3):
            raise ValueError("Expected (H,W) or (N,H,W) class-index tensor.")

        # enforce integer class indices
        if (
                (not item.dtype.is_floating_point)
                and item.dtype not in (
                        torch.int8,
                        torch.int16,
                        torch.int32,
                        torch.int64,
                        torch.uint8,
                )
        ) or item.dtype.is_floating_point:
            raise TypeError("Tensor must contain integer class indices.")

        item_long = item.to(torch.long)

        # (H,W) → (H,W,3)
        if item.dim() == 2:
            return self._lut_torch[item_long].contiguous()  # type: ignore[return-value]

        # (N,H,W) → (N,H,W,3)
        n, h, w = item.shape
        result = (
            self._lut_torch[item_long.view(-1)]
            .view(n, h, w, 3)
            .contiguous()
        )  # type: ignore[return-value]
        return result


    def __set_name__(self, owner, name):
        self.__name__ = name

    def __repr__(self):
        """Return readable representation showing class mappings."""
        lines = ["ColorMap("]

        # first pass: build all base parts and parenthetical parts
        entries = []
        for class_id in sorted(self.id2rgb.keys()):
            rgb = self.id2rgb[class_id]
            rgb_list = list(rgb)

            # base format: "ID -> [R, G, B]"
            base = f"  {class_id} -> {rgb_list}"

            # optional parenthetical info
            class_name = self.id2name.get(class_id)
            color_name = self.color_names.get(class_id)

            if class_name or color_name:
                # use class_id or rgb_list as fallback
                left = class_name if class_name else str(class_id)
                right = color_name if color_name else str(rgb_list)
                paren = f"({left} -> {right})"
            else:
                paren = None

            entries.append((base, paren))

        # find max width of base parts to align parentheses
        max_width = max(len(base) for base, _ in entries) if entries else 0

        # second pass: format with alignment
        for base, paren in entries:
            if paren:
                line = f"{base:<{max_width}} {paren}"
            else:
                line = base
            lines.append(line)

        lines.append(")")
        return "\n".join(lines)
