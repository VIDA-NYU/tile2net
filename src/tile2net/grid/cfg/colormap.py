from functools import *
from typing import TypeVar

import numpy as np
import torch
from PIL import Image

if False:
    pass

T = TypeVar('T')


class ColorMap:
    def __init__(self, palette: list[int] | None = None) -> None:
        if (
                palette is None
                or callable(palette)
        ):
            palette = [
                0, 0, 255,
                0, 128, 0,
                255, 0, 0,
                0, 0, 0
            ]
        if len(palette) % 3 != 0:
            raise ValueError("Palette length must be a multiple of 3.")
        if len(palette) > 256 * 3:
            raise ValueError("Palette may not exceed 256 colours (768 values).")

        self.palette = (
                palette
                + [0]
                * (256 * 3 - len(palette))
        )

    @cached_property
    def _lut_torch(self):
        return (
            torch
            .tensor(self.palette, dtype=torch.uint8, device='cuda')
            .view(-1, 3)
        )

    @cached_property
    def _lut_np(self):
        return (
            np
            .array(self.palette, dtype=np.uint8)
            .reshape(-1, 3)
        )

    @singledispatchmethod
    def __call__(self, item: object):
        # base fallback for unsupported types
        raise TypeError(f"Unsupported type: {type(item).__name__}")

    @__call__.register
    def _(
            self,
            item: Image.Image,
    ) -> Image.Image:
        # convert to palette-mode image and apply palette
        out = item.convert("P")
        out.putpalette(self.palette)
        return out

    @__call__.register
    def _(
            self,
            item: np.ndarray,
    ) -> np.ndarray:
        # expect (H,W) or (N,H,W) of class indices
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
        # expect (H,W) or (N,H,W) of class indices
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
