from typing import TypeVar, Union, Optional

import numpy as np
import torch
from PIL import Image

if False:
    from .tiles import Tiles

T = TypeVar('T')


class ColorMap:
    def __init__(self, palette: list[int] | None = None) -> None:
        if (
                palette is None
                or callable(palette)
        ):
            palette = [
                0, 0, 255,  # class 0 → blue
                0, 128, 0,  # class 1 → green
                255, 0, 0,  # class 2 → red
                0, 0, 0  # class 3 → black
            ]
        if len(palette) % 3 != 0:
            raise ValueError("Palette length must be a multiple of 3.")
        if len(palette) > 256 * 3:
            raise ValueError("Palette may not exceed 256 colours (768 values).")

        # pad to 256×3
        self.palette: list[int] = palette + [0] * (256 * 3 - len(palette))

        # vectorised lookup tables
        self._lut_np: np.ndarray = np.asarray(self.palette, dtype=np.uint8).reshape(-1, 3)
        self._lut_torch: torch.ByteTensor = torch.tensor(self.palette, dtype=torch.uint8).view(-1, 3)

    def __call__(
            self,
            item: Union[
                Image.Image,
                np.ndarray,
                torch.Tensor,
                T]
    ) -> T:
        # PIL.Image ----------------------------------------------------------
        if isinstance(item, Image.Image):
            out = item.convert("P")
            out.putpalette(self.palette)
            return out  # type: ignore[return-value]

        # NumPy ndarray ------------------------------------------------------
        if isinstance(item, np.ndarray):
            if item.ndim != 2:
                raise ValueError("Expected 2-D class-index array.")
            rgb: np.ndarray = self._lut_np[item.astype(np.intp)]  # (H, W, 3)
            return np.moveaxis(rgb, -1, 0).copy()  # (3, H, W)            # type: ignore[return-value]

        # torch.Tensor -------------------------------------------------------
        if isinstance(item, torch.Tensor):
            if item.dim() != 2:
                raise ValueError("Expected 2-D class-index tensor.")
            if item.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                raise TypeError("Tensor must contain integer class indices.")
            rgb = self._lut_torch[item.to(torch.long)]  # (H, W, 3)
            return rgb.permute(2, 0, 1).contiguous()  # (3, H, W)  # type: ignore[return-value]

        # Fallback -----------------------------------------------------------
        raise TypeError(f"Unsupported type: {type(item).__name__}")

    def __set_name__(self, owner, name):
        self.__name__ = name
    #
    # def __get__(
    #         self,
    #         instance: Optional[Tiles],
    #         owner
    # ):
    #     # todo: support custom palettes
    #     return self
