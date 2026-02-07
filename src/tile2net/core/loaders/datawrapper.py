import json
import os
from functools import cached_property
from typing import *
from typing import Any, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .stitch import StitchDataSet
from .. import frame

ArrayLike = Union[
    pd.Series,
    dict[Any, Any],
    list[Any],
    np.ndarray,
]


class DataWrapper(
    frame.FrameWrapper
):
    """
    Wrapper for a DataFrame which specifies the structure for the
    metadata of image tiles to be stitched into mosaics.
    """

    @frame.column
    def input_paths(self) -> pd.Series:
        """Input image filepaths."""

    @frame.column
    def row(self) -> pd.Series:
        """Row in the mosaic which the tile comprises."""

    @frame.column
    def col(self) -> pd.Series:
        """Column in the mosaic which the tile comprises."""

    @cached_property
    def background(self):
        """Background pixel value to use for padding; black by default."""
        return 0

    @frame.column
    def mask(self):
        ...

    @classmethod
    def from_columns(
            cls,
            *,
            input_paths: ArrayLike,
            index: ArrayLike,
            row: ArrayLike,
            col: ArrayLike,
            background: int = None,
            force: bool | ArrayLike = True,
            **kwargs,
    ) -> Self:
        """
        Instantiate a DataWrapper from the specified columns.
        This gives us DataFrame operations for the metadata management.

        Args:
            input_paths:
                series of input files
            index:
                identifier for each tile
            row:
                row index for each tile
            col:
                column index for each tile
            background:
                background value to use for padding
            force:
                whether to force stitching of tile
            **kwargs:
                additional columns to include in the DataFrame
        """
        data = dict(
            input_paths=input_paths,
            row=row,
            col=col,
            force=force,
            **kwargs
        )
        frame = (
            pd.DataFrame(data)
            .set_axis(index)
        )
        names = frame.index.names
        by = [*names, 'row', 'col']
        frame = (
            frame
            .loc[lambda df: df.force]
            .reset_index()
            .sort_values(by=by)
            .set_index(names)
        )

        result: Self = cls.from_frame(frame)
        if background is not None:
            result.background = background
        return result

    def dataset[T](
            self,
            threads: int = None,
            read=None,
            write=None,
            cls: type[T, type['StitchDataSet']] = None,
            *args,
            **kwargs,
    ) -> Union[T, 'StitchDataSet']:
        from tile2net.core.loaders.stitch import StitchDataSet
        if cls is None:
            cls = StitchDataSet
        out = cls(
            self,
            threads=threads,
            read=read,
            write=write,
            *args,
            **kwargs
        )
        return out

    def to_parquet(self, path: str) -> None:
        """Save DataWrapper to parquet file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        frame_to_save = self.frame.copy()
        frame_to_save.to_parquet(path)

        # Store metadata in a separate json file
        metadata = {
            'background': self.background,
        }
        metadata_path = path.replace('.parquet', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

    @classmethod
    def from_parquet(cls, path: str) -> Self:
        """Load DataWrapper from parquet file."""
        frame = pd.read_parquet(path)
        metadata_path = path.replace('.parquet', '_metadata.json')
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            background = metadata.get('background', 0)
        except FileNotFoundError:
            background = 0

        result = cls.from_frame(frame)
        result.background = background
        return result
