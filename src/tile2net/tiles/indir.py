from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import PathLike
from os import fspath

import dataclasses
import imageio.v3 as iio
import numpy as np
import os
import pandas as pd
import re
from numpy import ndarray
from pathlib import Path
from toolz import curried, curry as cur, pipe
from typing import Iterator

if False:
    from .tiles import Tiles


class Indir:
    tiles: Tiles

    characters = {
        c: i
        for i, c in enumerate('xyz')
    }

    @property
    def files(self) -> pd.Series:
        tiles = self.tiles
        format = self.format
        zoom = tiles.zoom
        it = zip(tiles.ytile, tiles.xtile)
        data = [
            format.format(z=zoom, y=ytile, x=xtile)
            for ytile, xtile in it
        ]
        result = pd.Series(data, index=tiles.index, name='path')
        return result

    @property
    def format(self) -> str | None:
        try:
            return self.__dict__['format']
        except KeyError:
            raise AttributeError('Indir.format is not set.')

    @format.setter
    def format(self, value: str | None):
        self.__dict__['format'] = value

    # root
    @property
    def root(self) -> str | None:
        # todo: why did I implement this?
        try:
            return self.__dict__['root']
        except KeyError:
            raise AttributeError('Indir.root is not set.')

    @root.setter
    def root(self, value: str | None):
        self.__dict__['root'] = value

    # original
    @property
    def original(self) -> str | None:
        try:
            return self.__dict__['original']
        except KeyError:
            raise AttributeError('Indir.original is not set.')

    @original.setter
    def original(self, value: str | Path | PathLike | None):
        if isinstance(value, (Path, PathLike)):
            value = fspath(value)
        self.__dict__['original'] = os.path.normpath(value) if value is not None else value

    # extension
    @property
    def extension(self) -> str | None:
        try:
            return self.__dict__['extension']
        except KeyError:
            raise AttributeError('Indir.extension is not set.')

    @extension.setter
    def extension(self, value: str | None):
        self.__dict__['extension'] = value

    @property
    def suffix(self):
        try:
            return self.__dict__['suffix']
        except KeyError:
            raise AttributeError('Indir.suffix is not set.')

    @suffix.setter
    def suffix(self, value: str | None):
        self.__dict__['suffix'] = value

    def __get__(
            self,
            instance: Tiles,
            owner
    ):
        self.tiles = instance
        self.Tiles = owner
        if instance is None:
            return self
        try:
            result = instance.attrs[self.__name__]
            result.tiles = instance
            result.Tiles = owner
            return result
        except KeyError as e:
            msg = (
                f'{self.__name__!r} is not set. '
                f'See `Tiles.with_indir()` to actually set an '
                f'input directory. See `Tiles.with_source()` to download '
                f'tiles from a source into an input directory.'
            )
            raise AttributeError(msg) from e

    def __bool__(self):
        return self.format is not None

    def __set__(
            self,
            instance: Tiles,
            value: str | PathLike
    ):
        self.tiles = instance
        self.Tiles = type(instance)
        if value is None:
            return
        if isinstance(value, self.__class__):
            instance.attrs[self.__name__] = value
            return
        if isinstance(value, Path):
            value = str(value)
        value = os.path.normpath(value)
        indir = self.__class__()
        instance.attrs[self.__name__] = indir

        indir.original = value

        try:
            indir.extension = value.rsplit('.', 1)[1]
        except IndexError:
            raise ValueError(f'No extension found in {value!r}')

        CHARACTERS: dict[str, str] = pipe(
            re.split(r'[/_. \-\\]', value),
            curried.filter(lambda c: len(c) == 1),
            curried.map(str.casefold),
            set,
            cur(set.__contains__),
            curried.keyfilter(d=indir.characters),
        )
        characters = CHARACTERS.copy()

        string = value
        match = indir._match(string, characters)
        indir.root = match[0]
        result = deque([match])

        while True:
            c = match[1]
            if not c:
                break
            del characters[match[1]]
            if not characters:
                break
            match = indir._match(match[0], characters)
            result.appendleft(match)

        failed = [
            c
            for c in 'xy'
            if c not in CHARACTERS
        ]
        if failed:
            raise ValueError(
                f'indir failed to parse {value!r} '
            )

        parts = []
        r = result[0]
        parts.append(r[0])
        parts.append(f'{{{r[1]}}}')
        parts.append(r[2])
        for r in list(result)[1:]:
            parts.append(f'{{{r[1]}}}')
            parts.append(r[2])
        format = ''.join(parts)
        indir.format = format
        indir.dir = parts[0].rsplit('/', 1)[0]
        indir.suffix = os.path.relpath(indir.original, indir.dir)

    @staticmethod
    def _match(string: str, characters: dict[str, str]):
        c = '|'.join(characters)
        pattern = rf"^(.*)({c})(.*)$"
        match = re.match(pattern, string)
        return match.groups()

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(\n'
            f'    format={self.format!r},\n'
            # f'    root={self.root!r},\n'
            f'    extension={self.extension!r}\n'
            f'    dir={self.dir!r}\n'
            f'    original={self.original!r},\n'
            f')'
        )

    def __set_name__(self, owner, name):
        self.__name__ = name

    def __init__(self, *args, **kwargs):
        ...

    @classmethod
    def is_valid(cls, indir: str):
        try:
            TestIndir(indir)
        except ValueError:
            return False
        else:
            return True


class TestIndir:
    indir = Indir()

    def __init__(self, indir: str | PathLike):
        self.attrs = {}
        self.zoom = 20
        self.extension = '.png'
        self.indir = indir


# class Loader:
#     def __init__(
#             self,
#             files: pd.Series,
#             shape: tuple
#     ):
#         if len(files.groupby(level=files.index.names).size().unique()) != 1:
#             raise ValueError('All file groups must have the same length.')
#         self.files = files
#         # self.dimension = dimension                       # target side length
#         self.shape = shape
#
#     @staticmethod
#     def _read(path: str | os.PathLike, fallback: ndarray) -> ndarray:
#         try:
#             return iio.imread(path) if Path(path).is_file() else fallback
#         except Exception:
#             return fallback
#
#     def __iter__(self) -> Iterator[ndarray]:
#         executor = ThreadPoolExecutor()
#         # groups = self.files.groupby(level=self.files.index.names, sort=False)
#         # sort by mosaic.xtile, mosaic.ytile, mosaic.r, mosaic.c
#
#         groups = self.files
#
#         tile_h, tile_w, *rest = self.fallback.shape
#         ch = rest[0] if rest else 1
#
#         for _, series in groups:
#             n_tiles = len(series)
#             cols = int(np.sqrt(n_tiles))  # row-major order
#             rows = (n_tiles + cols - 1) // cols
#
#             out = np.empty((rows * tile_h,
#                             cols * tile_w,
#                             ch), dtype=self.fallback.dtype)
#
#             fut2pos = {
#                 executor.submit(self._read, p, self.fallback): divmod(i, cols)
#                 for i, p in enumerate(series)
#             }
#
#             for fut in as_completed(fut2pos):
#                 r, c = fut2pos[fut]
#                 out[r * tile_h:(r + 1) * tile_h,
#                 c * tile_w:(c + 1) * tile_w] = fut.result()
#
#             yield out  # shape (H, W, ch)
#
#         executor.shutdown(wait=True)
class Loader:
    """
    Build full-size mosaics tile-by-tile (row-major).

    Parameters
    ----------
    files : pd.Series
        MultiIndex whose *first* level is the group ID
        and whose values are str / Path pointing to tile images.
        The series **must already be row-major sorted**
        (rows first, then columns inside each row).
    shape : tuple[int, int, int]
        (H, W, C) of every yielded mosaic.
    """

    def __init__(self, files: pd.Series, shape: tuple[int, int, int]):
        print('⚠️AI GENERATED🤖')

        if not files.index.is_monotonic_increasing:
            raise ValueError('files index must be sorted (row-major).')

        group_sizes = files.groupby(level=0).size().unique()
        if len(group_sizes) != 1:
            raise ValueError('All file groups must have the same length.')
        self.per_group = int(group_sizes[0])

        self.files = files
        self.H, self.W, self.C = shape

        # ── discover tile size & dtype from first existing image ──────────
        try:
            sample_path = next(p for p in files if Path(p).is_file())
        except StopIteration:
            raise FileNotFoundError('No image files found to infer tile metadata.')

        sample = iio.imread(sample_path)
        th, tw = sample.shape[:2]
        if self.H % th or self.W % tw:
            raise ValueError('`shape` is not divisible by tile size.')
        self.tile_h, self.tile_w = th, tw
        self.dtype = sample.dtype

    @staticmethod
    def _read(path: str | os.PathLike) -> ndarray | None:
        """Return the image or None if missing/broken."""
        try:
            if Path(path).is_file():
                return iio.imread(path)
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[ndarray]:
        ncols = self.W // self.tile_w            # grid dims
        if self.per_group != ncols * (self.H // self.tile_h):
            raise ValueError('group size does not fill the target mosaic.')

        with ThreadPoolExecutor() as pool:
            for _, series in self.files.groupby(level=0, sort=False):
                out = np.zeros((self.H, self.W, self.C), dtype=self.dtype)

                fut2idx = {
                    pool.submit(self._read, p): i
                    for i, p in enumerate(series)
                }

                for fut in as_completed(fut2idx):
                    img = fut.result()
                    if img is None:                # missing → keep zeros
                        continue
                    i = fut2idx[fut]
                    r, c = divmod(i, ncols)        # row-major placement
                    out[
                        r*self.tile_h:(r+1)*self.tile_h,
                        c*self.tile_w:(c+1)*self.tile_w,
                        :img.shape[2] if img.ndim == 3 else ...
                    ] = img

                yield out


#
# class Loader:
#     def __init__(
#         self,
#         files: pd.Series,
#         group: pd.Series,
#         row: pd.Series,
#         col: pd.Series,
#         tile: tuple[int, int, int],
#         mosaic: tuple[int, int, int],
#     ):
#         ...
#

class Loader:
    def __init__(
        self,
        files: pd.Series,
        group: pd.Series,
        row: pd.Series,
        col: pd.Series,
        tile: tuple[int, int, int],
        mosaic: tuple[int, int, int],
    ):
        print('⚠️AI GENERATED🤖')
        if not (len(files) == len(group) == len(row) == len(col)):
            raise ValueError('files, group, row, and col must be the same length')

        self.files = files.reset_index(drop=True)
        self.group = group.reset_index(drop=True)
        self.row = row.reset_index(drop=True)
        self.col = col.reset_index(drop=True)

        self.tile_h, self.tile_w, self.tile_c = tile
        self.mos_h, self.mos_w, self.mos_c = mosaic

        if self.mos_h % self.tile_h or self.mos_w % self.tile_w:
            raise ValueError('mosaic dimensions are not multiples of tile size')

        # dtype discovery from first existing file
        try:
            sample_path = next(p for p in self.files if Path(p).is_file())
            self.dtype = iio.imread(sample_path).dtype
        except StopIteration:
            self.dtype = np.uint8  # default

        # sort by group, then row, then col for deterministic iteration
        order = np.lexsort((self.col, self.row, self.group))
        self.files = self.files.iloc[order]
        self.group = self.group.iloc[order]
        self.row = self.row.iloc[order]
        self.col = self.col.iloc[order]

        self.unique_groups = self.group.unique()
        self.ncols = self.mos_w // self.tile_w
        self.nrows = self.mos_h // self.tile_h

    @staticmethod
    def _read(path: str | os.PathLike) -> ndarray | None:
        try:
            if Path(path).is_file():
                return iio.imread(path)
        except Exception:
            pass
        return None

    def __iter__(self) -> Iterator[ndarray]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor() as pool:
            for g in self.unique_groups:
                mask = self.group == g
                f_group = self.files[mask]
                r_group = self.row[mask].to_numpy()
                c_group = self.col[mask].to_numpy()

                out = np.zeros((self.mos_h, self.mos_w, self.mos_c), dtype=self.dtype)

                fut2idx = {
                    pool.submit(self._read, p): idx
                    for idx, p in enumerate(f_group)
                }

                for fut in as_completed(fut2idx):
                    img = fut.result()
                    if img is None:
                        continue
                    idx = fut2idx[fut]
                    r = r_group[idx]
                    c = c_group[idx]

                    y0 = r * self.tile_h
                    x0 = c * self.tile_w
                    out[y0:y0 + self.tile_h, x0:x0 + self.tile_w, :img.shape[2] if img.ndim == 3 else ...] = img

                yield out


class Loader:
    def __init__(
        self,
        files: pd.Series,
        group: pd.Series,
        row: pd.Series,
        col: pd.Series,
        tile: tuple[int, int, int],
        mosaic: tuple[int, int, int],
    ):
        print('⚠️AI GENERATED🤖')
        if not (len(files) == len(group) == len(row) == len(col)):
            raise ValueError('files, group, row, and col must be the same length')

        self.files = files.reset_index(drop=True)
        self.group = group.reset_index(drop=True)
        self.row = row.reset_index(drop=True)
        self.col = col.reset_index(drop=True)

        self.tile_h, self.tile_w, self.tile_c = tile
        self.mos_h, self.mos_w, self.mos_c = mosaic

        if self.mos_h % self.tile_h or self.mos_w % self.tile_w:
            raise ValueError('mosaic dimensions are not multiples of tile size')

        # dtype discovery from first existing file
        try:
            sample_path = next(p for p in self.files if Path(p).is_file())
            self.dtype = iio.imread(sample_path).dtype
        except StopIteration:
            self.dtype = np.uint8  # default

        # sort by group, then row, then col for deterministic iteration
        order = np.lexsort((self.col, self.row, self.group))
        self.files = self.files.iloc[order]
        self.group = self.group.iloc[order]
        self.row = self.row.iloc[order]
        self.col = self.col.iloc[order]

        self.unique_groups = self.group.unique()
        self.ncols = self.mos_w // self.tile_w
        self.nrows = self.mos_h // self.tile_h

    @staticmethod
    def _read(path: str | os.PathLike) -> ndarray | None:
        try:
            if Path(path).is_file():
                return iio.imread(path)
        except Exception:
            pass
        return None

    def __iter__(self) -> Iterator[ndarray]:

        with ThreadPoolExecutor() as pool:
            for g in self.unique_groups:
                mask = self.group == g
                f_group = self.files[mask]
                r_group = self.row[mask].to_numpy()
                c_group = self.col[mask].to_numpy()

                out = np.zeros((self.mos_h, self.mos_w, self.mos_c), dtype=self.dtype)

                fut2idx = {
                    pool.submit(self._read, p): idx
                    for idx, p in enumerate(f_group)
                }

                for fut in as_completed(fut2idx):
                    img = fut.result()
                    if img is None:
                        continue
                    idx = fut2idx[fut]
                    r = r_group[idx]
                    c = c_group[idx]

                    y0 = r * self.tile_h
                    x0 = c * self.tile_w
                    out[y0:y0 + self.tile_h, x0:x0 + self.tile_w, :img.shape[2] if img.ndim == 3 else ...] = img

                yield out

if __name__ == '__main__':
    class TestIndir:
        indir = Indir()

        def __init__(self, indir: str | PathLike):
            self.attrs = {}
            self.zoom = 20
            self.extension = '.png'
            self.indir = indir


    @dataclasses.dataclass
    class Tile:
        xtile: int
        ytile: int
        zoom: int


    tiles = [
        Tile(1, 2, 3),
        Tile(4, 5, 6),
        Tile(7, 8, 9),
    ]
    test = TestIndir('input/dir/x/y/z.png')
    test = TestIndir('input/dir/x_y_z.png')
    test = TestIndir('input/dir/y/x/z.png')
    test = TestIndir('input/dir/x/y.png')
    # test = TestIndir('input/dir/x/')

    try:
        test = TestIndir('input/dir/x.png')
    except ValueError:
        pass
    else:
        raise AssertionError
    try:
        test = TestIndir('input/dir/x.png')
    except ValueError as e:
        print(e)
    else:
        raise AssertionError
    try:
        test = TestIndir('input/dir/xy.png')
    except ValueError as e:
        print(e)
    else:
        raise AssertionError
    try:
        test = TestIndir('input/dir/x_y_z')
    except ValueError as e:
        print(e)
    else:
        raise AssertionError

    test = TestIndir('input/dir/arst_x_y_z.png')
    test
