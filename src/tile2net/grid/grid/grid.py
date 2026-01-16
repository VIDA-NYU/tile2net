from __future__ import annotations

import os
import os.path
import shutil
import sys
import tempfile
from functools import *
from pathlib import Path
from typing import *

import imageio.v3 as iio

from tile2net.grid import util
from tile2net.grid.basegrid.basegrid import BaseGrid
from tile2net.grid.basegrid.static import Static
from tile2net.grid.cfg import cfg
from tile2net.grid.cfg.cfg import Cfg
from tile2net.grid.cfg.logger import logger
from tile2net.grid.dir.outdir import Outdir
from tile2net.grid.dir.tempdir import TempDir
from tile2net.grid.grid import delayed
from tile2net.grid.grid.file import File
from tile2net.grid.grid.network import Network
from tile2net.grid.grid.polygons import Polygons
from tile2net.grid.grid.segtile import SegTile
from tile2net.grid.grid.vectile import VecTile
from tile2net.grid.sampler.benchmark import Benchmark
from tile2net.grid.seggrid.seggrid import SegGrid
from tile2net.grid.source.remote import Remote
from tile2net.grid.source.source import Source
from tile2net.grid.util import assert_perfect_overlap
from tile2net.grid.vecgrid.vecgrid import VecGrid

if False:
    from .filled import Filled
    from .broadcast import Broadcast
    from . import construct


class Grid(
    BaseGrid
):
    """
    "Input Grid" (Grid), comprised of "input tiles" (in-tiles).
    Each tile is an image from the source.

    Example construction:
        >>> grid = Grid.from_location('Boston Common, MA')
        Grid:
                             lonmin        latmax        lonmax        latmin
        xtile  ytile
        317280 387840 -7.911538e+06  5.214840e+06 -7.911500e+06  5.214802e+06
    """

    @File
    def file(self):
        """
        Namespace container for files aligned with the tiles of a Grid.

        Example:
            >>> grid: Grid
            >>> grid.file.Static
            xtile   ytile
            317280  387840    /home/<user>/tile2net/ma/grid/static/20/31...
        """

    @VecGrid
    def vecgrid(self) -> VecGrid:
        """
        Wraps vectorization operations with a grid data structure

        After performing Grid.set_vectorization(), Grid.vecgrid is
        available for performing vectorization on the stitched tiles.

        Example:
            >>> Grid.vecgrid
            VecGrid:
                       lonmin        latmax        lonmax        latmin  \
            xtile ytile
            9915  12120 -7.911538e+06  5.214840e+06 -7.910315e+06  5.213617e+06
                                                                  geometry
            xtile ytile
            9915  12120  POLYGON ((-7910315.183 5214839.818, -7910315.1...

        Handles lazy-loading of Grid.VecGrid:
            >>> VecGrid._get

        Handles vectorization process from stitched seg-tiles:
            >>> VecGrid.vectorize
        """

    @SegGrid
    def seggrid(self) -> SegGrid:
        """
        Wraps segmentation prediction operations with a grid data structure

        After performing Grid.set_segmentation(), Grid.seggrid is
        available for performing segmentation on the stitched tiles.

        Example:
            >>> Grid.seggrid
            SegGrid:
                           lonmin        latmax        lonmax        latmin  \
            xtile ytile
            79320 96960 -7.911538e+06  5.214840e+06 -7.911385e+06  5.214687e+06
                                                                  geometry
            xtile ytile
            79320 96960  POLYGON ((-7911385.302 5214839.818, -7911385.3...
            [64 rows x 5 columns]

        Handles lazy-loading of Grid.SegGrid:
        >>> SegGrid._get

        Handles prediction of seg-tiles from stitched input tiles:
        >>> SegGrid._predict
        """

    @cached_property
    def dimension(self) -> int:
        """
        Tile dimension in pixels; inferred from input files.

        Example:
            >>> grid: Grid
            >>> grid.dimension
            256
        """
        try:
            sample = next(
                p
                for p in self.file.static
                if Path(p).is_file()
            )
        except StopIteration:
            self.download(one=True)
            try:
                sample = next(
                    p
                    for p in self.file.static
                    if Path(p).is_file()
                )
            except StopIteration:
                raise FileNotFoundError('No image files found to infer dimension.')
        result = iio.imread(sample).shape[1]  # width
        return result

    @cached_property
    def name(self) -> str:
        """
        Grid name; inferred from config, location, or input directory.

        Example:
            >>> grid: Grid
            >>> grid.name
            'Boston Common, MA'
        """
        name = (
                self.cfg.name
                or self.cfg.indir.name
                or self.location
                or self.indir.dir.rsplit(os.sep, 1)[-1]
        )
        return name

    @property
    def shape(self):
        """
        Tile shape as (height, width) in pixels.

        Example:
            >>> grid: Grid
            >>> grid.shape
            (256, 256)
        """
        return self.dimension, self.dimension

    @Network
    def network(self):
        """
        Networks from all features (e.g. sidewalks, crosswalks) from all tiles dissolved.

        Example:
            >>> grid: Grid
            >>> grid.network
            Lines:
                                                    geometry
            feature
            crosswalk  LINESTRING (-7910926 5213692.6, -7910925.6 521...
        """

    @Polygons
    def polygons(self):
        """
        Polygons from all features (e.g. sidewalks, crosswalks) from all tiles dissolved into
        continuous polygon geometries.

        Example:
            >>> grid: Grid
            >>> grid.polygons
            Polygons:
                                                    geometry
            feature
            crosswalk  POLYGON ((-7911335.6 5213618.8, -7911339.8 521...
        """

    @Static
    def static(self):
        """
        Namespace for static assets (e.g., placeholder images, weights).

        Example:
            >>> grid: Grid
            >>> grid.Static.black
            >>> grid.Static.hrnet_checkpoint
            >>> grid.Static.snapshot
        """

    @Outdir
    def outdir(self):
        """
        Output in which the results, such as annotated images and geometry, will be stored:
        
        >>> grid: Grid
        >>> Outdir(
        >>>     format='/home/<user>/tile2net/{z}/{x}_{y}',
        >>>     dir='/home/<user>/tile2net',
        >>>     original='/home/<user>/tile2net/z/x_y',
        >>>     suffix='z/x_y'
        >>> )

        Setting the output directory:
        >>> grid: Grid
        >>> grid = grid.set_outdir('/path/to/output')
        """

    @Source
    def source(self):
        """
        Returns the Source class, which wraps a tile server.
        See `Grid.set_remote()` to actually set a source.

        Automatically sets the source:
        >>> grid: Grid
        >>> grid = grid.set_source(...)
        """

    @delayed.Construct
    def construct(self) -> construct.Construct:
        """
        Module which offers pre-constructed `Grid` instances.
        """

    @TempDir
    def tempdir(self):
        """
        Temporary directory for intermediate processing files.

        Example:
            >>> grid: Grid
            >>> grid.tempdir
            Tempdir(
                dir='/tmp/tile2net/ma/grid/static'
                original='/tmp/tile2net/ma/grid/static/z/x_y',
            )
        """
        template = os.path.join(
            tempfile.gettempdir(),
            'tile2net',
            self.indir.suffix
        )
        out = TempDir.from_template(template)
        return out

    @SegTile
    def segtile(self):
        """
        Namespace for seg-tile properties aligned with in-tiles.

        Example:
            >>> grid: Grid
            >>> grid.segtile.xtile
            xtile   ytile
            317280  387840    79320
        """

    @VecTile
    def vectile(self):
        """
        Namespace for vec-tile properties aligned with in-tiles.

        Example:
            >>> grid: Grid
            >>> grid.vectile.xtile
            xtile   ytile
            317280  387840    9915
        """

    def download(
            self,
            retry: bool = True,
            force: bool = False,
            max_workers: int = 64,
            one: bool = False
    ):
        """
        Download tiles from the configured source to the input directory.

        Args:
            retry:
                Retry failed downloads once
            force:
                Re-download existing files
            max_workers:
                Maximum concurrent download threads
            one:
                Download only one file for testing

        Returns:
            Self with downloaded files available at Grid.file.static
        """
        if isinstance(self.source, Remote):
            self.source.download(
                retry=retry,
                force=force,
                max_workers=max_workers,
                one=one,
            )
        else:
            raise TypeError('Grid.source must be set to a Remote to download tiles. ')

    @delayed.Filled
    def filled(self) -> Filled:
        """
        "Fills" the Grid with extra tiles implicated by the VecGrid to avoid missing tiles.
        For example, if the Grid is 1x1, but the VecGrid tiles are supposed to be 2 in-tiles wide,
        this will fill in the missing tiles so there are 2x2 total tiles in the Grid.
        """

    @delayed.Broadcast
    def broadcast(self) -> Broadcast:
        """
        While a seg-tile is comprised of in-tiles, an in-tile may belong to multiple seg-tiles due to
        overlaps. The base Grid dataframe has one row per unique in-tile. The Broadcast extension
        overcomes this limitation by possibly having multiple rows per in-tile.

        Example:
            >>> grid: Grid
            >>> grid.broadcast

                        frame.sort_index()
            Out[17]:
                           segtile.xtile  segtile.ytile  segtile.r  segtile.c
            xtile  ytile
            317275 387839          79319          96959          4          0
                   387839          79319          96960          0          0
        """

    # def set_source(
    #         self,
    #         source: str=None,
    #         outdir: Union[str, Path] = None,
    # ) -> Self:
    #     """
    #     Assign a tile remote for downloading imagery.
    #
    #     Args:
    #         source: remote name, abbreviation, or remote instance. If None, infers from location.
    #         outdir: Optional output directory path
    #
    #     Returns:
    #         Grid with remote and directories configured
    #
    #     Example:
    #         >>> grid: Grid = Grid.from_location('Boston Common, MA').set_source('ma')
    #     """
    #     if source is None:
    #         source = self.cfg.remote
    #     if source is None:
    #         try:
    #             bbox_geom = shapely.geometry.box(*self.frame.total_bounds)
    #             source = (
    #                 gpd.GeoSeries([bbox_geom], crs=self.frame.crs)
    #                 .to_crs(4326)
    #                 .pipe(Remote.from_inferred)
    #             )
    #         except RemoteNotFound as e:
    #             msg = f'Unable to infer a remote for {self.location=}'
    #             raise ValueError(msg) from e
    #     elif isinstance(source, Remote):
    #         source = copy.copy(source)
    #     else:
    #         try:
    #             source = Remote.from_inferred(source)
    #         except RemoteNotFound as e:
    #             msg = (
    #                 f'Unable to infer a remote from {source}. '
    #                 f'Please specify a valid remote or use a '
    #                 f'different method to set the grid.'
    #             )
    #             raise ValueError(msg) from e
    #
    #     result = self.copy()
    #     result.remote = source
    #     msg = (
    #         f'Setting remote to {source.__class__.__name__} '
    #         f'({source.name})'
    #     )
    #     logger.info(msg)
    #
    #     if not outdir:
    #         outdir = (
    #             Path(cfg.outdir)
    #             .expanduser()
    #             .resolve()
    #             .__str__()
    #         )
    #
    #     try:
    #         dir = Dir.from_format(outdir)
    #     except ExtensionNotFoundError:
    #         dir = outdir
    #         try:
    #             dir = Dir.from_format(dir)
    #         except XYNotFoundError as e:
    #             dir = f'{outdir}/z/x_y'
    #             dir = Dir.from_format(dir)
    #
    #     except XYNotFoundError as e:
    #         # msg = (
    #         #     f'Invalid output directory: extension included but '
    #         #     f'no `x` or `y` in the format: {outdir}. '
    #         # )
    #         # raise ValueError(msg) from e
    #         dir = f'{outdir}/z/x_y'
    #         dir = Dir.from_format(dir)
    #
    #     outdir = dir.original
    #
    #     result = result.set_outdir(outdir)
    #
    #     grid = result.outdir.grid
    #     format = os.path.join(
    #         grid.dir,
    #         'static',
    #         f'z/x_y'
    #     )
    #     result = result.set_indir(format)
    #
    #     return result

    def set_source(
            self,
            source: Union[Source, str, None] = None,
    ) -> Self:
        """
        Assign a tile remote for downloading imagery.

        Args:
            source: remote name, abbreviation, or remote instance. If None, infers from location.
            outdir: Optional output directory path

        Returns:
            Grid with remote and directories configured

        Example:
            >>> grid: Grid = Grid.from_location('Boston Common, MA').set_source('ma')

        See:
            >>> Source.__set__
        """
        # todo: merge docstring from old set_source and set_indir:
        """
        Assign an input directory where tiles are stored.

        The directory path must include x and y tile coordinate placeholders.

        Args:
            indir: Directory path with x/y format (e.g., 'input/dir/z/x_y.png')
            name: Optional grid name

        Returns:
            Grid with input directory configured

        Example:
            >>> grid: Grid = grid.set_indir('/path/to/tiles/20/x_y.png')
        """

        result = self.copy()
        if source is None:
            del result.source
            _ = result.source
        else:
            result.source = source

        return result

    # def set_outdir(
    #         self,
    #         outdir: Union[str, Path] = None,
    # ) -> Self:
    #     """
    #     Assign an output directory for processed results.
    #
    #     Args:
    #         outdir: Output directory path
    #
    #     Returns:
    #         Grid with output directory configured
    #
    #     Example:
    #     >>> grid: Grid
    #     >>> grid: Grid = grid.set_outdir('/path/to/output')
    #     """
    #     result: BaseGrid = self.copy()
    #
    #     if not outdir:
    #         outdir = cfg.outdir
    #
    #     if not outdir:
    #         raise ValueError(f'No output directory specified. ')
    #
    #     dir = outdir
    #     try:
    #         result.outdir = dir
    #     except XYNotFoundError as e:
    #         _dir = f'{outdir}/z/x_y'
    #         msg = (
    #             f'XYZ format not implicated in given outdir format: '
    #             f'{dir}. Defaulting to {_dir}'
    #         )
    #         logger.info(msg)
    #         dir = _dir
    #
    #     try:
    #         result.outdir = dir
    #     except ExtensionNotFoundError:
    #         dir = f'{dir}.png'
    #
    #     result.outdir = dir
    #     # noinspection PyUnresolvedReferences
    #     msg = f'Setting output directory to \n\t{result.outdir.original} '
    #     logger.info(msg)
    #
    #     return result

    def set_outdir(
            self,
            outdir: Union[str, Path] = None,
    ) -> Self:
        """
        Assign an output directory for processed results.

        Args:
            outdir: Output directory path

        Returns:
            Grid with output directory configured

        Example:
        >>> grid: Grid
        >>> grid: Grid = grid.set_outdir('/path/to/output')
        """
        result = self.copy()
        result.outdir = outdir
        return result

    def set_segmentation(
            self,
            *,
            dimension: int = None,
            length: int = None,
            mosaic: int = None,
            scale: int = None,
            fill: bool = None,
            batch_size: int = None,
            pad=None,
    ) -> Self:
        """
        Configure segmentation grid dimensions and create Grid.seggrid.

        Args:
            dimension: Pixel dimension of each seg-tile
            length: Number of in-tiles per seg-tile dimension
            scale: Zoom scale for seg-tiles
            fill: Whether to fill missing tiles
            batch_size: Batch size for model inference
            pad: Padding pixels for seg-tiles

        Returns:
            Grid with Grid.seggrid configured

        Example:
            >>> grid: Grid = grid.set_segmentation(dimension=1024, pad=64)
        """
        from ..seggrid import SegGrid
        # todo: if all are None, determine dimension using VRAM

        if dimension or length or scale:
            # directly passed
            ...
        elif (
                cfg.segmentation.dimension !=
                cfg._default.segmentation.dimension
        ):
            # dimension in config
            dimension = cfg.segmentation.dimension
        elif (
                cfg.segmentation.length !=
                cfg._default.segmentation.length
        ):
            # length in config
            length = cfg.segmentation.length
        elif (
                cfg.segmentation.scale !=
                cfg._default.segmentation.scale
        ):
            # scale in config
            scale = cfg.segmentation.scale
        else:
            # use defaults
            dimension = cfg.segmentation.dimension
            length = cfg.segmentation.length
            scale = cfg.segmentation.scale

        scale = self._to_scale(dimension, length, mosaic, scale)

        if batch_size:
            self.cfg.validation.batch_size = batch_size
        if fill is None:
            fill = self.cfg.segmentation.fill

        msg = 'Filling Grid to align with SegGrid'
        logger.debug(msg)
        grid = (
            self
            .to_scale(scale, fill=fill)
            .to_scale(self.scale, fill=fill)
        )
        seggrid = SegGrid.from_rescale(grid, scale, fill)
        grid.seggrid = seggrid
        seggrid = grid.seggrid
        if pad is not None:
            seggrid.pad = pad

        assert (
            grid.filled.segtile.index
            .isin(seggrid.filled.index)
            .all()
        )
        assert (
            seggrid.filled.index
            .isin(grid.filled.segtile.index)
            .all()
        )

        assert seggrid.scale == scale
        assert len(seggrid) <= len(grid)
        assert len(self) <= len(grid)

        area = 4 ** (self.scale - scale)
        assert len(grid) == len(seggrid) * area
        assert_perfect_overlap(seggrid, grid)

        assert seggrid.index.difference(grid.segtile.index).empty

        return grid

    def set_vectorization(
            self,
            *,
            dimension: int = None,
            length: int = None,
            # mosaic: int = None,
            scale: int = None,
            fill: bool = True,
            pad: int = None,
    ) -> Self:
        """
        Configure vectorization grid dimensions and create Grid.vecgrid.

        Args:
            dimension: Pixel dimension of each vec-tile including padding
            length: Number of seg-tiles per vec-tile dimension
            scale: Zoom scale for vec-tiles
            fill: Whether to fill missing tiles
            pad: Padding pixels for vec-tiles

        Returns:
            Grid with Grid.vecgrid configured

        Example:
            >>> grid: Grid = grid.set_vectorization(length=5, pad=128)
        """

        if dimension or length or scale:
            # directly passed
            ...
        elif (
                cfg.vectorization.dimension !=
                cfg._default.vectorization.dimension
        ):
            # dimension in config
            dimension = cfg.vectorization.dimension
        elif (
                cfg.vectorization.length !=
                cfg._default.vectorization.length
        ):
            # length in config
            length = cfg.vectorization.length
        elif (
                cfg.vectorization.scale !=
                cfg._default.vectorization.scale
        ):
            # scale in config
            scale = cfg.vectorization.scale
        else:
            # use defaults
            dimension = cfg.vectorization.dimension
            length = cfg.vectorization.length
            scale = cfg.vectorization.scale

        # todo: if all are None, determine dimension using RAM
        seggrid = self.seggrid
        # if dimension:
        #     dimension -= 2 * seggrid.dimension
        # if length:
        #     assert length >= 3
        #     length -= 2
        # length *= seggrid.length
        # if mosaic:
        #     raise NotImplementedError
        #     mosaic **= 1 / 2
        #     mosaic -= 2
        #     mosaic **= 2
        #     mosaic = int(mosaic)

        scale = self.seggrid._to_scale(dimension, length, scale)

        msg = 'Filling Grid to align with VecGrid'
        logger.debug(msg)
        grid = (
            self
            .to_scale(scale, fill=fill)
            .to_scale(self.scale, fill=fill)
        )

        assert grid.scale == self.grid.scale
        msg = 'Filling SegGrid to align with VecGrid'
        logger.debug(msg)
        seggrid = (
            self.seggrid
            .to_scale(scale, fill=fill)
            .to_scale(self.seggrid.scale, fill=fill)
        )

        grid.seggrid = seggrid

        assert grid.filled.segtile.index.isin(seggrid.filled.index).all()
        assert seggrid.filled.index.isin(grid.filled.segtile.index).all()
        assert seggrid.scale == self.seggrid.scale
        vecgrid = VecGrid.from_rescale(grid, scale, fill=fill)

        if pad is not None:
            vecgrid.pad = pad

        grid.vecgrid = vecgrid
        seggrid = grid.seggrid
        vecgrid = grid.vecgrid

        assert len(self) <= len(grid)
        assert len(vecgrid) <= len(seggrid) <= len(grid)
        area = 4 ** (self.scale - scale)
        assert len(grid) == len(vecgrid) * area

        return grid

    def summary(self) -> None:
        """
        Prints a summary of the grid's contents and performance.
        """

        # helpers
        def _p(v) -> Path | None:
            if v is None:
                return None
            p = Path(v)
            return p if str(p).strip() else None

        def _abs(p: Path) -> str:
            s = str(p.resolve())
            try:
                home = str(Path.home())
                if s == home or s.startswith(home + os.sep):
                    s = '~' + s[len(home):]
            except Exception:
                pass
            return s

        # collect resources
        rows = []

        outdir = _p(getattr(self.outdir, 'dir', None) or getattr(self.outdir, 'root', None))
        tempdir = _p(getattr(self.tempdir, 'dir', None) or getattr(self.tempdir, 'root', None))

        if outdir:
            rows.append(('Output directory', outdir))

        rows.append(('Input imagery', _p(self.outdir.grid.Static.dir)))
        if self.cfg.segmentation.colorized:
            rows.append(('Segmentation (colorized)', _p(self.outdir.seggrid.colorized.dir)))
        if self.cfg.polygon.concat:
            rows.append(('Polygons', _p(self.outdir.polygons.parquet)))
        if self.cfg.line.concat:
            rows.append(('Network', _p(self.outdir.network.parquet)))
        if self.cfg.polygon.preview:
            rows.append(('Polygon preview', _p(self.outdir.polygons.preview)))
        if self.cfg.line.preview:
            rows.append(('Network preview', _p(self.outdir.network.preview)))

        # compute formatting
        label_w = max(len(k) for k, _ in rows)
        term_w = shutil.get_terminal_size((100, 20)).columns
        sep = '=' * min(term_w, 80)

        # color if TTY
        use_color = sys.stdout.isatty()
        BOLD = '\033[1m' if use_color else ''
        DIM = '\033[2m' if use_color else ''
        GRN = '\033[32m' if use_color else ''
        CYN = '\033[36m' if use_color else ''
        YEL = '\033[33m' if use_color else ''
        RST = '\033[0m' if use_color else ''

        # header
        print(sep)
        print(f"{BOLD}Tile2Net run complete!{RST}")
        print(sep)

        # performance summaries
        def _fmt_pct(v: float) -> str:
            return f"{v:.1f}%"

        def _fmt_duration(
                v: float
        ) -> str:
            # choose human-friendly units: d, h, m, s, ms
            if v is None:
                return "—"
            secs = float(v)
            if secs < 0:
                secs = 0.0
            ms = secs * 1000.0
            if secs < 1e-3:
                return "0s"
            if secs < 1.0:
                return f"{ms:.0f}ms"
            days = int(secs // 86400)
            secs -= days * 86400
            hours = int(secs // 3600)
            secs -= hours * 3600
            minutes = int(secs // 60)
            secs -= minutes * 60
            parts = []
            if days:
                parts.append(f"{days}d")
            if hours and len(parts) < 2:
                parts.append(f"{hours}h")
            if minutes and len(parts) < 2:
                parts.append(f"{minutes}m")
            if len(parts) < 2:
                if secs >= 10:
                    parts.append(f"{int(secs)}s")
                else:
                    parts.append(f"{secs:.1f}s")
            return " ".join(parts)

        def _print_line(label: str, avg: float | None, maxv: float | None) -> None:
            parts = []
            if avg is not None:
                parts.append(f"{DIM}avg{RST} {CYN}{_fmt_pct(avg)}{RST}")
            if maxv is not None:
                parts.append(f"{DIM}max{RST} {GRN}{_fmt_pct(maxv)}{RST}")
            if parts:
                print(f"{label}: " + " | ".join(parts))

        # segmentation summary
        try:
            seg_s = self.seggrid.benchmark.samples
            seg_vals = {
                'elapsed': seg_s.time_elapsed,
                'gpu_avg': seg_s.avg_gpu,
                'gpu_max': seg_s.max_gpu,
                'vram_avg': seg_s.avg_vram,
                'vram_max': seg_s.max_vram,
                'ram_avg': seg_s.avg_ram,
                'ram_max': seg_s.max_ram,
                'cpu_avg': seg_s.avg_cpu,
                'cpu_max': seg_s.max_cpu,
            }
            if any(v is not None for v in seg_vals.values()):
                print(sep)
                print(f"{BOLD}Segmentation benchmark{RST}")
                print(sep)
                if seg_vals['elapsed'] is not None:
                    print(f"Time Elapsed: {CYN}{_fmt_duration(seg_vals['elapsed'])}{RST}")
                _print_line("GPU Compute", seg_vals['gpu_avg'], seg_vals['gpu_max'])
                _print_line("VRAM Usage", seg_vals['vram_avg'], seg_vals['vram_max'])
                _print_line("RAM Usage", seg_vals['ram_avg'], seg_vals['ram_max'])
        except Exception:
            pass

        # vectorization summary
        try:
            vec_s = self.vecgrid.benchmark.samples
            vec_vals = {
                'elapsed': vec_s.time_elapsed,
                'ram_avg': vec_s.avg_ram,
                'ram_max': vec_s.max_ram,
                'cpu_avg': vec_s.avg_cpu,
                'cpu_max': vec_s.max_cpu,
            }
            if any(v is not None for v in vec_vals.values()):
                print(sep)
                print(f"{BOLD}Vectorization benchmark{RST}")
                print(sep)
                if vec_vals['elapsed'] is not None:
                    print(f"Time Elapsed: {CYN}{_fmt_duration(vec_vals['elapsed'])}{RST}")
                _print_line("RAM Usage", vec_vals['ram_avg'], vec_vals['ram_max'])
                _print_line("CPU Usage", vec_vals['cpu_avg'], vec_vals['cpu_max'])
        except Exception:
            pass

        # polygon concatenation time
        try:
            poly_s = self.polygon_benchmark.samples  # preferred if available
            if getattr(poly_s, 'time_elapsed', None) is not None:
                print(sep)
                print(f"{BOLD}Polygon concatenation{RST}")
                print(sep)
                print(f"Time Elapsed: {CYN}{_fmt_duration(poly_s.time_elapsed)}{RST}")
        except Exception:
            pass

        # network concatenation time
        try:
            line_s = self.line_benchmark.samples  # preferred if available
            if getattr(line_s, 'time_elapsed', None) is not None:
                print(sep)
                print(f"{BOLD}Network concatenation{RST}")
                print(sep)
                print(f"Time Elapsed: {CYN}{_fmt_duration(line_s.time_elapsed)}{RST}")
        except Exception:
            pass

        # body (two-line format + blank line between rows)
        for label, path in rows:
            if path is None:
                print(f"{label:<{label_w}}")
                print(f"{DIM}— not set —{RST}")
            else:
                path_str = _abs(path)
                if path.exists():
                    color = GRN if path.is_dir() else CYN
                    print(f"{label:<{label_w}}")
                    print(f"{color}{path_str}{RST}")
                else:
                    print(f"{label:<{label_w}}")
                    print(f"{YEL}{path_str}{RST} {DIM}(missing){RST}")
            print()  # <-- blank line between rows

    def cleanup(
            self,
            polygons: bool = True,
            lines: bool = True,
            static: bool = True,
            grayscale: bool = True
    ):
        """ignore for now."""
        raise NotImplementedError
        self.outdir.cleanup()
        self.tempdir.cleanup()

        if polygons:
            msg = (
                f'Cleaning up the polygons for each tile from '
                f'\n\t{self.grid.outdir.vecgrid.polygons.dir}'
            )
            logger.info(msg)
            util.cleanup(self.vecgrid.file.polygons)

        if lines:
            msg = (
                f'Cleaning up the lines for each tile from '
                f'\n\t'
                f'{self.grid.outdir.vecgrid.network.dir}'
            )
            logger.info(msg)
            util.cleanup(self.vecgrid.file.network)

        if static:
            msg = (
                f'Cleaning up previously downloaded imagery '
                f'from {self.grid.indir.dir} and '
                f'{self.grid.outdir.seggrid.Static.dir}'
            )
            logger.info(msg)
            util.cleanup(self.grid.file.Static)
            util.cleanup(self.seggrid.file.Static)

        if grayscale:
            msg = (
                f'Cleaning up segmentation masks '
                f'from \n\t{self.outdir.seggrid.pred.dir} and '
                f'\n\t{self.outdir.vecgrid.pred.dir}'
            )
            logger.info(msg)
            util.cleanup(self.seggrid.file.pred)
            util.cleanup(self.vecgrid.file.pred)

    @cached_property
    def disk_usage(self) -> int:
        """
        Total disk space used by all segmentation files in bytes.
        """
        result = self.broadcast.file.disk_usage.sum()
        return result

    @cached_property
    def time_usage(self) -> float:
        """
        Time spent on segmentation operations in seconds.
        """
        return 0.

    @cached_property
    def polygon_benchmark(self) -> Benchmark:
        result = Benchmark(include_gpu=True)
        return result

    @cached_property
    def line_benchmark(self) -> Benchmark:
        result = Benchmark(include_gpu=True)
        return result

    @classmethod
    def from_cfg(
            cls,
            cfg: Cfg = None
    ) -> Self:
        """
        Construct Grid from a configuration object or JSON file.

        Args:
            cfg: Configuration object, path to JSON config, or None for CLI args

        Returns:
            Fully configured Grid instance

        Example:
            >>> grid: Grid = Grid.from_cfg('config.json')
        """
        if isinstance(cfg, (str, Path)):
            cfg = Cfg.from_json(cfg)
        if cfg is None:
            cfg = Cfg.from_parser()
        with cfg:
            grid = Grid.from_location(
                location=cfg.location,
                zoom=cfg.zoom
            )
            grid.cfg = cfg

            if cfg.indir.path:
                # use input imagery
                grid = grid.set_indir()
            else:
                # set a source if specified or infer from location
                grid = grid.set_source()

            if cfg.outdir:
                grid = grid.set_outdir()

            # configure segmentation using cfg parameters
            grid = grid.set_segmentation()
            # configure vectorization using cfg parameters
            grid = grid.set_vectorization()

        return grid

    def to_cfg(
            self,
            file: Union[str, Path] = None
    ):
        raise NotImplementedError
        ...

    @classmethod
    def from_basic(
            cls,
            outdir=None,
            location=None,
            pad=None,
            length=None
    ) -> Self:
        """
        Construct Grid with basic configuration in one step.

        Args:
            outdir: Output directory path
            location: Location string (e.g., 'Boston Common, MA')
            pad: Padding pixels for segmentation
            length: vec-tile length

        Returns:
            Fully configured Grid instance

        Example:
            >>> grid: Grid = Grid.from_basic(location='Boston, MA', pad=64)
        """

        grid = (
            Grid
            .from_location(location)
            .set_source(
                outdir=outdir,
            )
            .set_segmentation(
                pad=pad,
            )
            .set_vectorization(
                length=length,
            )
        )
        return grid

    @classmethod
    def from_bounds(
            cls,
            latlon: Union[
                str,
                list[float],
                list[int],
                tuple[float, float, float, float],
                tuple[int, int, int, int],
            ],
            zoom: int = None,
    ) -> Self:
        out = (
            super()
            .from_bounds(latlon, zoom)
            .set_segmentation()
            .set_vectorization()
        )
        return out


    @property
    def grid(self) -> Grid:
        """Quick access for the Grid of a project."""
        return self

    __name__ = 'grid'
