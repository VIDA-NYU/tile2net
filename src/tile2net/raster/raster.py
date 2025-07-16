from __future__ import annotations

import copy
import inspect
import itertools
import json
import logging
import math
import mimetypes
import os
import subprocess
import sys
import weakref
from concurrent.futures import Future, as_completed
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from os import PathLike as _PathLike
from pathlib import Path
from typing import Iterator, Optional, Type, Union

import certifi
import imageio.v2
import numpy as np
import requests
import toolz
from PIL import Image
from numpy import ndarray
from toolz import curried, pipe, partial, curry
from tqdm import tqdm

from tile2net.raster import util
from tile2net.raster.grid import Grid
from tile2net.raster.input_dir import InputDir
from tile2net.raster.project import Project
from tile2net.raster.source import Source
from tile2net.logger import logger


def get_extension(url):
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        content_type = response.headers.get('content-type')
        if content_type:
            extension = mimetypes.guess_extension(content_type.split(';')[0])
            return extension if extension else 'unknown'
        return 'unknown'
    except Exception:
        return 'unknown'

def get_extensions(
        urls: list[str] | ndarray,
) -> list[str]:
    with ThreadPoolExecutor() as executor:
        extensions = list(executor.map(get_extension, urls))
    return extensions


PathLike = Union[str, _PathLike]


class Black:
    def __get__(self, instance, owner) -> Black:
        self.owner = weakref.proxy(instance)
        if instance is None:
            return self
        if self.name in instance.__dict__:
            return instance.__dict__[self.name]
        result = copy.copy(self)
        instance.__dict__[self.name] = result
        return result

    def __set_name__(self, owner, name):
        self.name = name

    @cached_property
    def path(self):
        result: Path = self.owner.project.resources.path / 'black' / f'{self.owner.zoom}.png'
        result.parent.mkdir(parents=True, exist_ok=True)
        if not result.exists():
            imageio.v3.imwrite(result, self.array)
        return result

    def __fspath__(self):
        return str(self.path)

    @cached_property
    def array(self):
        return np.zeros((self.owner.tile_size, self.owner.tile_size, 3), dtype=np.uint8)


class Raster(Grid):
    Project = Project
    input_dir = InputDir()
    black = Black()

    @classmethod
    def from_nyc(cls, outdir: PathLike = None) -> "Raster":
        """
        Create a Raster object for NYC

        Parameters
        ----------
        outdir : PathLike
            Path to the output directory

        Returns
        -------
        Raster
        """
        return cls(
            location=[
                40.91766362458114,
                -73.70040893554688,
                40.476725355504186,
                -74.25933837890625,
            ],
            name="nyc",
            output_dir=outdir,
        )

    @classmethod
    def from_info(cls, info: PathLike | dict) -> "Raster":
        """
        Create a Raster object from a json file

        Parameters
        ----------
        info : PathLike | dict
            Path to a json file or a dictionary containing the following keys:

            - location: list
                Bounding box of the region of interest
            - name: str
                Name of the region
            - output_dir: PathLike
                Path to the output directory
            - source: PathLike
                Path to the source directory

        Returns
        -------
        Raster
        """
        if not isinstance(info, dict):
            with open(info) as f:
                kwargs = json.load(f)
        else:
            kwargs = info
        relevant = toolz.keyfilter(
            inspect.signature(cls.__init__).parameters.__contains__, kwargs
        )
        res = cls(**relevant)
        return res

    @classmethod
    def from_stitched(cls, info: PathLike | dict, tile_step=4) -> "Raster":
        """
        Create a Raster object from stitched tiles

        Parameters
        ----------
        info : PathLike | dict
            Path to a json file or a dictionary containing the following keys:

            - location: list
                Bounding box of the region of interest
            - name: str
                Name of the region
            - output_dir: PathLike
                Path to the output directory
            - source: PathLike
                Path to the source directory
            - tile_step: int
                Step size for the tiles
        tile_step : int
            Step size for the tiles

        Returns
        -------
        Raster
        """
        if not isinstance(info, dict):
            with open(info) as f:
                kwargs = json.load(f)
        else:
            kwargs = info
        return cls(
            location=kwargs["location"],
            name=kwargs["name"],
            output_dir=kwargs["output_dir"],
            # source=kwargs['source'],
            tile_step=tile_step,
        )

    def __init__(
            self,
            *,
            location: list | str,  # region of interest to get its bounding box
            name: str = None,
            # source: PathLike = None,
            input_dir: PathLike = None,
            output_dir: PathLike = None,
            num_class: int = 4,  # # of classes for annotation creation
            base_tilesize: int = 256,
            zoom: int = None,
            crs: int = 4326,
            tile_step: int = 1,
            boundary_path: str = None,  # path to a shapefile to filter out of boundary tiles
            padding=True,
            # extension: str = 'png',
            source: Source | Type[Source] = None,
            dump_percent: int = 0,
            debug: bool = False,
    ):
        """

        Parameters
        ----------
        location : list | str
            region of interest to get its bounding box
        name : str
            name of the project
        input_dir : str
            path to the directory containing the input images;
            this must implicate the format of the contained files,
            containing the xtile, ytile, and extension,
            and may or may not contain the zoom level, e.g.
            path/to/files/x/y.ext
            path/to/files/z/x_y.ext
            path/to/files/z/y/x.ext

            if input_dir is passed, zoom must be passed as well
        output_dir : PathLike
            path to the directory containing the output images
        num_class : int
            # of classes for annotation creation
        base_tilesize : int
            size of the base tile in pixels (default: 256)
        zoom : int
            zoom level of the tiles (default: None)
        crs : int
            coordinate reference system (default: 4326)
        tile_step : int
            step size for the tiles (default: 1)
        boundary_path : str
            path to a shapefile to filter out of boundary tiles (default: None)
        padding : bool
            whether to pad the tiles to the base tile size (default: True)
        extension : str
            extension of the input images (default: 'png')
        source : Source | Type[Source] | str
            tile source (default: None)
        dump_percent : int
            percentage of the tiles to dump (default: None)
        """
        global logger
        if debug:
            logger.setLevel(logging.DEBUG)

            # logger = tile2net.logger.logger = logging.getLogger('debug')
        if name is None:
            name = util.name_from_location(location)
        location = util.geocode(location)
        if input_dir is not None and source is not None:
            raise ValueError("Cannot specify both source and input_dir")
        if input_dir is None:
            if source is None:
                source = Source[location]
                if source is None:
                    ...
                else:
                    logger.info(
                        f"Using {source.__class__.__name__} as the source at {location=}"
                    )
            elif isinstance(source, type):
                source = source()
            elif isinstance(source, str):
                source = Source[source]
            elif isinstance(source, Source):
                ...
            else:
                raise TypeError(f"Invalid source type: {type(source)=}")

            if source is None:
                logger.warning(
                    f"No source found for {location=}; cannot stitch or inference. "
                )
            else:
                if zoom is None:
                    zoom = source.zoom
                    logger.info(f"Using {zoom=} from source")
                base_tilesize = source.tilesize
                logger.info(f"Using {base_tilesize=} from source")
        else:
            if base_tilesize is None:
                raise ValueError("Tile size must be specified with input_dir")

        if base_tilesize < 256:
            raise ValueError("Tile sizes cannot be smaller than 256")
        if not base_tilesize % 256 == 0:
            raise ValueError("Tile size must be a multiple of 256")
        if zoom is None:
            raise ValueError("Zoom level must be specified")
        if tile_step & (tile_step - 1):
            raise ValueError("Tile step must be a power of 2")
        if not 0 <= dump_percent <= 100:
            raise ValueError("Dump percent must be between 0 and 100")

        self.zoom = zoom
        self.source = source
        self.zoom = zoom
        self.location = location
        self.num_class = num_class
        self.class_names = []
        self.class_colors = []
        self.class_order = []
        self.dest = ""
        self.name = name
        self.boundary_path = ""
        self.input_dir = input_dir
        self.source = source
        self.dump_percent = dump_percent
        self.batch = -1
        self.debug = debug

        if boundary_path:
            self.boundary_path = boundary_path
            self.get_in_boundary(path=boundary_path)

        super().__init__(
            location=location,
            name=name,
            base_tilesize=base_tilesize,
            zoom=zoom,
            crs=crs,
            tile_step=tile_step,
            padding=padding,
            output_dir=output_dir,
        )

    def __repr__(self):
        if self.boundary_path != "":
            tiles_within = f"{(self.num_inside / self.num_tiles) * 100:.1f}"
            return (
                f"{self.name} Data Constructor. \nCoordinate reference system (CRS): {self.crs} \n"
                f"Tile size (pixel): {self.tile_size} \nZoom level: {self.zoom} \n"
                f"Number of columns: {self.width:,} \n"
                f"Number of rows: {self.height:,} \n"
                f"Total tiles: {self.num_tiles:,} \n"
                f"Number of tiles inside the boundary: {self.num_inside:,} ({tiles_within}%) \n"
            )
        else:
            return (
                f"{self.name} Data Constructor. \nCoordinate reference system (CRS): {self.crs} \n"
                f"Tile size (pixel): {self.tile_size} \nZoom level: {self.zoom} \n"
                f"Number of columns: {self.width:,} \n"
                f"Number of rows: {self.height:,} \n"
                f"Total tiles: {self.num_tiles:,} \n"
            )

    def update_data_setting(self):
        raise NotImplementedError

    def create_config_json(self):
        raise NotImplementedError

    def download_google_api(self, api_key) -> None:
        raise NotImplementedError

    def check_tile_size(self, img_path: str):
        """
        Check if the input tile image size is the same as the base tile size

        Parameters
        ----------
        img_path : str
            Path to the input tile image

        Returns
        -------
        bool
            True if the input tile image size is the same as the base tile size
        """
        im = Image.open(img_path)
        if im.size[0] == im.size[1] == self.base_tilesize:
            return True
        elif im.size[0] != im.size[1]:
            raise ValueError(
                "Input Tile Image Height and Width values should be the same!"
            )
        elif im.size[0] == im.size[1] and im.size[1] != self.base_tilesize:
            raise ValueError(
                "Input Tile image Height and Width values does not match the"
                ' Grid tile size "{self.tile_size}"!'
            )
        else:
            return False

    """
    Stitch Tiles
    """

    def stitch(self, step: int, force=False) -> None:
        """
        Stitch tiles

        Parameters
        ----------
        step : int
            Stitch step.
                The amount of tiles that an output file from the semantic segmentation will represent.
            For instance, to get a 512x512 tiles from the 256x256 base, the stitch step is 2
        extension : str
            File extension of the tiles. Default is 'png'.
        loc_abr : str
            The abbreviation of the area/state to download the missing tiles when padding.
            If None, missing tiles will be replaced with gray tiles.

        Returns
        ------
        None.
            Nothing is returned.
        """
        logger.info(f"Stitching {len(self.tiles):,} tiles...")
        self.stitch_step = step
        self.calculate_padding()
        self.update_tiles()
        self.download()
        self.project.tiles.stitched.path.mkdir(parents=True, exist_ok=True)
        if not (self.source or self.input_dir):
            raise RuntimeError(
                "No source or input directory specified. Cannot stitch tiles."
            )
        outfiles = pipe(
            # self.tiles[:r:step, :c:step],
            self.tiles[::step, ::step],
            self.project.tiles.stitched.files,
            list,
        )
        not_exists = [not os.path.exists(outfile) for outfile in outfiles]
        if not force:
            outfiles = list(itertools.compress(outfiles, not_exists))
        if not outfiles:
            logger.info(f"All tiles already stitched.")
            return

        infiles: np.ndarray = pipe(
            self.tiles,
            self.project.tiles.static.files,
            list,
            np.array
        )
        indices = np.arange(self.tiles.size).reshape((self.width, self.height))
        indices = (
            indices
            # iterate by step to get the top left tile of each new merged tile
            # [:r:step, :c:step]
            [::step, ::step]
            # reshape to broadcast so offsets can be added
            .reshape((-1, 1, 1))
            # add offsets to get the indices of the tiles to merge
            .__add__(indices[:step, :step])
            # flatten to get a list of merged tiles
            .reshape((-1, step * step))
        )
        if not force:
            # filter for tiles that are not stitched
            indices = indices[not_exists]
        list_infiles = (
            # get files from 2d indices to get list of lists
            infiles
            [indices]
            .tolist()
        )
        assert len(list_infiles) == len(outfiles)
        if not list_infiles:
            # todo
            return

        threads = ThreadPoolExecutor()
        if not any(
                os.path.exists(file)
                for file in itertools.chain.from_iterable(list_infiles)
        ):
            raise FileNotFoundError(
                f"No relevant tiles found in {self.project.tiles.static.path}. "
                f"If multiple sources were matched, consider specifying a different source."
            )
        sample: np.ndarray = next(
            imageio.v3.imread(file)
            for file in itertools.chain.from_iterable(list_infiles)
            if os.path.exists(file)
        )
        if sample.shape[:2] != (self.base_tilesize, self.base_tilesize):
            raise ValueError(
                f"Input tile size {sample.shape[:2]} does not match "
                f"expected tile size {self.base_tilesize}."
            )

        # black = np.zeros((self.base_tilesize, self.base_tilesize, 3), dtype=np.uint8)
        # self.project.resources
        black = self.black.array

        def imread(file) -> np.ndarray:
            if (
                    os.path.islink(file)
                    or not os.path.exists(file)
            ):
                return black
            res = imageio.v3.imread(file)
            return res

        def gen_infiles():
            """
            Generator that yields lists of lists of input files.
            Each list of input files is a list of files to merge into a single output file.
            The generator yields a list of input files each time a list of output files is yielded.
            This allows the generator to preemptively load the next list of input files while the
            current list of input files is being processed.

            Returns
            -------
            list[list[str]]
                List of lists of input files.
            """
            it_infiles = iter(list_infiles)
            # load first two lists
            # first: list[Future] = [
            #     threads.submit(imageio.v2.imread, infile)
            #     for infile in next(it_infiles)
            # ]
            first: list[Future] = [
                threads.submit(imread, infile)
                for infile in next(it_infiles)
            ]
            try:
                second: list[Future] = [
                    threads.submit(imread, infile)
                    for infile in next(it_infiles)
                ]
            except StopIteration:
                yield [
                    future.result()
                    for future in first
                ]
                return
            else:
                yield [
                    future.result()
                    for future in first
                ]
            first = second

            # preemptively load the next list each time a list is yielded

            while True:
                try:
                    second = [
                        threads.submit(imread, infile)
                        for infile in next(it_infiles)
                    ]
                except StopIteration:
                    break
                yield [
                    future.result()
                    for future in first
                ]
                first = second
            yield [
                future.result()
                for future in first
            ]

        size = self.base_tilesize
        writes = []
        shape = self.base_tilesize * step, self.base_tilesize * step, 4
        desc = f"Stitching {len(outfiles):,} tiles..."
        desc = desc.rjust(len(desc) + 11).ljust(50)
        CANVAS = np.zeros(shape, dtype=np.uint8)
        CANVAS[:, :, 3] = 255
        bval = self.black.array[0, 0, :]
        for infiles, outfile in tqdm(
                zip(gen_infiles(), outfiles),
                total=len(outfiles),
                desc=desc,
                # disable when piping
                # disable=not sys.stdout.isatty()
        ):
            canvas = CANVAS.copy()
            for i, infile in enumerate(infiles):
                # the grid is transposed so the row and column indices are swapped
                r = i % step * size
                c = i // step * size
                infile = np.array(infile)
                canvas[r: r + size, c: c + size, : infile.shape[2]] = infile
            loc = canvas[:, :, 3] != 255
            loc |= (canvas[:, :, :3] == 0).all(axis=2)
            canvas[loc, :3] = bval
            canvas = canvas[:, :, :3]
            writes.append(threads.submit(imageio.v3.imwrite, outfile, canvas))

        for write in writes:
            write.result()
        threads.shutdown(wait=True)

    def download(self, retry: bool = True):
        """
        Download tiles from the source.

        Parameters
        ----------
        retry : bool
            When True, tries to serialize tiles a second time if the first time fails
        Returns
        -------
        None
        """
        with (
            ThreadPoolExecutor(max_workers=5) as threads,
            requests.Session() as session,
        ):
            if not self.source:
                return
            self.project.tiles.static.path.mkdir(parents=True, exist_ok=True)

            paths = np.fromiter(self.project.tiles.static.files(), dtype=object, count=self.tiles.size)
            urls = np.fromiter(self.source[self.tiles.flat], dtype=object, count=self.tiles.size)
            loc = [
                not os.path.exists(path)
                for path in paths
            ]
            if any(loc):
                logger.info(f'{sum(loc)} tiles missing out of {len(loc)} total.')
            paths = paths[loc]
            urls = urls[loc]

            def head(url: str) -> int:
                try:
                    response = session.head(url, timeout=10)
                    return response.status_code
                except requests.exceptions.RequestException as e:
                    logger.error(f"Request to {url} failed: {e}")
                    return -1

            futures = threads.map(head, urls)
            codes = np.fromiter(futures, dtype=np.int32, count=len(urls))
            loc = codes == 404
            loc |= codes == 200
            if not np.all(loc):
                logger.warning(
                    f"Unexpected status codes: {np.unique(codes[~loc])}"
                )
            loc = codes == 404

            if loc.any():
                logger.info(f'{loc.sum():,} tiles returned 404 from the server.')
            if loc.any():
                src = self.black.__fspath__()
                for path in paths[loc]:
                    os.symlink(src, path)
                logger.debug(
                    f'{loc.sum():,} tiles were not found and returned 404 from the server '
                    f'so they were set as symlinks to {src}.'
                )
            paths = paths[~loc]
            urls = urls[~loc]

            desc = f"Downloading {len(paths)} files..."
            desc = desc.rjust(len(desc) + 11).ljust(50)
            submit = partial(session.get, verify=certifi.where())
            downloads = {
                threads.submit(submit, url=url): path
                for url, path in tqdm(
                    zip(urls, paths),
                    total=self.tiles.size,
                    desc=desc,
                    # disable when piping
                    # disable=not sys.stdout.isatty()
                )
                if not path.exists()
            }

            def submit(
                    path: Path,
                    content: bytes,
            ) -> Optional[Path]:
                # return path if failed
                path.write_bytes(content)
                try:
                    imageio.v3.imread(path)
                except ValueError:
                    path.unlink()
                    return path

            def write(futures: Iterator[Future]):
                for future in futures:
                    response: requests.Response = future.result()
                    path = downloads[future]
                    try:
                        response.raise_for_status()
                    except requests.exceptions.HTTPError:
                        continue
                    else:
                        yield threads.submit(submit, path, response.content)

            if downloads:
                desc = f"Downloading {len(downloads):,} tiles..."
                desc = desc.rjust(len(desc) + 11).ljust(50)
                writes = pipe(
                    downloads,
                    as_completed,
                    curry(
                        tqdm,
                        total=len(downloads),
                        desc=desc,
                        # disable when piping
                        # disable=not sys.stdout.isatty(),
                    ),
                    write,
                    list,
                )
                if not writes:
                    # todo this was likely caused by downloads raising HTTPError
                    logger.warning(
                        f"Downloads were attempted, but not a single image was written to file. "
                        f"Check that everything is correct for {self.source=}."
                    )
            else:
                writes = []

            failed_paths = pipe(
                writes,
                as_completed,
                curried.map(Future.result),
                curried.filter(Path.__instancecheck__),
                list,
            )
            if any(failed_paths):
                path: Path = failed_paths[0]
                i = next(i for i, p in enumerate(downloads.values()) if p == path)
                url = urls[i]
                if retry:
                    logger.error(
                        f"{len(failed_paths):,} tiles failed to serialize, one of which is "
                        f"{failed_paths[0]} from {url}. Trying again."
                    )
                    self.download(retry=False)
                else:
                    raise FileNotFoundError(
                        f"{len(failed_paths):,} tiles failed to serialize, one of which is "
                        f"{failed_paths[0]} from {url}."
                    )
            logger.info(
                f"All {self.tiles.size} tiles are on disk.",
            )

    def save_info_json(self, **kwargs):
        """
        Saves the grid info as a json file

        Parameters
        ----------
        kwargs
            new_tstep : int
                if the tile size is changed, the new tile size is saved in the json file
            return_dict : bool
                whether or not to return the updated grid info as a dict instead of saving to a json file
        """
        city_info = dict(
            name=self.name,
            bbox=self.bbox,
            location=self.location,
            size=self.tile_size,
            zoom=self.zoom,
            crs=self.crs,
            tile_step=self.tile_step,
            project=dict(self.project.structure),
            debug=self.debug,
        )
        if self.output_dir:
            city_info["output_dir"] = str(self.output_dir)
        if self.input_dir:
            city_info["input_dir"] = str(self.input_dir)
        if self.source:
            city_info["source"] = str(self.source)
            city_info["year"] = self.source.year
            city_info['server'] = self.source.server

        if "new_tstep" in kwargs:
            city_info["size"] = self.tile_size * kwargs["new_tstep"]
            city_info["tile_step"] = kwargs["new_tstep"]

        if "return_dict" in kwargs and kwargs["return_dict"]:
            city_info.update(city_info["project"])
            return city_info
        else:
            self.write_json_file(self.project.tiles.info, city_info)
            self.write_json_file(
                self.project.structure.__fspath__(), dict(self.project.structure)
            )

    def write_json_file(self, file_path, data):
        """
        writes a json file

        Parameters
        ----------
        file_path : str
            path to the json file
        data : dict
            data to be written
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w+") as f:
            json.dump(data, f, indent=4)

    # @validate
    def generate(self, step):
        """
        generates the project structure,
        creates the tiles and saves the info json file

        Parameters
        ----------
        step : int
            tile step to stitch the tiles
        """
        self.stitch(step)
        self.save_info_json(new_tstep=step)
        self.save_csv()
        logger.info(f"Dumping to {self.project.tiles.info}")
        json.dump(
            dict(self.project.structure),
            fp=sys.stdout,
            allow_nan=False,
            indent=4,
        )

    # todo: it would have ben better to not use subprocess.run however the default args are hard-coded
    #   to the commandline wrapper, thus it's hard to do this without the commandline call

    # @validate
    def inference(self, *args: str, ):
        """
        runs the inference on the tiles

        Parameters
        ----------
        *args: str
            arguments to be passed to the commandline inference
        """
        info = toolz.get_in(
            "project tiles info".split(),
            self.save_info_json(return_dict=True),
            no_default=True,
        )
        args = [
            "python",
            "-m",
            "tile2net",
            "inference",
            *args,
        ]
        sargs = set(args)
        extend = getattr(args, 'extend')
        if '--city_info' not in sargs:
            extend(["--city_info", str(info)])
        if '--dump_percent' not in sargs:
            extend(["--dump_percent", str(self.dump_percent)])
        if '--interactive' not in sargs:
            args.append("--interactive")
        logger.debug(f'Running {" ".join(args)}')
        try:
            # todo: capture_outputs=False if want instant printout
            subprocess.run(
                args,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Command {e.cmd} returned non-zero exit status {e.returncode}.\n"
                f"Stdout: {e.stdout}\n"
                f"Stderr: {e.stderr}"
            )
            raise e

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def __truediv__(self, n: int) -> list[Raster]:
        """

        Parameters
        ----------
        n : int
            Number of parts to divide the Raster into

        Returns
        -------
        list[Raster]
            List of Rasters divided by columns into n parts,
            with the last part being the remainder

        """
        tiles = self.tiles
        r, c = tiles.shape
        STEP = math.ceil(c / n)
        ceil = math.ceil(c / STEP) * STEP + 1
        steps = np.arange(0, ceil, STEP)
        steps[-1] = c
        starts = steps[:-1]
        stops = steps[1:]

        rasters = []
        for i, (start, stop) in enumerate(zip(starts, stops)):
            raster = copy.copy(self)
            raster.batch = i
            raster.tiles = tiles[:, start:stop]
            raster.xtile = self.xtile + start
            raster.xtilem = self.xtile + stop
            rasters.append(raster)

        assert rasters[0].xtile == self.xtile
        assert rasters[-1].xtilem == self.xtilem
        assert all(
            raster.ytile == self.ytile
            and raster.ytilem == self.ytilem
            for raster in rasters
        )
        assert sum(
            raster.tiles.size
            for raster in rasters
        ) == self.tiles.size
        assert len(rasters) == n
        return rasters

    @property
    def batches(self) -> Iterator[range]:
        # todo: compute based on available memory
        n = 50

        tiles = self.tiles
        r, c = tiles.shape
        STEP = math.ceil(c / n)
        ceil = math.ceil(c / STEP) * STEP + 1
        steps = np.arange(0, ceil, STEP)
        steps[-1] = c
        starts = steps[:-1]
        stops = steps[1:]
        # noinspection PyTypeChecker
        yield from map(range, starts, stops)

    @cached_property
    def extension(self):
        if self.source:
            return self.source.extension
        elif self.input_dir:
            return self.input_dir.extension
        else:
            raise ValueError("No source or input_dir specified")


if __name__ == '__main__':
    location = "42.35555189953313, -71.07168915322092, 42.35364837213307, -71.06437423368418"
    raster = Raster(
        location=location,
        name='example',
        output_dir='',
    )
    raster.generate(2)
    raster.inference()
