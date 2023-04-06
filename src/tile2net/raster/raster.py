import inspect
import subprocess
import os

import geopy
import imageio.v2
from geopy.geocoders import Nominatim
import sys

import numpy as np
import pandas as pd
import requests
import itertools

from tqdm import tqdm
# from itertools import *
from more_itertools import *
from typing import Iterator, Union

from pathlib import Path
from os import PathLike as _PathLike
import json
from concurrent.futures import ThreadPoolExecutor, Future, as_completed

from toolz import *

from PIL import Image
import cv2
import matplotlib.pyplot as plt

from tile2net.raster.grid import Grid
from tile2net.raster.tile_utils.genutils import createfolder
from tile2net.raster.tile_utils.geodata_utils import prepare_gdf, read_dataframe, prepare_spindex
from tile2net.raster.project import Project
from tile2net.raster.source import Source
import toolz
from toolz import curried, pipe
import tile2net.raster.util

# import logging
from tile2net.logger import logger


PathLike = Union[str, _PathLike]

class Raster(Grid):
    Project = Project

    @classmethod
    def from_nyc(cls, outdir: PathLike = None) -> 'Raster':
        """
        Create a Raster object for NYC
        Parameters
        ----------
        outdir: PathLike
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
            name='nyc',
            output_dir=outdir,
        )

    @classmethod
    def from_info(cls, info: PathLike | dict) -> 'Raster':
        """
        Create a Raster object from a json file
        Parameters
        ----------
        info: PathLike | dict
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
            inspect.signature(cls.__init__).parameters.__contains__,
            kwargs
        )
        res = cls(**relevant)
        return res

    @classmethod
    def from_stitched(cls, info: PathLike | dict, tile_step=4) -> 'Raster':
        """
        Create a Raster object from stitched tiles
        Parameters
        ----------
        info: PathLike | dict
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
        tile_step: int
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
            location=kwargs['location'],
            name=kwargs['name'],
            output_dir=kwargs['output_dir'],
            # source=kwargs['source'],
            tile_step=tile_step
        )

    def __init__(
        self,
        *,
        location: list | str,  # region of interest to get its bounding box
        name: str,
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
        extension: str = 'png',
    ):
        """

        Parameters
        ----------
        location: list | str
            region of interest to get its bounding box
        name: str
            name of the project
        input_dir: PathLike
            path to the directory containing the input images
        output_dir: PathLike
            path to the directory containing the output images
        num_class: int
            # of classes for annotation creation
        base_tilesize: int
            size of the base tile in pixels (default: 256)
        zoom: int
            zoom level of the tiles (default: None)
        crs: int
            coordinate reference system (default: 4326)
        tile_step: int
            step size for the tiles (default: 1)
        boundary_path: str
            path to a shapefile to filter out of boundary tiles (default: None)
        padding: bool
            whether to pad the tiles to the base tile size (default: True)
        extension:
            extension of the input images (default: 'png')
        """
        if isinstance(location, str):
            try:
                location: list[float] = pipe(
                    location.split(','),
                    curried.map(float),
                    list,
                    self.round_loc
                )
            except (ValueError, AttributeError):  # fails if address or list
                nom: geopy.Location = Nominatim(user_agent='tile2net').geocode(location)
                logger.info(f"Geocoded '{location}' to\n\t'{nom.raw['display_name']}'")
                location = pipe(
                    nom.raw['boundingbox'],
                    # convert lon, lon, lat, lat
                    # to lat, lon, lat, lon
                    curried.get([0, 2, 1, 3]),
                )
        location = pipe(
            location,
            curried.map(float),
            list,
            self.round_loc,
            tile2net.raster.util.southwest_northeast,
        )

        self.input_dir = input_dir
        if input_dir is None:
            try:
                self.source = Source[location]
            except KeyError:
                logger.warning('No source found for this location. ')
                self.source = None
            else:
                if zoom is None:
                    zoom = self.source.zoom
        else:
            self.source = None
        if zoom is None:
            raise ValueError('Zoom level must be specified')

        # if zoom is not None:
        #     zoom = zoom

        if base_tilesize < 256:
            raise ValueError(
                'Tile sizes cannot be smaller than 256')
        if not base_tilesize % 256 == 0:
            raise ValueError(
                'Tile size must be a multiple of 256'
            )

        self.location = location
        self.extension = extension
        self.num_class = num_class
        self.class_names = []
        self.class_colors = []
        self.class_order = []
        self.dest = ''
        self.name = name
        self.boundary_path = ''
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
        # project.mkdirs()

    def __repr__(self):
        if self.boundary_path != -1:
            tiles_within = f'{(self.num_inside / self.num_tiles) * 100:.1f}'
            return f"{self.name} Data Constructor. \nCoordinate reference system (CRS): {self.crs} \n" \
                   f"Tile size (pixel): {self.base_tilesize} \nZoom level: {self.zoom} \n" \
                   f"Number of columns: {self.width:,} \n" \
                   f"Number of rows: {self.height:,} \n" \
                   f"Total tiles: {self.num_tiles:,} \n" \
                   f"Number of tiles inside the boundary: {self.num_inside:,} ({tiles_within}%) \n"
        else:
            return f"{self.name} Data Constructor. \nCoordinate reference system (CRS): {self.crs} \n" \
                   f"Tile size (pixel): {self.base_tilesize} \nZoom level: {self.zoom} \n" \
                   f"Number of columns: {self.width:,} \n" \
                   f"Number of rows: {self.height:,} \n" \
                   f"Total tiles: {self.num_tiles:,} \n"

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
        img_path: str
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
            raise ValueError('Input Tile Image Height and Width values should be the same!')
        elif im.size[0] == im.size[1] and im.size[1] != self.base_tilesize:
            raise ValueError('Input Tile image Height and Width values does not match the'
                             ' Grid tile size "{self.tile_size}"!')
        else:
            return False

    """
    Stitch Tiles
    """

    def stitch(self, step: int) -> None:
        """Stitch tiles
        Args:
            step (int): Stitch step. How many adjacent tiles on row/colum to stitch together.
                For instance, to get a 512x512 tiles from the 256x256 base, the stitch step is 2
            extension (str): File extension of the tiles. Default is 'png'.
            loc_abr (str): The abbreviation of the area/state to download the missing tiles when padding.
                If None, missing tiles will be replaced with gray tiles.
        Returns
        ------
        None.
            Nothing is returned.
        """
        logger.info(f'Starting Stitching Tiles...')
        self.stitch_step = step
        self.calculate_padding()
        self.update_tiles()
        self.download()
        self.project.tiles.stitched.path.mkdir(parents=True, exist_ok=True)
        if not (
                self.source
                or self.input_dir
        ):
            raise RuntimeError(
                'No source or input directory specified. Cannot stitch tiles.'
            )
        # todo: only stitch tiles that are unstitched
        outfiles = pipe(
            self.tiles[::step, ::step],
            self.project.tiles.stitched.files,
            list
        )
        not_exists = [
            not os.path.exists(outfile)
            for outfile in outfiles
        ]
        outfiles = list(itertools.compress(outfiles, not_exists))
        if not outfiles:
            logger.info(f'All tiles already stitched.')
            return
        # logger.info(f'Stitching {len(outfiles):,} tiles...'.ljust(25))
        infiles: np.ndarray = pipe(
            self.tiles,
            self.project.tiles.static.files,
            toolz.curry(np.fromiter, dtype=object)
        )

        # for file in infiles:
        #     if not os.path.exists(file):
        #         raise FileNotFoundError(
        #             f'File {file} does not exist. Cannot stitch tiles. '
        #         )
        indices = np.arange(self.height * self.width).reshape((self.width, self.height))
        indices = (
            indices
            # iterate by step to get the top left tile of each new merged tile
            [::step, ::step]
            # reshape to broadcast so offsets can be added
            .reshape((-1, 1, 1))
            # add offsets to get the indices of the tiles to merge
            .__add__(indices[:step, :step])
            # flatten to get a list of merged tiles
            .reshape((-1, step * step))
                # filter for tiles that are not stitched
            [not_exists]
        )
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
                f'No relevant tiles found in {self.project.tiles.static.path}. '
            )
        sample: np.ndarray = next(
            imageio.v2.imread(file)
            for file in itertools.chain.from_iterable(list_infiles)
            if os.path.exists(file)
        )
        if sample.shape[:2] != (self.base_tilesize, self.base_tilesize):
            raise ValueError(
                f'Input tile size {sample.shape[:2]} does not match '
                f'expected tile size {self.base_tilesize}.'
            )

        gray = np.full((self.base_tilesize, self.base_tilesize, 3), 50, dtype=np.uint8)

        def imread(file) -> np.ndarray:
            # just returns a gray tile if the file doesn't exist
            if not os.path.exists(file):
                return gray
            return imageio.v2.imread(file)

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
        shape = self.base_tilesize * step, self.base_tilesize * step, 3
        desc = f"Stitching {len(outfiles):,} tiles..."
        desc = desc.rjust(len(desc) + 11).ljust(50)
        # shape = imageio.imread_v2(infiles[0]).shape
        for infiles, outfile in tqdm(
                zip(gen_infiles(), outfiles),
                total=len(outfiles),
                desc=desc,
                # disable when piping
                # disable=not sys.stdout.isatty()
        ):
            canvas: np.ndarray = np.zeros(shape, dtype=np.uint8)
            for i, infile in enumerate(infiles):
                # the grid is transposed so the row and column indices are swapped
                r = i % step * size
                c = i // step * size
                canvas[r:r + size, c:c + size] = np.array(infile)[:, :, :3]
            writes.append(threads.submit(imageio.v2.imwrite, outfile, canvas))

        for write in writes:
            write.result()
        threads.shutdown(wait=True)

    """
    Download Tiles 
    """
    def download(self):
        """
        Download tiles from the source.
        Returns
        -------
        None
        """
        self.project.tiles.static.path.mkdir(parents=True, exist_ok=True)
        with (
            ThreadPoolExecutor(max_workers=5) as threads,
            requests.Session() as session,
        ):
            writes = []
            if not self.source:
                pipe(
                    writes,
                    as_completed,
                    curried.map(Future.result),
                    consume,
                )
                return

            paths = self.project.tiles.static.files()
            urls = self.source[self.tiles]
            desc = f"Checking {self.tiles.size:,} files..."
            desc = desc.rjust(len(desc) + 11).ljust(50)
            downloads = {
                threads.submit(session.get, url): path
                for url, path
                in tqdm(
                    zip(urls, paths),
                    total=self.tiles.size,
                    desc=desc,
                    # disable when piping
                    # disable=not sys.stdout.isatty()
                )
                if not path.exists()
            }


            def write(futures: Iterator[Future]):
                for future in futures:
                    response: requests.Response = future.result()
                    path = downloads[future]
                    try:
                        response.raise_for_status()
                    except requests.exceptions.HTTPError:
                        continue
                    else:
                        yield threads.submit(path.write_bytes, response.content)

            if downloads:
                desc = f"Downloading {len(downloads):,} tiles..."
                desc = desc.rjust(len(desc) + 11).ljust(50)
                pipe(
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
                    curried.map(writes.append),
                    consume,
                )

            pipe(
                writes,
                as_completed,
                curried.map(Future.result),
                consume,
            )
            logger.info(f'All {self.tiles.size} tiles are on disk.', )


    def create_mask(self, dest_path=None, **kwargs) -> None:
        """
        create the annotation label masks for the tiles
        Parameters
        ----------
        dest_path: str
            path to the folder where the masks will be saved
        kwargs:
            class: dict
            path: str
                path to the shapefile
            usecols: list
                list of columns to be used
            col: dict
                column name to be used as the label
            if the column is a string, the label will be the same as the column name
            if the column is a dict, the label will be the value of the key

        Returns
        -------
        None
        """
        logger.info(f'{self.num_tiles} annotation masks will be created')


        # stitched_dir_name = f'{self.name}_stitched-{self.base_tilesize * step}'
        # dest_path = createfolder(os.path.join(tile_group_dir_path, stitched_dir_name))

        urb_gdf = []
        inds = []
        for c, cls in enumerate(kwargs):
            if isinstance(kwargs[cls], dict):
                cols = kwargs[cls]['usecols']
                gdf = read_dataframe(kwargs[cls]['path'], cols=cols, geo=True)
                if isinstance(kwargs[cls]['col'], dict):
                    gdf = prepare_gdf(gdf, **kwargs[cls]['col'])
                urb_gdf.append(gdf)
                inds.append(c)
            assert isinstance(kwargs[cls],
                              dict), f"incorrect class config file, expected a path for class {cls}."

        for c, tile in enumerate(self.tiles.flatten()):
            if tile.active:
                idd = tile.idd
                pos = tile.position
                tile.setLatlon
                img = np.zeros([tile.size, tile.size, 3], np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                dpi = 1200
                # prepare the canvas
                # fixme this is not efficient - pasting arrays should be a better way
                fig, ax = plt.subplots(
                    figsize=((img.shape[0] / float(dpi)), (img.shape[1] / float(dpi))))
                plt.box(False)
                fig.dpi = dpi
                fig.tight_layout(pad=0)
                fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
                ax.margins(0)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.set_facecolor('black')
                for c2, ugdf in enumerate(urb_gdf):
                    spind = prepare_spindex(ugdf)
                    ucls = tile.get_region(ugdf, spind)
                    if isinstance(ucls, pd.DataFrame):
                        name = list(kwargs.keys())[inds[c2]]
                        color = kwargs[name]['color']
                        zorder = kwargs[name]['order']
                        ucls.plot(ax=ax, color=color, alpha=1, zorder=zorder, antialiased=False)

                top, left, bottom, right = tile.get_metric()
                ax.imshow(img, extent=[top, bottom, right, left])

                s, (width, height) = fig.canvas.print_to_buffer()
                data = np.frombuffer(s, dtype=np.uint8).reshape(width, height, 4)
                data = data[:, :, 0:3]
                save_path = os.path.join(dest_path, f'annotations')
                createfolder(save_path)

                cv2.imwrite(os.path.join(save_path, f'{pos[0]}_{pos[1]}_{idd}.png'),
                    cv2.cvtColor(data, cv2.COLOR_RGB2BGR))
                fig.clf()
                plt.close('all')
                if c % 20 == 0:
                    logger.info(f'{c} of {self.num_tiles}')
            else:
                continue


    def save_info_json(self, **kwargs):
        """
        saves the grid info as a json file
        Parameters
        ----------
        kwargs
            new_tstep: int
                if the tile size is changed, the new tile size is saved in the json file
            return_dict: bool
        """
        city_info = {
            'name': self.name,
            'bbox': self.bbox,
            'location': self.location,
            'size': self.tile_size,
            'zoom': self.zoom,
            'crs': self.crs,
            'tile_step': self.tile_step,
            'project': dict(self.project.structure),
            'source': str(self.source),
        }
        if self.output_dir:
            city_info['output_dir'] = str(self.output_dir)
        if self.input_dir:
            city_info['input_dir'] = str(self.input_dir)

        if 'new_tstep' in kwargs:
            city_info['size'] = self.tile_size * kwargs['new_tstep']
            city_info['tile_step'] = kwargs['new_tstep']

        if 'return_dict' in kwargs and kwargs['return_dict']:
            city_info.update(city_info['project'])
            return city_info
        else:
            self.write_json_file(self.project.tiles.info, city_info)
            self.write_json_file(self.project.structure.__fspath__(), dict(self.project.structure))

    def write_json_file(self, file_path, data):
        """
        writes a json file
        Parameters
        ----------
        file_path: str
            path to the json file
        data: dict
            data to be written
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w+') as f:
            json.dump(data, f, indent=4)

    def generate(self, step):
        """
        generates the project structure,
        creates the tiles and saves the info json file

        Parameters
        ----------
        step: int
            tile step to stitch the tiles
        """
        self.stitch(step)
        self.save_info_json(new_tstep=step)
        logger.info(f'Dumping to {self.project.tiles.info}')
        json.dump(
            dict(self.project.structure),
            fp=sys.stdout,
            allow_nan=False,
            indent=4,
        )

    def inference(
        self,
        eval_folder: str = None,
    ):
        """
        runs the inference on the tiles
        Parameters
        ----------
        eval_folder: str
            path to the folder containing the images to run inference on
        """
        info = toolz.get_in(
            'project tiles info'.split(),
            self.save_info_json(return_dict=True),
            no_default=True,
        )
        args = [
            'python',
            Path(__file__).parent.parent.absolute().__str__(),
            'inference',
            '--city_info',
            str(info),
        ]
        if eval_folder:
            args.extend(['--eval_folder', str(eval_folder)])
        subprocess.run(args, check=True)
