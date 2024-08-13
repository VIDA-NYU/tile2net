import glob
import json
import os
import shutil
import tempfile
import warnings
from collections import UserDict
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import Iterator, Union
from weakref import WeakKeyDictionary

import gdown
import numpy as np
import psutil
import toolz.curried
from more_itertools import *
from numpy import ndarray
from toolz.curried import *

if False:
    from tile2net.raster.raster import Raster
    from tile2net.raster.source import Source


class directory_property(property):
    def __init__(self, func):
        super().__init__(func)
        self.cache: WeakKeyDictionary = WeakKeyDictionary()

    # noinspection PyMethodOverriding
    def __get__(self, instance, owner):
        if instance is None:
            return self
        if instance not in self.cache:
            self.cache[instance] = super().__get__(instance, owner)
        return self.cache[instance]

    def __set__(self, instance, value):
        self.cache[instance] = value

    def __delete__(self, instance):
        del self.cache[instance]


def directory_method(func):
    cache: WeakKeyDictionary = WeakKeyDictionary()

    def wrapper(*args, **kwargs):
        self = args[0]
        if self not in cache:
            cache[self] = func(*args, **kwargs)
        return cache[self]

    return wrapper


class Directory:
    # @directory_method

    def files(self, **kwargs) -> list[Path]:
        raise NotImplementedError

    # @directory_property
    @property
    def path(self) -> Path:
        return Path(self)

    path: Path

    @property
    def ends(self):
        # returns all directories that are not parents of other directories
        stack = [self]
        ends = []
        while stack:
            child = stack.pop()
            if child.is_end:
                ends.append(child)
            else:
                stack.extend(child)
        return ends

    @property
    def is_end(self) -> bool:
        # returns True if the directory is not a parent of other directories
        return not self._children

    @property
    def project(self) -> 'Project':
        # returns the project instance
        child: Project | Directory = self
        while not isinstance(child, Project):
            child = child.parent
        return child

    @property
    def _children(self):
        return list(self)

    def __iter__(self) -> Iterator['Directory']:
        yield from (
            getattr(self, key)
            for key, value in self.__class__.__dict__.items()
            if isinstance(value, Directory)
        )

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance: 'Directory', owner):
        self.parent = instance
        self.owner = owner
        return self

    # @directory_method
    def __fspath__(self):
        return os.path.join(self.parent, self.name)

    def __hash__(self):
        project = self.project
        return hash((
            id(self),
            id(project)
        ))

    def __eq__(self, other):
        return (
                self is other
                and self.project is other.project
        )

    def __repr__(self):
        return self.__fspath__()

    @property
    def folders(self) -> Iterator[tuple[str, str]]:
        # yields all folders in the directory
        for path in glob.iglob(f'{self.__fspath__()}{os.sep}/*'):
            name = path.rpartition(os.sep)[-1]
            if '.' in name:
                continue
            yield name, path


class File(Directory):
    def __init__(self, extension: str):
        self.extension = extension

    def __fspath__(self):
        return super().__fspath__() + self.extension

    def files(self, raster: 'Raster', **kwargs) -> list[Path]:
        raise NotImplementedError


class Weights(Directory):
    satellite_2021 = File('.pth')
    hrnetv2_w48_imagenet_pretrained = File('.pth')

    @staticmethod
    def download():
        url = 'https://drive.google.com/drive/folders/1cu-MATHgekWUYqj9TFr12utl6VB-XKSu'
        Project.resources.assets.weights.path.mkdir(parents=True, exist_ok=True)
        gdown.download_folder(
            url=url,
            quiet=True,
            output=Project.resources.assets.weights.__fspath__(),
        )

    def __get__(self, instance, owner) -> 'Weights':
        # noinspection PyTypeChecker
        return super().__get__(instance, owner)

class Segmentation(Directory):
    def __fspath__(self):
        raster = self.project.raster
        return os.path.join(
            super().__fspath__(),
            self.project.raster.name,
            f'{raster.base_tilesize}_{raster.zoom}_{raster.stitch_step}',
        )

    def files(self, tiles: ndarray = None) -> Iterator[str]:
        if tiles is None:
            tiles = self.project.raster.tiles
        path = self.path
        path = path.__fspath__()
        R, C = np.meshgrid(
            np.arange(tiles.shape[0]),
            np.arange(tiles.shape[1]),
            indexing='ij'
        )
        extension = 'npy'
        for i, (r, c) in enumerate(zip(R.flat, C.flat)):
            yield os.path.join(path, f'{r}_{c}_{i}.{extension}')



class Assets(Directory):
    # @directory_method
    def files(self, raster: 'Raster', **kwargs) -> list[Path]:
        raise RuntimeError

    def __get__(self, instance, owner) -> 'Assets':
        # noinspection PyTypeChecker
        return super().__get__(instance, owner)

    weights = Weights()



class Config(File):
    # @directory_method
    def __fspath__(self):
        # returns path of one and only one config file
        return self.project.tile2net.joinpath(
            # 'tile2net',
            'tileseg',
            f'{self.name}.py',
        ).__fspath__()

    @property
    def temp(self) -> Path:
        # returns path of temporary config file raw config is overwritten
        # def replace(lines: Iterator[str]):
        #     # noinspection PyTypeChecker
        #     project: Project = self.project
        #     # todo: probably just do away with project.config
        #     raster = self.project.raster
        #     # stitched = project.tiles.stitched.path / f'{raster.base_tilesize}_{raster.zoom}_{raster.stitch_step}'
        #     stitched = project.tiles.stitched.path
        #     raise NotImplementedError
        #     # mapping: dict[str, str] = {
        #     #     '__C.ASSETS_PATH =': f" r\'{project.resources.segmentation.assets.path.absolute()}\'",
        #     #     '__C.RESULT_DIR =': f" r\'{project.resources.segmentation.results.path.absolute()}\'",
        #     #     '__C.CITY_INFO_PATH =': f" r\'{project.tiles.info.path.absolute()}\'",
        #     #     '__C.MODEL.SNAPSHOT': f" r\'{project.resources.segmentation.assets.weights.satellite_2021.path.absolute()}\'",
        #     #     '__C.MODEL.HRNET_CHECKPOINT': f" r\'{project.resources.segmentation.assets.weights.hrnetv2_w48_imagenet_pretrained.path.absolute()}\'",
        #     #     '__C.EVAL_FOLDER': f" r\'{stitched.absolute()}\'",
        #     # }
        #     # visited: set[str] = set()
        #     # pattern = re.compile(r'^__C.*=')
        #     #
        #     # for i, line in enumerate(lines):
        #     #     match = pattern.search(line)
        #     #     if match is not None:
        #     #         key = match.group()
        #     #         if key not in mapping:
        #     #             yield line
        #     #             continue
        #     #         if key in visited:
        #     #             raise ValueError(f"Duplicate \'{key}\' at line {i}")
        #     #         visited.add(key)
        #     #         yield f"{key}{mapping[key]}\n"
        #     #     else:
        #     #         yield line
        #     #

        path = Path(
            tempfile.mkdtemp(),
            self.__fspath__().rpartition(os.sep)[-1]
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            pipe(
                with_iter(open(self, 'r')),
                replace,
                f.writelines,
            )
        return path

    def modify(self):
        raise DeprecationWarning('no loonger necessary due to reworked config/args')
        # replace environ variables with project paths
        self.path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.temp, self)

    def __get__(self, instance, owner) -> 'Config':
        # noinspection PyTypeChecker
        return super().__get__(instance, owner)


class Static(Directory):

    def __get__(self, instance, owner) -> 'Static':
        # noinspection PyTypeChecker
        return super().__get__(instance, owner)

    _directory = WeakKeyDictionary()


    def __fspath__(self):
        raster = self.project.raster
        source: 'Source' = raster.source
        if source:
            return os.path.join(
                super().__fspath__(),
                source.name,
                f'{raster.base_tilesize}_{raster.zoom}'
            )
        elif raster.input_dir:
            return raster.input_dir.__fspath__()
        else:
            raise ValueError('raster has no source or input_dir')


    def files(self, tiles: ndarray = None) -> Iterator[Path]:
        raster = self.project.raster
        if tiles is None:
            tiles = raster.tiles
        if isinstance(tiles, ndarray):
            tiles = tiles.flat
        extension = raster.extension
        # yield from raster.input_dir(tiles)
        if raster.input_dir:
            yield from raster.input_dir(tiles)
        else:
            dir = self.path
            dir.mkdir(parents=True, exist_ok=True)
            yield from (
                dir / f'{tile.xtile}_{tile.ytile}.{extension}'
                for tile in tiles
            )



class Stitched(Directory):

    def files(self, tiles: ndarray = None) -> Iterator[str]:
        if tiles is None:
            tiles = self.project.raster.tiles
        path = self.path
        path.mkdir(parents=True, exist_ok=True)
        path = path.__fspath__()
        R, C = np.meshgrid(
            np.arange(tiles.shape[0]),
            np.arange(tiles.shape[1]),
            indexing='ij'
        )
        extension = self.project.raster.extension
        for i, (r, c) in enumerate(zip(R.flat, C.flat)):
            yield os.path.join(path, f'{r}_{c}_{i}.{extension}')

    def __fspath__(self):
        raster = self.project.raster
        return os.path.join(
            super().__fspath__(),
            f'{raster.base_tilesize}_{raster.zoom}_{raster.stitch_step}',
        )


class Info(File):
    def __fspath__(self):
        raster = self.project.raster
        return os.path.join(
            self.parent,
            f'{raster.name}_{raster.tile_size}_{self.name}{self.extension}'
        )


class Tiles(Directory):
    def __get__(self, instance, owner) -> 'Tiles':
        # noinspection PyTypeChecker
        return super().__get__(instance, owner)

    info = Info('.json')
    static = Static()
    stitched = Stitched()


class StructureDict(UserDict):
    def __init__(self, dct: dict, path: PathLike | str):
        super().__init__(dct)
        self.path = Path(path)

    def dump(self, path: PathLike = None) -> Path:
        # dump the file structure to a json file within the project directory
        if not path:
            path = self.path
        else:
            path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w') as f:
            json.dump(dict(self), f, indent=4)
        return path

    def __repr__(self):
        return json.dumps(dict(self), indent=4)

    def __fspath__(self):
        return self.path.__str__()


class Structure(PathLike):
    # @directory_method
    def __fspath__(self):
        return os.path.join(self.parent, self.name + '.json')

    def __set_name__(self, owner, name):
        self.name = name

    def __init__(self):
        self.cache: WeakKeyDictionary = WeakKeyDictionary()
        self.visited: set[Directory] = set()

    def __get__(self, instance: 'Project', owner) -> Union['Structure', 'StructureDict']:
        instance: Project
        self.parent: Project = instance
        if instance is None:
            return self
        project = instance
        self.visited.clear()
        dct = {
            resource.name: self(resource)
            for resource in project.resources
            if resource not in self.visited
        }
        self.visited.add(project.resources)
        dct.update(self(project))
        dct['name'] = project.name
        res = StructureDict(
            dct=dct,
            path=os.path.join(project.path.__str__(), self.name + '.json')
        )
        return res

    def __call__(self, directory: 'Directory') -> Union[dict, str]:
        # recursviely returns a path if the directory has no subdirectories
        #   or a dict of the subdirectories
        if (
                isinstance(directory, File)
        ):
            return directory.__fspath__()

        if (
                directory.is_end
                and not isinstance(directory, Directory)
        ):
            return directory.__fspath__()

        directory: Directory
        if directory.is_end:
            ret = directory.__fspath__()
        else:
            ret = {
                # return super().__fspath__()
                child.name: self(child)
                for child in directory
                if child not in self.visited
            }

        return ret


class Resources(Directory):
    config = Config('.py')
    assets = Assets()
    segmentation = Segmentation()

    @cached_property
    def path(self):
        path = Path(__file__).parent
        while path.parent.name != 'tile2net':
            path = path.parent
            if not path.name:
                raise FileNotFoundError('Could not find tile2net directory')
        return path / self.name

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance: 'Project', owner) -> 'Resources':
        # noinspection PyTypeChecker
        return super().__get__(instance, owner)

    def __fspath__(self):
        return self.path.__str__()

    def __repr__(self):
        return self.path.__str__()

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return self is other


class Polygons(Directory):
    def files(self) -> list[Path]:
        # defined in grid
        # todo: the current implementation has datetime in the name?
        ...


class Network(Directory):
    def files(self) -> list[Path]:
        # defined under pednet
        # todo: the current implementation has datetime in the name?
        ...


class Project(Directory):
    tiles = Tiles()
    polygons = Polygons()
    network = Network()
    structure = Structure()
    resources = Resources()
    config = Config('.py')
    segmentation = Segmentation()

    def __init__(
        self,
        name: str,
        outdir: PathLike,
        raster: 'Raster',
    ):
        """
        file structure for tile2net project
        :param name:
        :param outdir:
        :param raster:
        """
        # if ' ' in name:
        #     raise ValueError('Avoid spaces in project name')
        # if '.' in name:
        #     raise ValueError('Avoid periods in project name')
        # if len(name) > 31:
        #     raise ValueError(
        #         'Avoid using long names for project since it will '
        #         'be added to beginning of each image name!'
        #     )
        if not outdir:
            outdir = os.path.join(tempfile.gettempdir(), 'tile2net')

        self.name = name
        self.parent = self.resources
        self.outdir = Path(outdir)
        self.raster = raster

        # noinspection PyTypeChecker
        # self.mkdirs()
        # noinspection PyTypeChecker
        # free = psutil.disk_usage(self.__fspath__()).free
        # free = psutil.disk_usage(self.__fspath__()).free
        path = Path(self.__fspath__())
        path.mkdir(parents=True, exist_ok=True)

        # disabled because of python 3.12 bug
        # free = psutil.disk_usage(path.__fspath__()).free
        # if free > 2 * 2 ** 30:
        #     ...
        # elif free > 1 * 2 ** 30:
        #     warnings.warn(f'Low disk space: {free / 2 ** 30} GB')
        # else:
        #     warnings.warn(f'Very low disk space: {free / 2 ** 20} MB')

    def to_file(self, path: PathLike = None) -> Path:
        if path is None:
            path = self.structure
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.structure, f, indent=4)
        return path

    def mkdirs(self):
        # creates all directories in project
        for end in self.ends:
            path = end.path
            if path.suffix:
                path = path.parent
            path.mkdir(parents=True, exist_ok=True)

    def rmdir(self):
        shutil.rmtree(self)

    # @lru_cache()  # todo: lru_cache on methods causes a memory leak
    def __fspath__(self):
        return os.path.join(
            self.outdir.__fspath__(),
            self.name,
        )

    def __get__(self, instance, owner) -> 'Project':
        # noinspection PyTypeChecker
        return self

    def __repr__(self):
        return self.structure.__repr__()

    def __getitem__(self, item):
        return self.structure[item]

    def __setitem__(self, key, value):
        self.structure[key] = value

    tile2net = Path(__file__).parent
    while tile2net.name != 'tile2net':
        tile2net = tile2net.parent
        if not tile2net.name:
            raise FileNotFoundError('Could not find tile2net directory')
