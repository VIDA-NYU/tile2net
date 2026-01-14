from __future__ import annotations

import pathlib
from functools import cached_property
from typing import *

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries

from tile2net.logger import logger

if TYPE_CHECKING:
    from .remote import Remote


class RemoteNotFound(Exception):
    """Raised when no appropriate remote source can be found for a location."""
    ...


class Coverage(
):
    """
    Descriptor providing spatial coverage lookup for Remote sources.

    Lazily loads coverage geometries from all registered Remote subclasses,
    caches them to disk, and provides location-based remote selection.
    """

    @cached_property
    def file(self) -> pathlib.Path:
        """Path to cached coverage feather file."""
        # todo: has bad code smell?
        return pathlib.Path(
            __file__, '..', '..', 'resources', 'coverage.feather'
        ).resolve()

    def __get__(
            self,
            instance: None,
            owner: type[Remote]
    ) -> Union[Self, GeoSeries]:
        """
        Returns a GeoSeries of all remote coverages, indexed by remote name.
        Caches the result to disk for faster subsequent access.
        """

        if instance is not None:
            msg = (
                f'Coverage descriptor should be accessed on the class. \n'
                f'You may have forgotten to override `coverage` for a concrete class.'
            )
            raise ValueError(msg)

        if (
                self.manifest is None
                and self.file.exists()
        ):
            # try loading from file
            coverage = gpd.read_feather(self.file)
            if (
                    coverage.index
                            .symmetric_difference(owner.catalog.keys())
                            .empty
            ):
                self.manifest = coverage.geometry

        if self.manifest is None:
            # instantiate from catalog
            coverages: list[GeoSeries] = []
            for remote in owner.catalog.values():
                if remote.ignore:
                    continue

                try:
                    coverage = remote.prototype.coverage
                except Exception as e:
                    logger.error(
                        f'Could not get coverage for {remote.prototype.name},'
                        f' skipping:\n\t'
                        f'{e}'
                    )
                    continue
                if not (
                        isinstance(coverage, (GeoSeries, GeoDataFrame))
                        and not coverage.empty
                ):
                    msg = f'Coverage for {remote.prototype.name} is invalid, skipping.'
                    logger.warning(msg)
                    continue

                data = np.full(len(coverage), remote.prototype.name)
                axis = pd.Index(data, name='remote')
                coverage = (
                    coverage
                    .set_crs('epsg:4326', allow_override=True)
                    .set_axis(axis)
                )
                coverages.append(coverage)

            data = dict(geometry=pd.concat(coverages))
            coverage = GeoDataFrame(data, crs='epsg:4326')
            self.file.parent.mkdir(parents=True, exist_ok=True)
            coverage.to_feather(self.file)
            self.manifest = coverage.geometry

        return self.manifest

    # locals().update(__get__=_get)

    def __set_name__(self, owner, name):
        self.__name__ = name
        self.manifest = None

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        """Empty init just to allow use as a decorator."""
