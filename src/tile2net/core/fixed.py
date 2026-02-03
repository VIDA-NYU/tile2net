from __future__ import annotations

import copy

from geopandas.base import is_geometry_type
from geopandas.geodataframe import _ensure_geometry
from geopandas.geoseries import GeoSeries
from pandas.core.generic import NDFrame

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import warnings
from copy import deepcopy
from pandas._typing import (
    Self,
)
from pandas.core.dtypes.common import (
    is_list_like,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
)
from pandas.util._exceptions import find_stack_level
from pathlib import Path
from typing import *
from typing import (
    final,
)
from functools import cached_property
import pandas
from .geox import GeoX


class GeoDataFrameFixed(
    gpd.GeoDataFrame
):
    """
    Do not bother reading this class. It's just fixing some buggy code
    that has been left in (geo)pandas.
    """

    @GeoX
    def geox(self):
        ...

    @final
    def __setattr__(self, name: str, value) -> None:
        # See https://github.com/pandas-dev/pandas/issues/56793
        # I fixed this long ago, but they never accepted it.
        # Pandas' DataFrame has a bug where all `set` will first call `get`

        """
        If it can be found nowhere else than the info_axis, it must be a column.
        This allows simpler access to columns for interactive use.
        """
        if (
                name not in self.__dict__
                and name not in self._internal_names_set
                and name not in self._metadata
                and name in self._info_axis
        ):
            try:
                self[name] = value
            except (AttributeError, TypeError):
                pass
            else:
                return

        if (
                isinstance(self, ABCDataFrame)
                and name not in self.__dict__
                and not hasattr(type(self), name)
                and name not in self._internal_names
                and name not in self._metadata
                and is_list_like(value)
        ):
            warnings.warn(
                "Pandas doesn't allow columns to be "
                "created via a new attribute name - see "
                "https://pandas.pydata.org/pandas-docs/"
                "stable/indexing.html#attribute-access",
                stacklevel=find_stack_level(),
            )
        object.__setattr__(self, name, value)

    @cached_property
    def _constructor(self):
        return type(self)

    def _constructor_from_mgr(self, mgr, axes):
        # In original geopandas it's hard-coded to return a GeoDataFrame
        return self.__class__._from_mgr(mgr, axes)

    @property
    def _constructor_sliced(self):
        def _geodataframe_constructor_sliced(*args, **kwargs):
            """
            A specialized (Geo)Series constructor which can fall back to a
            Series if a certain operation does not produce geometries:

            - We only return a GeoSeries if the data is actually of geometry
              dtype (and so we don't try to convert geometry objects such as
              the normal GeoSeries(..) constructor does with `_ensure_geometry`).
            - When we get here from obtaining a row or column from a
              GeoDataFrame, the goal is to only return a GeoSeries for a
              geometry column, and not return a GeoSeries for a row that happened
              to come from a DataFrame with only geometry dtype columns (and
              thus could have a geometry dtype). Therefore, we don't return a
              GeoSeries if we are sure we are in a row selection case (by
              checking the identity of the index)
            """
            srs = pd.Series(*args, **kwargs)
            is_row_proxy = srs.index is self.columns
            if is_geometry_type(srs) and not is_row_proxy:
                srs = GeoSeries(srs)
            return srs

        return _geodataframe_constructor_sliced

    def __bool__(self):
        return False

    def __eq__(self, other):
        return False

    def __deepcopy__(self, memo) -> Self:
        return self.copy()

    def __set_name__(self, owner, name):
        self.__name__ = name

    @final
    def __finalize__(self, other, method: str | None = None, **kwargs) -> Self:
        # Checking for the attr equality during concat is rarely useful
        # and is quite a nuisance if you're storing stuff like DataFrames

        """
        Propagate metadata from other to self.

        Parameters
        ----------
        other : the object from which to get the attributes that we are going
            to propagate
        method : str, optional
            A passed method name providing context on where ``__finalize__``
            was called.

            .. warning::

               The value passed as `method` are not currently considered
               stable across pandas releases.
        """
        if isinstance(other, NDFrame):
            if other.attrs:
                # We want attrs propagation to have minimal performance
                # impact if attrs are not used; i.e. attrs is an empty dict.
                # One could make the deepcopy unconditionally, but a deepcopy
                # of an empty dict is 50x more expensive than the empty check.
                # self.attrs = deepcopy(other.attrs)
                self.attrs = copy.copy(other.attrs)

            self.flags.allows_duplicate_labels = other.flags.allows_duplicate_labels
            # For subclasses using _metadata.
            for name in set(self._metadata) & set(other._metadata):
                assert isinstance(name, str)
                object.__setattr__(self, name, getattr(other, name, None))

        if method == "concat":
            # propagate attrs only if all concat arguments have the same attrs
            # if all(bool(obj.attrs) for obj in other.objs):
            #     # all concatenate arguments have non-empty attrs
            #     attrs = other.objs[0].attrs
            #     have_same_attrs = all(obj.attrs == attrs for obj in other.objs[1:])
            #     if have_same_attrs:
            #         self.attrs = deepcopy(attrs)

            allows_duplicate_labels = all(
                x.flags.allows_duplicate_labels for x in other.objs
            )
            self.flags.allows_duplicate_labels = allows_duplicate_labels

        return self

    # Allows using this class as a decorator
    def __init__(
            self,
            *args,
            **kwargs,
    ):
        if (
                args
                and callable(args[0])
        ):
            super().__init__(*args[1:], **kwargs)
        else:
            super().__init__(*args, **kwargs)


_getattr = pandas.core.generic.NDFrame.__getattr__


def _getattribute(self: pandas.core.generic.NDFrame, name: str):
    """
    After regular attribute access, try looking up the name
    This allows simpler access to columns for interactive use.
    """
    # Note: obj.x will always call obj.__getattribute__('x') prior to
    # calling obj.__getattr__('x').
    try:
        result = object.__getattribute__(self, name)
    except AttributeError as e:
        if hasattr(self.__class__, name):
            raise
        result = _getattr(self, name)
    return result


del pandas.core.generic.NDFrame.__getattr__
pandas.core.generic.NDFrame.__getattribute__ = _getattribute


def __finalize__(self, other, method: str | None = None, **kwargs) -> Self:
    """
    Propagate metadata from other to self.

    Parameters
    ----------
    other : the object from which to get the attributes that we are going
        to propagate
    method : str, optional
        A passed method name providing context on where ``__finalize__``
        was called.

        .. warning::

           The value passed as `method` are not currently considered
           stable across pandas releases.
    """
    if isinstance(other, NDFrame):
        # if other.attrs:
        #     # We want attrs propagation to have minimal performance
        #     # impact if attrs are not used; i.e. attrs is an empty dict.
        #     # One could make the deepcopy unconditionally, but a deepcopy
        #     # of an empty dict is 50x more expensive than the empty check.
        #     self.attrs = deepcopy(other.attrs)

        self.flags.allows_duplicate_labels = other.flags.allows_duplicate_labels
        # For subclasses using _metadata.
        for name in set(self._metadata) & set(other._metadata):
            assert isinstance(name, str)
            object.__setattr__(self, name, getattr(other, name, None))

    if method == "concat":
        # propagate attrs only if all concat arguments have the same attrs
        # if all(bool(obj.attrs) for obj in other.objs):
        #     # all concatenate arguments have non-empty attrs
        #     attrs = other.objs[0].attrs
        #     have_same_attrs = all(obj.attrs == attrs for obj in other.objs[1:])
        #     if have_same_attrs:
        #         self.attrs = deepcopy(attrs)

        allows_duplicate_labels = all(
            x.flags.allows_duplicate_labels for x in other.objs
        )
        self.flags.allows_duplicate_labels = allows_duplicate_labels

    return self


pandas.core.generic.NDFrame.__finalize__ = __finalize__
