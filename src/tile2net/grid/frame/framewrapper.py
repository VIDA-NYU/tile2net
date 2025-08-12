from __future__ import annotations

from typing import (
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    pass

from typing import (
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from pandas._typing import (
        Self,
    )

from tile2net.grid.frame.namespace import namespace

import geopandas as gpd


import copy
from functools import *
from typing import *

import pandas as pd
from .wrapper import Wrapper

class Loc:
    instance: FrameWrapper

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        ...


    def __get__(
            self,
            instance,
            owner
    ):
        self.instance = instance
        return copy.copy(self)
    def __set_name__(self, owner, name):
        self.__name__ = name

    def __getitem__(self, item):
        result = (
            getattr(self.instance.frame, self.__name__)
            [item]
            .pipe(self.instance.from_frame, self.instance)
        )
        return result







class FrameWrapper(
    Wrapper,
    namespace,
):
    frame: pd.DataFrame | gpd.GeoDataFrame

    # def __getattr__(self, item):
    #     return getattr(self.frame, item)


    # def _repr_data_resource_(self):
    #     return self.frame._repr_data_resource_()
    #
    # def _repr_fits_vertical_(self) -> bool:
    #     return self.frame._repr_fits_vertical_()
    #
    # def _repr_fits_horizontal_(self) -> bool:
    #     return self.frame._repr_fits_horizontal_()
    #
    # def _repr_html_(self) -> str | None:
    #     return self.frame._repr_html_()

    def __init__(
            self,
            frame: Union[
                gpd.GeoDataFrame,
                pd.DataFrame,
            ] = None,
            *args,
            **kwargs
    ):
        super().__init__(frame, *args, **kwargs)
        if frame is None:
            self.frame = gpd.GeoDataFrame()
        elif isinstance(frame, (pd.DataFrame, gpd.GeoDataFrame)):
            self.frame = frame

    @property
    def index(self):
        return self.frame.index

    def _iloc(self, item) -> Self:
        frame = self.frame.iloc[item]
        result = self.copy()
        result.frame = frame
        return result

    @property
    def iloc(self) -> Self:
        """
        Wrapper for self.frame.iloc.
        self.iloc[...] is equivalent to self.frame.iloc[...] but
        returns a Grid instance instead of a GeoDataFrame.
        """
        return partial(self._iloc)

    def _loc(self, item) -> Self:
        frame = self.frame.loc[item]
        result = self.copy()
        result.frame = frame
        return result

    @property
    def loc(self):
        """
        Wrapper for self.frame.loc.
        self.loc[...] is equivalent to self.frame.loc[...] but
        returns a Grid instance instead of a GeoDataFrame.
        """
        return partial(self._loc)


    def copy(self) -> Self:
        result = copy.copy(self)
        return result

    def __delitem__(self, key):
        del self.frame[key]

    def __setitem__(self, key, value):
        self.frame[key] = value

    # def __getitem__(self, item) -> Union[
    #     pd.Series,
    #     pd.DataFrame,
    #     gpd.GeoDataFrame,
    #     gpd.GeoSeries,
    #     pd.Index,
    #     pd.MultiIndex,
    # ]:
    #     return self.frame[item]

    @classmethod
    def from_frame(
            cls,
            frame: pd.DataFrame,
            wrapper: Self,
    ) -> Self:
        result = cls()
        result.__dict__.update(wrapper.__dict__)
        result.frame = frame.copy()
        return result

    @classmethod
    def from_wrapper(
            cls,
            wrapper: Self,
            frame: pd.DataFrame = None,
    ) -> Self:
        result = cls()
        result.__dict__.update(wrapper.__dict__)
        if frame is not None:
            result.frame = frame.copy()
        else:
            result.frame = result.frame.copy()
        return result

    def to_copy(
            self,
            frame=None,
            **kwargs,
    ) -> Self:
        result = self.copy()
        if frame is not None:
            result.frame = frame.copy()
        else:
            result.frame = self.frame.copy()

        result.__dict__.update(**kwargs)
        return result

    def __len__(self):
        return len(self.frame)

    @property
    def crs(self):
        return self.frame.crs

    @property
    def columns(self):
        return self.frame.columns

    @property
    def geometry(self) -> gpd.GeoSeries:
        return self.frame.geometry

    def __set_name__(self, owner, name):
        self.__name__ = name

    def __copy__(self) -> Self:
        result = self.__class__()
        result.__dict__.update(self.__dict__)
        return result

    @Loc
    def loc(self):
        ...

    @Loc
    def iloc(self):
        ...






# from IPython import get_ipython
# ip = get_ipython()
# for mt in (
#     "application/vnd.dataframe+json",
#     "application/vnd.dataresource+json",  # older frontends
#     "text/html",  # fallback
# ):
#     fmt = ip.display_formatter.formatters[mt]
#     base = fmt.lookup_by_type(pd.DataFrame)
#     if base is not None:
#         # bind delegating formatter for your wrapper
#         fmt.for_type(FrameWrapper, lambda obj, _base=base: _base(obj.frame))
# import pandas as pd
# pd.options.display.html.table_schema = True
# # Optional if you want HTML fallback too:
# pd.options.display.notebook_repr_html = True