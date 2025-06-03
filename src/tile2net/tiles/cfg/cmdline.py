from __future__ import annotations

import builtins
import functools
import inspect
from functools import *
from torch.fx.experimental.recording import trace_shape_events_log
from typing import *

from pandas.core.indexing import is_nested_tuple
from .nested import Nested

if False:
    from .cfg import Cfg
    from ..tiles import Tiles

T = TypeVar(
    'T',
    bound=Callable[..., Any]
)

P = ParamSpec("P")  # parameters of the wrapped function
R = TypeVar("R")  # return type of the wrapped function


# todo: if Config is just the class, we should return self

def __get__(
        self: property,
        instance: Nested,
        owner: type[Nested]
):
    from .cfg import Cfg
    self.instance = instance
    self.owner = owner
    if issubclass(owner, Cfg):
        self.cfg = instance
        self.Cfg = owner
    else:
        self.cfg = instance.cfg
        self.Cfg = instance.Cfg

    cfg = self.cfg
    if (
            cfg is None
            or not cfg._active
    ):
        return self
    trace = self._trace
    if trace in cfg:
        return cfg[trace]
    cfg[trace] = self.default
    result = cfg[trace]
    return result


class property(
    Nested
):
    """
    This class allows for properties to also generate metadata
    that can be used as a command line argument.
    """
    locals().update(
        __get__=__get__,
    )

    def __set__(
            self,
            instance: Nested,
            value,
    ):
        from .cfg import Cfg
        self.instance = instance
        self.owner = owner = type(instance)
        if issubclass(owner, Cfg):
            self.cfg = instance
            self.Cfg = owner
        else:
            self.cfg = instance.cfg
            self.Cfg = instance.Cfg

        return self
        cfg = instance.cfg
        cfg[self._trace] = value

    def __delete__(
            self,
            instance: Nested,
    ):
        cfg = instance.cfg
        if self._trace in cfg:
            del cfg[self._trace]
        else:
            msg = f"{type(self).__name__!r} object has no attribute {self._trace!r}"
            raise AttributeError(msg)

    # @classmethod
    # def with_options(
    #         cls,
    #         short: str | None = None,
    #         long: str | None = None,
    # ) -> Type[property]:
    #     def wrapper(func: Callable[..., T]) -> "property":
    #         inst = cls(func)
    #         if short:
    #             inst.short = short
    #         if long:
    #             inst.long = long
    #         return inst  # <<< missing
    #
    #     return wrapper

    def add_options(
            self,
            short: str | None = None,
            long: str | None = None,
    ):
        if short:
            self.short = short
        if long:
            self.long = long
        return self

    def __init__(
            self,
            func: Callable[..., T],
            *args,
            **kwargs
    ):
        functools.update_wrapper(self, func)
        self.__doc__ = func.__doc__
        super().__init__(*args, **kwargs)

    @cached_property
    def _annotation(self) -> Optional[Type[Any]]:
        """Return annotation of wrapped function (if any)."""
        return get_type_hints(self.__wrapped__).get("return")

    # @cached_property
    # def default(self) -> Any:
    #     """Best-effort evaluation of the wrapped function to obtain a default."""
    #     try:
    #         sig = inspect.signature(self.__wrapped__)
    #         if not sig.parameters:
    #             return self.__wrapped__(self)  # type: ignore[arg-type]
    #     except Exception:
    #         pass
    #     return None
    #

    @cached_property
    def default(self) -> Any:
        # try:
        #     sig = inspect.signature(self.__wrapped__)
        #     params = tuple(sig.parameters.values())
        #     if not params:  # () -> T
        #         return self.__wrapped__()                    # type: ignore[misc]
        #     if (
        #         len(params) == 1
        #         and self.instance is not None
        #     ):
        #         return self.__wrapped__(self.instance)       # type: ignore[arg-type]
        # except Exception:
        #     pass
        sig = inspect.signature(self.__wrapped__)
        params = tuple(sig.parameters.values())
        if not params:  # () -> T
            return self.__wrapped__()  # type: ignore[misc]
        if (
                len(params) == 1
                and self.instance is not None
        ):
            return self.__wrapped__(self.instance)  # type: ignore[arg-type]
        return None

    @cached_property
    def type(self):
        ann = self._annotation
        return ann if ann in {int, float, str} else None

    @cached_property
    def action(self) -> Optional[str]:
        ann = self._annotation
        if ann is bool:
            return "store_false" if bool(self.default) else "store_true"
        if ann in {list, set, tuple}:
            return "append"
        return None

    @cached_property
    def nargs(self) -> Optional[str]:
        if self._annotation in {list, set, tuple}:
            return "+"
        return None

    @cached_property
    def short(self) -> Optional[str]:
        return None

    @cached_property
    def long(self) -> str:
        return f"--{self._trace}"

    @cached_property
    def dest(self) -> str:
        return self._trace.replace(".", "_")

    @cached_property
    def help(self) -> str:
        return self.__doc__ or ""

    @cached_property
    def posargs(self) -> List[str]:
        opts: List[str] = []
        if self.short:
            opts.append(self.short)
        opts.append(self.long)
        return opts

    @cached_property
    def kwargs(self) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "dest": self.dest,
            "help": self.help,
        }
        if self.action:
            kw["action"] = self.action
        if self.type:
            kw["type"] = self.type
        if self.nargs:
            kw["nargs"] = self.nargs
        if self.default is not None:
            kw["default"] = self.default
        return kw

    def __repr__(self):
        opts = ", ".join(self.posargs)
        meta = []
        if self.action:
            meta.append(f"action={self.action}")
        if self.type:
            meta.append(f"type={self.type.__name__}")
        if self.nargs:
            meta.append(f"nargs={self.nargs}")
        if self.default is not None:
            meta.append(f"default={self.default!r}")
        joined = ", ".join(meta)
        return f"<{opts}{', ' + joined if joined else ''}>"


class Namespace(
    Nested
):
    """
    This class allows for cmdline properties to be nested within
    namespaces, similar to how argparse works with subparsers.
    """
