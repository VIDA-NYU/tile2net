from __future__ import annotations
import types

import builtins
import functools
import inspect
from functools import *
from torch.fx.experimental.recording import trace_shape_events_log
from typing import *

from pandas.core.indexing import is_nested_tuple
from .nested import Nested
import argparse

if False:
    from .cfg import Cfg
    from ..grid import Grid

T = TypeVar(
    'T',
    bound=Callable[..., Any]
)

P = ParamSpec("P")  # parameters of the wrapped function
R = TypeVar("R")  # return type of the wrapped function


# ── helpers ────────────────────────────────────────────────────────────────────

def _is_dict_like(ann) -> bool:
    if ann is None:
        return False
    origin = get_origin(ann)
    if origin is dict:
        return True
    if origin is Union:
        return any(_is_dict_like(a) for a in get_args(ann))
    return False


def _is_dict_like(ann) -> bool:
    """True if *ann* is (or contains) a Dict-style annotation."""
    if ann is None:
        return False
    origin = get_origin(ann)
    if origin is dict:  # plain Dict[…]
        return True
    if origin in {Union, types.UnionType}:  # PEP484 / PEP604 unions
        return any(_is_dict_like(a) for a in get_args(ann))
    return False


def _parse_dict_or_float(tok: str):
    if ":" in tok:
        k, v = tok.split(":", 1)
        return k, float(v)
    return float(tok)


class _DictOrFloatAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        scalar: float | None = None
        mapping: dict[str, float] = {}
        for val in values:
            if isinstance(val, tuple):  # key:value
                k, v = val
                mapping[k] = v
            else:  # bare float
                scalar = val
        if scalar is not None and mapping:
            parser.error(f"{option_string} cannot mix scalar and key:value pairs")
        setattr(namespace, self.dest, scalar if scalar is not None else mapping)


# ── patched cmdline.property ───────────────────────────────────────────────────


class property(
    Nested
):
    """
    This class allows for properties to also generate metadata
    that can be used as a command line argument.
    """

    def _get(
            self: property,
            instance: Nested,
            owner: type[Nested]
    ):
        from .cfg import Cfg
        self.instance = instance
        self.owner = owner
        if issubclass(owner, Cfg):
            self._cfg = instance
            self._Cfg = owner
        else:
            self._cfg = instance._cfg
            self._Cfg = instance._Cfg

        cfg = self._cfg
        if (
                cfg is None
                or not cfg._active
        ):
            return self

        trace = self._trace
        stack = []
        if cfg is not cfg._default:
            stack.append(cfg)
        if cfg._context is not None:
            stack.append(cfg._context)
        if cfg._default is not None:
            stack.append(cfg._default)

        for _cfg in  stack :
            if trace in _cfg:
                return _cfg[trace]
        msg = f'No default value for {self._trace!r} in {cfg!r}'
        raise AttributeError(msg)

    locals().update(
        __get__=_get,
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
            self._cfg = instance
            self._Cfg = owner
        else:
            self._cfg = instance._cfg
            self._Cfg = instance._Cfg

        cfg = instance._cfg
        cfg[self._trace] = value

    def __delete__(
            self,
            instance: Nested,
    ):
        cfg = instance._cfg
        if self._trace in cfg:
            del cfg[self._trace]
        else:
            msg = f"{type(self).__name__!r} object has no attribute {self._trace!r}"
            raise AttributeError(msg)

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

    @cached_property
    def default(self) -> Any:
        # sig = inspect.signature(self.__wrapped__)
        # params = tuple(sig.parameters.values())
        # if not params:  # () -> T
        #     return self.__wrapped__(self.instance)  # type: ignore[misc]
        # if (
        #         len(params) == 1
        #         and self.instance is not None
        # ):
        #     return self.__wrapped__(self.instance)  # type: ignore[arg-type]
        return self.__wrapped__(self.instance)
        return None

    @cached_property
    def _dict_like(self) -> bool:
        return _is_dict_like(self._annotation)

    @cached_property
    def type(self):
        if self._dict_like:
            return _parse_dict_or_float
        ann = self._annotation
        return ann if ann in {int, float, str} else None

    @cached_property
    def action(self):
        if self._dict_like:
            return _DictOrFloatAction
        ann = self._annotation
        if ann is bool:
            return "store_false" if bool(self.default) else "store_true"
        if ann in {list, set, tuple}:
            return "append"
        return None

    @cached_property
    def nargs(self):
        if self._dict_like or self._annotation in {list, set, tuple}:
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
        return self._trace
        # return self._trace.replace(".", "_")

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
        kw = {
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

    # def _trace_key(self, key: str) -> str:
    #     key = key.lower() if key.isupper() else key
    #     return f"{self._trace}.{key}" if self._trace else key
    #
    # def __getitem__(self, key: str):
    #     trace = self._trace_key(key)
    #     return self._cfg[trace]
    #
    # def __setitem__(self, key: str, value):
    #     trace = self._trace_key(key)
    #     self._cfg[trace] = value
    #
    # def __delitem__(self, key: str):
    #     trace = self._trace_key(key)
    #     if trace in self._cfg:
    #         del self._cfg[trace]
    #     else:
    #         raise KeyError(trace)

    # def _trace_key(self, key: str) -> str:
    #     key = key.lower() if key.isupper() else key
    #     return f"{self._trace}.{key}" if self._trace else key
    #
    # def _navigate(self, dotted: str):
    #     obj: Any = self
    #     for part in dotted.split("."):
    #         part = part.lower() if part.isupper() else part
    #         obj = getattr(obj, part)
    #     return obj
    #
    # def __getitem__(self, key: str):
    #     trace = self._trace_key(key)
    #     try:
    #         return self._cfg[trace]
    #     except KeyError:
    #         if "." in key:
    #             return self._navigate(key)
    #         raise
    #
    # def __setitem__(self, key: str, value):
    #     if "." in key:
    #         obj = self._navigate(".".join(key.split(".")[:-1]))
    #         leaf = key.split(".")[-1]
    #         setattr(obj, leaf.lower() if leaf.isupper() else leaf, value)
    #     else:
    #         self._cfg[self._trace_key(key)] = value
    #
    # def __delitem__(self, key: str):
    #     if "." in key:
    #         obj = self._navigate(".".join(key.split(".")[:-1]))
    #         leaf = key.split(".")[-1]
    #         delattr(obj, leaf.lower() if leaf.isupper() else leaf)
    #     else:
    #         trace = self._trace_key(key)
    #         if trace in self._cfg:
    #             del self._cfg[trace]
    #         else:
    #             raise KeyError(trace)

    # --------------------------- helpers ---------------------------------
    def _trace_key(self, key: str) -> str:
        key = key.lower() if key.isupper() else key
        return f"{self._trace}.{key}" if self._trace else key

    def _navigate(self, dotted: str):
        obj: Any = self
        for part in dotted.split("."):
            part = part.lower() if part.isupper() else part
            obj = getattr(obj, part)
        return obj

    # --------------------- mapping interface -----------------------------
    def __getitem__(self, key: str):
        trace = self._trace_key(key)
        try:
            return self._cfg[trace]
        except KeyError:
            try:
                return self._navigate(key)
            except AttributeError:
                raise KeyError(trace) from None

    def __setitem__(self, key: str, value):
        if "." in key:
            parent_path, leaf = key.rsplit(".", 1)
            parent = self._navigate(parent_path)
            setattr(parent, leaf.lower() if leaf.isupper() else leaf, value)
        else:
            self._cfg[self._trace_key(key)] = value

    def __delitem__(self, key: str):
        if "." in key:
            parent_path, leaf = key.rsplit(".", 1)
            parent = self._navigate(parent_path)
            delattr(parent, leaf.lower() if leaf.isupper() else leaf)
        else:
            trace = self._trace_key(key)
            if trace in self._cfg:
                del self._cfg[trace]
            else:
                raise KeyError(trace)
