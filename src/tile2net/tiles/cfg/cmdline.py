from __future__ import annotations

import builtins
import functools
import inspect
from functools import *
from typing import *

if False:
    from .cfg import Cfg
    from ..tiles import Tiles

T = TypeVar(
    'T',
    bound=Callable[..., Any]
)


def __get__(
        self: property,
        instance: Union[
            Cfg,
            Namespace
        ],
        owner: Union[
            type[Cfg],
            type[Namespace]
        ]
):
    """"""
    self.instance = instance
    self.owner = owner
    return self


class Nested(

):
    instance: Union[Cfg, Namespace] = None
    owner: Union[Type[Cfg], Type[Namespace]] = None
    _nested: dict[str, Nested]

    locals().update(
        __get__=__get__
    )

    @builtins.property
    def tiles(self) -> Tiles:
        instance = self.instance
        while not isinstance(instance, Cfg):
            instance = instance.instance
        return instance.instance


    def __init__(
            self,
            *args,
            **kwargs,
    ):
        ...

    @cached_property
    def _trace(self):
        from .cfg import Cfg
        if issubclass(self.owner, Cfg):
            return self.__name__
        elif isinstance(self.instance, Nested):
            return f'{self.instance._trace}.{self.__name__}'
        else:
            msg = (
                f'Cannot determine trace for {self.__name__} in '
                f'{self.owner} with {self.instance=}'
            )
            raise ValueError(msg)

    def __set_name__(self, owner: type[Nested], name):
        self.__name__ = name
        self.owner = owner
        if issubclass(owner, Nested):
            if '_nested' not in owner.__dict__:
                owner._nested = {}
            owner._nested[name] = self


class property(
    Nested
):
    """
    This class allows for properties to also generate metadata
    that can be used as a command line argument.
    """
    __wrapped__: Callable[..., T] = None

    @classmethod
    def with_options(cls: Self, short: str | None = None) -> Callable[..., Type[Self]]:
        def wrapper(func: Callable[..., T]) -> "property":
            inst = cls(func)
            if short is not None:
                inst.short = short
            return inst              # <<< missing
        return wrapper

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
        """Best-effort evaluation of the wrapped function to obtain a default."""
        try:
            sig = inspect.signature(self.__wrapped__)
            if not sig.parameters:
                return self.__wrapped__(self)  # type: ignore[arg-type]
        except Exception:
            pass
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

property.__set_name__