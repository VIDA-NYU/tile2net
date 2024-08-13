from __future__ import annotations

import itertools
import json
import os.path
import re
from collections import defaultdict
from functools import *

from tile2net.namespace import Namespace


class AttributeAccesses:
    regex = r'\bargs[a-zA-Z.]+[a-zA-Z]\b'

    # regex = r""
    # regex = r"(?<!#.*)\bargs[a-zA-Z.]+[a-zA-Z]\b"

    @classmethod
    def test_regex(cls):
        regex = cls.regex
        assert re.search(regex, 'args.hello')
        assert re.search(regex, 'args.hello.world()')
        assert re.search(regex, 'args.hello')
        assert re.search(regex, 'args.hello.world').group(0)
        assert re.search(regex, 'args.hello.world()').group(0)
        assert re.search(regex, '[args.hello.world]').group(0)
        assert not re.search(regex, 'kwargs.hello.world')
        assert not re.search(regex, 'bargs.hello')

    def __init__(
            self,
            top: str,
            name: str,
            namespace: type[Namespace] = Namespace,
            exclude: set[str] = frozenset()
    ):
        self.top = top
        self.name = name
        self.namespace = namespace
        self.exclude = exclude

    @cached_property
    def accesses(self) -> dict:
        top = self.top
        regex = self.regex
        accesses = defaultdict(lambda: defaultdict(set))
        for root, dirs, files in os.walk(top):
            for file in files:
                if not file.endswith('.py'):
                    continue
                path = os.path.join(root, file)
                if any(exclude in path for exclude in self.exclude):
                    continue
                with open(path) as f:
                    for n, line in enumerate(f, 1):
                        if line.strip().startswith('#'):
                            continue
                        matches = re.findall(regex, line)
                        if len(matches):
                            accesses[path][n].update(matches)

        return accesses

    @cached_property
    def misses(self) -> dict:
        exceptions = set(itertools.chain(
            dir(int),
            dir(float),
            dir(str),
            dir(bool),
        ))

        result = defaultdict(lambda: defaultdict(set))
        for path, line_matches in self.accesses.items():
            for line, matches in line_matches.items():
                namespace = self.namespace
                for match in matches:
                    obj = namespace
                    parts = match.split('.')[1:]
                    for part in parts:
                        try:
                            obj = getattr(obj, part)
                        except AttributeError:
                            if part in exceptions:
                                continue
                            result[path][line].add(match)
                            break
        return result


def test_namespaces():
    top = os.path.join(
        __file__,
        '..',
        '..',
        'src',
        'tile2net',
    )
    top = os.path.abspath(top)
    attrs = AttributeAccesses(
        top=top,
        name='args',
        exclude={'tile2net/src/tile2net/tileseg/tests/'}
    )
    assert not attrs.misses, json.dumps(attrs.misses, indent=4)


if __name__ == '__main__':
    test_namespaces()
