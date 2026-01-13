

class ExtensionNotFoundError(ValueError):
    def __init__(
            self,
            path: str,
    ):
        super().__init__(f'No extension found in {path!r}')


class XYNotFoundError(ValueError):
    def __init__(
            self,
            path: str,
            missing: set[str] | list[str] | tuple[str, ...],
    ):
        missing_fmt = ', '.join(missing)
        super().__init__(
            f'indir failed to parse {path!r}; missing required characters: {missing_fmt}'
        )
