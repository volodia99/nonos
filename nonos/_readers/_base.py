__all__ = [
    "ReaderMixin",
]


class ReaderMixin:
    __slots__: list[str] = []

    def __init__(self, *args, **kwargs):  # noqa: ARG002
        raise TypeError(f"{self.__class__} is not instantiable")
