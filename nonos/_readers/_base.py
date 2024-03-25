__all__ = [
    "ReaderMixin",
]
from typing import List


class ReaderMixin:
    __slots__: List[str] = []

    def __init__(self, *args, **kwargs):  # noqa: ARG002
        raise TypeError(f"{self.__class__} is not instantiable")
