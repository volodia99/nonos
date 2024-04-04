"""
This module contains low-level tests for the internal details of
nonos._reader and nonos.loaders.
They are meant to ensure that the constraints ingrained
in the original design continue to hold in the future, and provide
immediate feedback if a refactor breaks any of the initial promises,
in areas that cannot be easily checked by a type checker.

These tests are *not* sacred, and may need to be adjusted in case of
a new change in design.
"""

import inspect
import sys
from enum import Enum
from types import ModuleType
from typing import List, Protocol, Type

import pytest

import nonos._readers as readers
from nonos import _types
from nonos._readers._base import ReaderMixin


def get_classes_from(module: ModuleType) -> List[Type]:
    retv: List[Type] = []
    for objname in module.__all__:
        obj = module.__dict__[objname]
        if inspect.isclass(obj):
            if obj.__class__ is type:
                continue
            if issubclass(obj, (Protocol, Enum)):  # type: ignore [arg-type]
                continue
            retv.append(obj)
    return retv


_reader_classes: List[Type] = []
for module in [
    readers._base,
    readers.ini,
    readers.planet,
    readers.binary,
]:
    _reader_classes.extend(get_classes_from(module))

_all_classes: List[Type] = _reader_classes.copy()
_all_classes.extend(get_classes_from(_types))


@pytest.fixture(
    params=_reader_classes, ids=lambda cls: f"{cls.__module__}.{cls.__name__}"
)
def reader_class(request):
    return request.param


@pytest.fixture(params=_all_classes, ids=lambda cls: f"{cls.__module__}.{cls.__name__}")
def interface_class(request):
    return request.param


def test_use_reader_mixin(reader_class):
    assert issubclass(reader_class, ReaderMixin)


def test_cannot_instanciate(reader_class):
    cls = reader_class
    with pytest.raises(TypeError, match=rf"^{cls} is not instantiable$"):
        cls()


def test_have_slots(interface_class):
    cls = interface_class
    assert hasattr(cls, "__slots__")
    assert isinstance(cls.__slots__, list)
    assert all(isinstance(k, str) for k in cls.__slots__)


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="runtime inspection of final classes requires Python 3.11 or newer",
)
def test_abstract_final_pattern(interface_class):
    # check that all interface classes are exactly one of
    # - abstract (ABC)
    # - @final
    # - Protocol
    # - *Mixin

    cls = interface_class
    isabstract = inspect.isabstract(cls)
    isfinal = getattr(cls, "__final__", False)
    isprotocol = issubclass(cls, Protocol)
    ismixin = cls.__name__.endswith("Mixin")
    assert isabstract ^ isfinal ^ isprotocol ^ ismixin
