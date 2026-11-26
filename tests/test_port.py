"""Check that pyExSi exposes exactly the sdypy-excitation code."""
import pytest

import sdypy.excitation
import sdypy.excitation.signals
import pyExSi

NAMES = [n for n in dir(sdypy.excitation.signals) if not n.startswith('_')]


@pytest.mark.parametrize("name", NAMES)
def test_names(name):
    assert getattr(pyExSi, name) is getattr(sdypy.excitation.signals, name)


@pytest.mark.parametrize("name", sdypy.excitation.__all__)
def test_public_names(name):
    assert getattr(pyExSi, name) is getattr(sdypy.excitation, name)


def test_signals_module():
    import pyExSi.signals
    from pyExSi.signals import random_gaussian

    assert pyExSi.signals is sdypy.excitation.signals
    assert random_gaussian is sdypy.excitation.random_gaussian
