"""Excitation signals as used in structural dynamics and vibration fatigue.

pyExSi is a port of sdypy-excitation (https://github.com/sdypy/sdypy-excitation),
where the code is maintained. This package re-exports sdypy.excitation.signals
under the pyExSi name.
"""
__version__ = '0.43.4'

import sys as _sys

from sdypy.excitation import signals
from sdypy.excitation.signals import *

# Register the submodule under the pyExSi name, so that `import pyExSi.signals`
# returns sdypy.excitation.signals instead of looking for a file in this package.
_sys.modules[f"{__name__}.signals"] = signals
