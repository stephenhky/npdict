
from typing import Tuple, Generator
import sys
from itertools import product

import numpy as np
from nptyping import NDArray, Shape, Float

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


class NumpyNDArrayWrappedDict(type):
    def __new__(cls, *args, **kwargs):
        pass

