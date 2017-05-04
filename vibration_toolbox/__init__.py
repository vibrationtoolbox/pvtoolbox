"""
The Vibration Toolbox, Python Edition

Joseph C. Slater and Raphael Timbó
"""


"""
With this, `import vibration_toolbox as vtb` will import
all toolbox functions in the name space

`import vibration_toolbox as vtb` will keep them tucked behind `vtb`

`import vibration_toolbox.sdof as sdof` will tuck the sdof functions in the
 `sdof` name space.
"""

__title__ = 'vibration_toolbox'
__version__ = '0.5b11'
__author__ = u'Joseph C. Slater and Raphael Timbó'
__license__ = 'MIT'
__copyright__ = 'Copyright 1991-2017 Joseph C. Slater'
__all__ = ['sdof', 'mdof', 'ema', 'vibesystem', 'continuous_systems',
           '__version__']

import os
import sys
import matplotlib as mpl

if str(sys.argv[0])[-6:] == 'pytest':
    print('setting backend to agg')
    mpl.use('agg')

from .sdof import *
from .mdof import *
from .ema import *
from .vibesystem import *
from .continuous_systems import *

# print options were change inside modules to produce better
# outputs at examples. Here we set the print options to the
# default values after importing the modules to avoid changing
# np default print options when importing the toolbox.
np.set_printoptions(edgeitems=3, infstr='inf', linewidth=75,
                    nanstr='nan', precision=8, suppress=False,
                    threshold=1000, formatter=None)
