# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Andreas Möri on Oct 9 13:36:21 2019.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# importing module
import os
import matplotlib
import sys

# Creates a new file
with open('batch_run.txt', 'w') as fp:
    pass

import radial_viscosity_explicit

os.remove('batch_run.txt')

matplotlib.pyplot.close('all')

sys.exit(0)