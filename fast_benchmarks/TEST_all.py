# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on 01.09.20.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

from fast_benchmarks.TEST_tools import test_
run = True
test_('radial_M_explicit_newfront', run=run)
test_('radial_M_explicit_oldfront', run=run)
test_('radial_M_implicit_newfront', run=run)
test_('radial_M_implicit_oldfront', run=run)
test_('radial_K_explicit_newfront', run=run)
test_('radial_K_explicit_oldfront', run=run)
test_('radial_K_implicit_newfront', run=run)
test_('radial_K_implicit_oldfront', run=run)