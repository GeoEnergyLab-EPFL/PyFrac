# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on 01.09.20.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

import pytest
import numpy as np

from continuous_front_reconstruction import ray_tracing_numpy
from continuous_front_reconstruction import find_indexes_repeatd_elements
from continuous_front_reconstruction import Point
from continuous_front_reconstruction import distance
from continuous_front_reconstruction import copute_area_of_a_polygon
from continuous_front_reconstruction import pointtolinedistance


def test_ray_tracing_numpy():
    x=0.
    y=0.
    poly=np.asarray([[-1,-1],[1,-1],[0,1]])
    answer = ray_tracing_numpy(x, y, poly)
    assert answer[0] == True
    x=np.asarray([[0.,5.]])
    y=np.asarray([[0.,5.]])
    answer = ray_tracing_numpy(x, y, poly)
    assert answer[1] == False

def test_find_indexes_repeatd_elements():
    list = [10, 15, 33, 33, 18, 22, 16, 22]
    indexes = find_indexes_repeatd_elements(list)
    assert indexes[0] == 5
    assert indexes[1] == 7
    assert indexes[2] == 2
    assert indexes[3] == 3

def test_distance():
    p1=Point('p1',0,0)
    p2=Point('p2',3,4)
    assert distance(p1,p2) == 5

def test_copute_area_of_a_closed_front():
    assert copute_area_of_a_polygon(np.asarray([0,1,0]),np.asarray([0,0,1])) == 0.5
    assert copute_area_of_a_polygon(np.asarray([1,38,9]),np.asarray([2,2,20])) == 333
    assert copute_area_of_a_polygon(np.asarray([0,2,3,0.5]),np.asarray([0,0,3,3])) == 6.75

def test_pointtolinedistance():
    assert pointtolinedistance(3.,1.,2.,-2.,-13./3.,-3.) == pytest.approx(1./5.,0.00001)
