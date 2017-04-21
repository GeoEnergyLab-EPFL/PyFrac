#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 11:18:48 2016

@author: zia
"""

from src.PostProcess import *


fig_wdth, fig_radius, fig_pressure = plot_data(".\Data\LamCourse", loglog = False, plot_w_prfl=True, plot_p_prfl=True)
fig_wdth, fig_radius, fig_pressure = plot_data(".\Data\LamFine", fig_w_cntr=fig_wdth, fig_r=fig_radius, loglog=False )
fig_wdth, fig_radius, fig_pressure = plot_data(".\Data\TurbLam", fig_w_cntr=fig_wdth, fig_r=fig_radius, loglog=False,
                                               plot_w_prfl=True, plot_p_prfl=True)
plt.show()

