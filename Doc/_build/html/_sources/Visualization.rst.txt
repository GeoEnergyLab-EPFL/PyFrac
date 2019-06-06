

Post-processing and Visualization
=================================

A comprehensive set of post-processing and visualization routines are provided by PyFrac.

Plotting a Fracture
-------------------
Let us start by visualizing the initial fracture in the :ref:`run-a-simulation` section. To plot the fracture, we can use the :py:func:`Fracture.plot_fracture` function provided by the Fracture class.

.. code-block:: python

    Fr.plot_fracture()

With the default options, this function plots the mesh, the footprint and the fracture width with 3D projection. The plot is interactive and can be zoomed in using the mouse wheel.

.. image:: /images/default_fracture.png
    :align:   center
    :scale: 80 %

You can also provide the quantity you want to plot. The following quantities can be plotted:

.. csv-table:: supported variables
    :align:   center
    :header: "supported variables"

    'w' or 'width'
    'pf' or 'fluid pressure'
    'pn' or 'net pressure'
    'v' or 'front velocity'
    'Re' or 'Reynolds number'
    'ff' or 'fluid flux'
    'fv' or 'fluid velocity'
    'mesh'
    'footprint'
    'lk' or 'leak off'

.. note:: The variables 'Reynolds number', 'fluid flux' and 'fluid velocity' are not saved by default in the results. Their saving can be enabled using simulation properties. See :py:class:`Properties.SimulationProperties` for details.

For example, to plot fracture footprint in 2D projection, we can do the following:

.. code-block:: python

    Fig = Fr.plot_fracture(variable='mesh', projection='2D')
    Fig = Fr.plot_fracture(variable='footprint', fig=Fig, projection='2D')

The first instruction will plot mesh of the Fracture and will return a :py:class:`Figure` object. We can use the same figure to plot the footprint. In this case, it will be superimposed on the first plot. The variables can also be plotted as a colormap or contours. Let us plot the width of the our fracture in the form of a colormap. We can also superimpose contours on it.

.. code-block:: python

    Fig = Fr.plot_fracture(variable='width', projection='2D_clrmap')
    Fig = Fr.plot_fracture(variable='width', fig=Fig, projection='2D_contours')

Let us also superimpose fracture footprint to know where the fracture front is located. The color of the front line can be changed to distinguish it from the contour lines. This can be done by giving it customized plot properties.

.. code-block:: python

    from src.Properties import PlotProperties

    plot_properties = PlotProperties(line_color='tomato')
    Fig = Fr.plot_fracture(variable='footprint', fig=Fig, projection='2D', plot_prop=plot_properties)

.. image:: /images/width_contours.png
    :align:   center
    :scale: 80 %

The above example shows only some basic functionality. For a complete list of available options, see the documentation of the :py:func:`Fracture.plot_fracture` function.

Apart from plotting the whole fracture, you can also plot a slice of the fracture using the py:func:`Fracture.plot_fracture_slice` function. It plots a slice of the domain defined by two given points. let us plot a slice of our mesh passing from the two points (-7, -5) and (7, 5).

.. code-block:: python

    Fr.plot_fracture_slice(variable='width', point1=[-7, -5], point2=[7, 5])

By default, it will be plotted in 2D projection, but 3D projection can also be plotted.

.. image:: /images/fracture_slice.png
    :align:   center
    :scale: 80 %

If you want to have more control on your plots, you can use the underlying functions that are used by the :py:func:`Fracture.Fracture.plot_fracture` function. Worth mentioning among them are the :py:func:`Visualization.plot_fracture_variable_as_color_map`, :py:func:`Visualization.plot_fracture_variable_as_contours` and :py:func:`Visualization.plot_variable_vs_time`. To use these functions, you can load a fracture variable using :py:func:`PostProcessFracture.get_fracture_variable` function. See the documentation for more details.

Plotting Fracture Evolution
---------------------------
The fist step to visualize the fracture evolution is to load the fracture objects at different times from a stored simulation run. :py:func:`PostProcessFracture.load_fractures` function will do that for you. You can provide the times at which the state of fracture is to be loaded. The function will return a list of Fracture objects closest to the times given in the time series. It will also return the properties used in the simulation in the form of a tuple consisting of solid, fluid, injection and simulation properties in order. Note that for a time given in the time series, the fracture with the closest and larger time will be returned. Let us load the results from the simulation ran in the :ref:`run-a-simulation` section:

.. code-block:: python

    from src.PostProcessFracture import load_fractures

    Fr_list, properties = load_fractures(sim_name='radial')

Note that we have not provided any time series. In this case, all of the fractures will be loaded. Also, since we have not provided any disk address, the results will be loaded from the default folder. If multiple simulations with the same simulation name are found, the most recent run will be loaded. Now let us plot the evolution of the fracture radius of the loaded fractures. We can use the :py:func:`Visualization.plot_fracture_list` to do that.

.. code-block:: python

    from src.Properties import PlotProperties
    from src.Visualization import plot_fracture_list

    plot_prop = PlotProperties(line_style='.', graph_scaling='loglog')
    Fig_R = plot_fracture_list(Fr_list,
                               variable='d_mean',
                               plot_prop=plot_prop)

The above instructions will instantiate a :py:class:`Properties.PlotProperties` class object that can be used to specify the properties that are to be used to plot the given variable. The variable 'd_mean' here specifies the minimum distance of the front from the injection point. Below is the list of variables that can be plotted.

.. csv-table:: supported variables
    :align:   center
    :header: "supported variables"

    'front_dist_min' or 'd_min'
    'front_dist_max' or 'd_max'
    'front_dist_mean' or 'd_mean'
    'mesh'
    'footprint'
    'volume' or 'V'
    'lk' or 'leak off'
    'lkv' or 'leaked off volume'
    'ar' or 'aspect ratio'
    'efficiency' or 'ef'
    'w' or 'width'
    'p' or 'pressure'
    'v' or 'front velocity'
    'Re' or 'Reynolds number'
    'ff' or 'fluid flux'
    'fv' or 'fluid velocity'

.. note:: The variables 'Reynolds number', 'fluid flux' and 'fluid velocity' are not saved by default in the results. Their saving can be enabled using simulation properties. See :py:class:`Properties.SimulationProperties` for details.

PyFrac provides the capability to plot analytical solutions available in a number of limiting regimes. Let us compare the fracture radius we have to a fracture propagating in a toughness dominated regime.

.. code-block:: python

    from src.PostProcessFracture import get_fracture_variable
    from src.Visualization import plot_analytical_solution

    time_srs = get_fracture_variable(Fr_list, variable='time')
    Fig_R = plot_analytical_solution(regime='M',
                                     variable='d_mean',
                                     mat_prop=Solid,
                                     inj_prop=Injection,
                                     fluid_prop=Fluid,
                                     time_srs=time_srs,
                                     fig=Fig_R)

The function :py:func:`PostProcessFracture.get_fracture_variable` provides a list of the values of the required variable. Here, we have used it to get a list of times at which the solution is available in the fracture list. This list, along with the material, fluid and injection properties are then given to the :py:func:`PostProcessFracture.plot_analytical_solution` function to plot the analytical solution at the given times. Just like the case of a single fracture, the evolution of a fracture along a slice of the domain can also be plotted. Let us plot the fracture width evolution along a vertical slice passing through the injection point. Unlike the previous example where the solution was interpolated between the evaluated solution on the line joining the two given points, here we will plot the discrete values of the solution evaluated at the cell centers. This can be done by enabling the plot_cell_center argument. Below, we plot the width at cell centers along the vertical line passing through the center of the cell containing our point.

.. code-block:: python

    from src.Visualization import plot_fracture_list_slice
    import numpy as np

    time_srs = np.geomspace(2e-3, 1, 5)
    Fr_list, properties = load_fractures(sim_name="radial", time_srs=time_srs)

    # plot slice
    ext_pnts = np.empty((2, 2), dtype=np.float64)
    Fig_WS = plot_fracture_list_slice(Fr_list,
                                      variable='w',
                                      projection='2D',
                                      point1=[0., 0.],
                                      orientation='vertical',
                                      plot_cell_center=True,
                                      extreme_points=ext_pnts)

In the above code, we first load the state of the fracture at five equidistant times in geometric space. The fracture list is then passed to the :py:func:`Visualization.plot_fracture_list_slice` which plots the slice of the domain passing through the given point. To compare the solution, we can also plot slice of the analytical solution. We have passed an empty array to the slice plotting function which will be written by the extreme points on the mesh along the slice, which can be used to plot the analytical solution slice.

.. code-block:: python

    from src.Visualization import plot_analytical_solution_slice

    time_srs_loaded = get_fracture_variable(Fr_list, variable='time')
    Fig_WS = plot_analytical_solution_slice('M',
                                            'w',
                                            Solid,
                                            Injection,
                                            fluid_prop=Fluid,
                                            fig=Fig_WS,
                                            time_srs=time_srs_loaded,
                                            point1=ext_pnts[0],
                                            point2=ext_pnts[1])

Finally, in addition to the slice, solution at a single point can also be plotted using the :py:func:`Visualization.plot_fracture_list_at_point` function. See the documention of the functions for details.