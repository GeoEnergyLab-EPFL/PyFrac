

Post-processing and Visualization
=================================

A comprehensive set of post-processing and visualization routines are provided by **PyFrac**.

Plotting a Fracture
-------------------
Let us first initialize a fracture in the toughness dominated regime at 100 seconds after the injection started. We are using a mesh with :math:`41\times41` elements, descretizing a domain of :math:`10 \times 10` meters. The material, fluid and injection properties are the same as in the :ref:`run-a-simulation` section.

.. code-block:: python

    from src.Fracture import Fracture

   # initialization parameters
   init_time = 100
   init_param = ("K", "time", init_time)

   # creating fracture object
   Fr = Fracture(Mesh,
                 init_param,
                 Solid,
                 Fluid,
                 Injection,
                 simulProp)

To plot the fracture, we can use the :py:func:`Fracture.plot_fracture` function provided by the Fracture class.



.. code-block:: python

    Fr.plot_fracture()

With the default options, this function plots the mesh, the footprint and the fracture width with 3D projection. The 3D plot is interactive and can be zoomed in using the mouse wheel.

.. image:: /images/default_fracture.png
    :align:   center
    :scale: 80 %

You can also provide the quantity you want to plot. The following quantities can be plotted:

.. csv-table:: supported variables
    :align:   center
    :header: "supported variables"

    'w' or 'width'
    'p' or 'pressure'
    'v' or 'front velocity'
    'Re' or 'Reynolds number'
    'ff' or 'fluid flux'
    'fv' or 'fluid velocity'
    'mesh'
    'footprint'
    'lk' or 'leaked off'

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
The fist step to visualize the fracture evolution is to load the fracture objects at different times from a stored simulation run. :py:func:`PostProcessFracture.load_fractures` function will do that for you. You can provide the time series at which the fractures are to be loaded. The function will provide a list of Fracture objects closest to the times given in the time series. It will also return the properties used in the simulation in the form of a tuple consisting of solid, fluid, injection and simulation properties in order. Note that for a time given in the time series, the fracture with the closest and larger time will be returned. Lets load the results from the simulation ran in the :ref:`run-a-simulation` section:

.. code-block:: python

    From src.PostProcessFracture import load_fractures

    Fr_list, properties = load_fractures(sim_name='radial')

Note that we have not provided any time series. In this case, all of the fractures will be loaded. Also, since we have not provided any folder address, the results will be loaded from the default folder. If multiple simulations with the same simulation name are found, the most recent run will be loaded. Now lets plot the evolution of the fracture radius of the loaded fractures. We can use the :py:func:`Visualization.plot_fracture_list` to do that.

.. code-block:: python

    plot_prop = PlotProperties(line_style='.', graph_scaling='loglog')
    Fig_R = plot_fracture_list(Fr_list,
                               variable='d_mean',
                               plot_prop=plot_prop)

