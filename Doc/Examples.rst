.. PyFrac documentation master file, created by
   sphinx-quickstart on Mon Jun  4 15:58:10 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Examples
========

Penny shaped hydraulic fracture benchmark
-----------------------------------------
We first demonstrate the accuracy of PyFrac on the case of a penny-shaped hydraulic fracture propagating in a uniform permeable medium. The fracture initially starts propagating in the viscosity dominated regime and gradually transitions to toughness and finally to leak-off dominated regime. For this case, we have a semi-analytical solution available (see `[here] <https://books.google.ch/books/about/Fluid_driven_Penny_shaped_Fracture_in_El.html?id=a8wOtwAACAAJ&redir_esc=y>`_). Here, we will perform a simulation with the following parameters:

.. csv-table::
    :align:   center
    :header: "Paramters", "Values"

    :math:`E^\prime` (Plain strain modulus), :math:`39\textrm{GPa}`
    :math:`K_{Ic}` (fracture toughness), :math:`0.156~MPa\sqrt{\textrm{m}}`
    :math:`C_L` (Carter's leak-off coefficient), :math:`0.5\times10^{-6}~m/\sqrt{\textrm{s}}`
    :math:`\mu` (viscosity), :math:`8.3\times10^{-5}~Pa\cdot s`
    :math:`Q` (injection rate), :math:`0.01\textrm{m}^{3}/\textrm{s}`

Lets start the simulation at 0.5 seconds. At this time, the fracture has a radius of about 2 meters. We will make a mesh on a square domain of [-5, 5, -5, 5] meters with 41 cells in both :math:`x` and :math:`y` directions.

.. code-block:: python

   from src.Fracture import *
   from src.Controller import *

   # creating mesh
   Mesh = CartesianMesh(5, 5, 41, 41)

Next we setup the properties of the material by instantiating a :py:class:`Properties.MaterialProperties` object.

.. code-block:: python

   # solid properties
   nu = 0.4                            # Poisson's ratio
   youngs_mod = 3.3e10                 # Young's modulus
   Eprime = youngs_mod / (1 - nu**2)   # plain strain modulus
   K1c = 5e5 / (32 / math.pi)**0.5     # K' = 5e5
   Cl = 0.5e-6                         # Carter's leak off coefficient

   # material properties
   Solid = MaterialProperties(Mesh,
                              Eprime,
                              K1c,
                              Carters_coef=Cl)

After setting up the material properties, we next set up the properties of the fluid and its injection parameters by Instantiating the :py:class:`Properties.FluidProperties` and  :py:class:`Properties.InjectionProperties` classes. Also, to set the end time and the output folder, we will instantiate the :py:class:`Properties.SimulationProperties` object. Since we do not have fine scale heterogeneities present in the material, We will use the explicit front advancing algorithm here (see `[here] <https://onlinelibrary.wiley.com/doi/full/10.1002/nag.2898>`_ for more on that). To avoid excessive saving of the data, we will save only every third time step.

.. code-block:: python

   # injection parameters
   Q0 = 0.01  # injection rate
   Injection = InjectionProperties(Q0, Mesh)

   # fluid properties
   viscosity = 0.001 / 12  # mu' =0.001
   Fluid = FluidProperties(viscosity=viscosity)

   # simulation properties
   simulProp = SimulationProperties()
   simulProp.finalTime = 1e7                 # the time at which the simulation stops
   simulProp.set_outputFolder("./Data/MtoK_leakoff") # the disk address where the files are saved
   simulProp.outputEveryTS = 3               # the time after the output is generated (saving or plotting)
   simulProp.frontAdvancing = 'explicit'     # use explicit front advancing algorithm

We will start our simulation at 0.5 seconds after the start of injection. At this time, the fracture is propagating in viscosity dominated regime and we can initialize it with the viscosity dominated analytical solution. To do that, we will first instantiate the :py:class:`FractureInitialization.InitializationParameters` object and pass it to the constructor of :py:class:`Fracture.Fracture` class. We will also setup the Controller with a :py:class:`Controller.Controller` object and run the simulation.

.. code-block:: python

   # initializing fracture
   Fr_geometry = Geometry(shape='radial')
   init_param = InitializationParameters(Fr_geometry, regime='M', time=0.5)

   # creating fracture object
   Fr = Fracture(Mesh,
                 init_param,
                 Solid,
                 Fluid,
                 Injection,
                 simulProp)

   # create a Controller
   controller = Controller(Fr,
                           Solid,
                           Fluid,
                           Injection,
                           simulProp)

   # run the simulation
   controller.run()

Once the simulation is finished, or even when it is running, we can start visualizing the results. To do that, we first load the state of the fracture in the form of a list of :py:class:`Fracture.Fracture` objects.

.. code-block:: python

   Fr_list, properties = load_fractures("./Data/MtoK_leakoff")

To plot the evolution of radius of the fracture, we will use the :py:func:`Visualization.plot_fracture_list` function to plot the 'd_mean' variable. We will plot it in loglog scaling for better visualization. To do that, we will pass a :py:class:`Properties.PlotProperties` object with the graph_scaling attribute set to 'loglog'. The setting up of plot properties is, of course, optional.

.. code-block:: python

   # plotting efficiency
   plot_prop = PlotProperties(graph_scaling='loglog',
                              line_style='.')
   Fig_eff = plot_fracture_list(Fr_list,
                              variable='efficiency',
                              plot_prop=plot_prop)
To compare the solution with the semi-analytical solution, We have precomputed the solution using a matlab `[code] <https://datadryad.org/resource/doi:10.5061/dryad.gh469/1>`_ and directly inserted in as numpy array.

.. code-block:: python

   t = np.geomspace(0.5, 1e7, num=30)
   # solution taken from matlab code provided by Dontsov EV (2016)
   eff_analytical = np.asarray([0.9923, 0.9904, 0.9880, 0.9850, 0.9812, 0.9765, 0.9708, 0.9636, 0.9547, 0.9438, 0.9305,
                                0.9142, 0.8944, 0.8706, 0.8423, 0.8089, 0.7700, 0.7256, 0.6757, 0.6209, 0.5622, 0.5011,
                                0.4393, 0.3789, 0.3215, 0.2688, 0.2218, 0.1809, 0.1461, 0.1171])
   ax_eff = Fig_eff.get_axes()[0]
   ax_eff.semilogx(t, eff_analytical, 'r-', label='semi-analytical fracturing efficiency')
   ax_eff.legend()

Fracture radius is plotted in the same way

.. code-block:: python

   Fig_r = plot_fracture_list(Fr_list,
                              variable='d_mean',
                              plot_prop=plot_prop)

   # solution taken from matlab code provided by Dontsov EV (2016)
   r_analytical = np.asarray([0.0035, 0.0046, 0.0059, 0.0076, 0.0099, 0.0128, 0.0165, 0.0212, 0.0274, 0.0352, 0.0453,
                              0.0581, 0.0744, 0.0951, 0.1212, 0.1539, 0.1948, 0.2454, 0.3075, 0.3831, 0.4742, 0.5829,
                              0.7114, 0.8620, 1.0370, 1.2395, 1.4726, 1.7406, 2.0483, 2.4016])*1e3
   ax_r = Fig_r.get_axes()[0]
   ax_r.loglog(t, r_analytical, 'r-', label='semi-anlytical radius')
   ax_r.legend()


Height contained hydraulic fracture
-----------------------------------
This example simulates a hydraulic fracture propagating in a layer bounded with high stress layers from top and bottom, causing its height to be restricted to the height of the middle layer. The top and bottom layers have a confining stress of :math:`7.5\textrm{Mpa}`, while the middle layer has a confining stress of :math:`1\textrm{MPa}`. The fracture initially propagates as a radial fracture in the middle layer until it hits the high stress layers on the top and bottom. From then onwards, it propagates with the fixed height of the middle layer. The parameters used in the simulation are as follows:

.. csv-table::
    :align:   center
    :header: "Paramters", "Values"

    Plain strain modulus, :math:`39.2\textrm{GPa}`
    fracture toughness, :math:`0`
    viscosity, :math:`1.1\times10^{-3}\textrm{Pa.s}`
    injection rate, :math:`0.001\textrm{m}^{3}/\textrm{s}`
    confinning stress top & bottom layers, :math:`7.5\textrm{MPa}`
    confinning stress middle layer, :math:`1\textrm{MPa}`


Let us start by defining mesh. We are given the height of the middle layer to be 7 meters. Since we also want to simulate the early time of the propagation, when the fracture is radial, we start with a rectangular domain with dimensions of [-20, 20, -2.3, 2.3] meters. As the fracture will grow and reach the end of the domain along vertical axis, a re-meshing will be done to double the size of the domain to [-40, 40, -4.6, 4.6]. Since we want the simulation to take small time to finish, we discretize the domain relatively coarsely with 71 cells in the :math:`x` direction and 15 cells in the :math:`y` direction. This will result in slightly less accurate results. Of course, running the simulation with higher resolution will increase the accuracy of the solution.

.. code-block:: python

   from src.Fracture import *
   from src.Controller import *

   # creating mesh
   Mesh = CartesianMesh(20, 2.3, 71, 15)

Next we setup the properties of the material by instantiating a :py:class:`Properties.MaterialProperties` object. The material has uniform properties apart from the spatially varying confining stress, which is higher in the top and bottom layers. There are two possibilities to set spatially varying variables. We can either provide an array with the size of the mesh, giving them in each of the cell of the mesh. This will be problematic in case of re-meshing as the coordinates of the cells change when re-meshing is done. The second possibility is to provide a function giving the variable for the given set of coordinates. This function is evaluated on each re-meshing to get the variable on each cell of the new mesh. For this simulation, we set the spatially varying confining stress by providing the confining_stress_func argument while instantiating the :py:class:`Properties.MaterialProperties` object.

.. code-block:: python

   # solid properties
   nu = 0.4                            # Poisson's ratio
   youngs_mod = 3.3e10                 # Young's modulus
   Eprime = youngs_mod / (1 - nu ** 2) # plain strain modulus
   K_Ic = 0                            # fracture toughness of the material

   def sigmaO_func(x, y):
       """ The function providing the confining stress"""
       if abs(y) > 3:
           return 7.5e6
       else:
           return 1e6

   Solid = MaterialProperties(Mesh,
                              Eprime,
                              K_Ic,
                              confining_stress_func=sigmaO_func)

After setting up the material properties, we next set up the properties of the fluid and its injection parameters by Instantiating the :py:class:`Properties.FluidProperties` and  :py:class:`Properties.InjectionProperties` classes. Also, to set the end time and the output folder, we will instantiate the :py:class:`Properties.SimulationProperties` object.

.. code-block:: python

   # fluid properties
   Fluid = FluidProperties(viscosity=1.1e-3)

   # injection parameters
   Q0 = 0.001  # injection rate
   Injection = InjectionProperties(Q0, Mesh)

   # simulation properties
   simulProp = SimulationProperties()
   simulProp.finalTime = 145.                                  # the time at which the simulation stops
   simulProp.set_outputFolder("./Data/confined_propagation")   # the output folder

We will start our simulation with a fracture of 1.3 meters radius. Since we have zero toughness, we can initialize it in the viscosity dominated regime. To do that, we will first instantiate the :py:class:`FractureInitialization.InitializationParameters` object and pass it to the constructor of :py:class:`Fracture.Fracture` class. We will also setup the Controller with a :py:class:`Controller.Controller` object and run the simulation.

.. code-block:: python

   # initializing fracture
   Fr_geometry = Geometry(shape='radial', radius=1.3)
   init_param = InitializationParameters(Fr_geometry, regime='M')

   # creating fracture object
   Fr = Fracture(Mesh,
                 init_param,
                 Solid,
                 Fluid,
                 Injection,
                 simulProp)

   # create a Controller
   controller = Controller(Fr,
                           Solid,
                           Fluid,
                           Injection,
                           simulProp)

   # run the simulation
   controller.run()

Once the simulation is finished, or even when it is running, we can start visualizing the results. To do that, we first load the state of the fracture in the form of a list of :py:class`Fracture.Fracture` objects. From the list, we can extract any fracture variable we want to visualize. Here we first extract the times at which the state of the fracture was evaluated.

.. code-block:: python

   Fr_list, properties = load_fractures(address="./Data/confined_propagation")
   time_srs = get_fracture_variable(Fr_list, variable='time')

Lets first visualize the evolution of the fracture length with time. We can do that using the :py:func:`Visualization.plot_fracture_list` function to plot the 'd_max' variable. We will plot it in loglog scaling for better visualization. To do that, we will pass a :py:class:`Properties.PlotProperties` object with the graph_scaling attribute set to 'loglog'. For better legends of the plot, we will pass a :py:class:`Properties.LabelProperties` object whose legend variable is set to 'fracture length'. The setting up of plot properties and labels is, of course, optional.

.. code-block:: python

   label = LabelProperties('d_max', 'wm')
   label.legend = 'fracture length'

   plot_prop = PlotProperties(line_style='.',
                              graph_scaling='loglog')

   Fig_r = plot_fracture_list(Fr_list,            # plotting fracture length
                              variable='d_max',
                              plot_prop=plot_prop,
                              labels=label)

Lets compare the fracture length with the analytical solutions for the radial and PKN fractures. To do that, we will make use of the :py:func:`Visualization.plot_analytical_solution` function. To superimpose the analytical solutions on the figure we already have generated for the fracture radius (:code:`Fig_r`), we pass it to the function plotting the analytical solution using the :code:`fig` argument.

.. code-block:: python

   label.legend = 'fracture length analytical (PKN)'
   Fig_r = plot_analytical_solution('PKN',
                                     variable='d_max',
                                     mat_prop=Solid,
                                     inj_prop=Injection,
                                     fluid_prop=Fluid,
                                     fig=Fig_r,
                                     time_srs=time_srs,
                                     h=7.0,
                                     labels=label)

   label.legend = 'radius analytical (viscosity dominated)'
   plot_prop.lineColorAnal = 'b'
   Fig_r = plot_analytical_solution('M',
                                     variable='d_max',
                                     mat_prop=Solid,
                                     inj_prop=Injection,
                                     fig=Fig_r,
                                     fluid_prop=Fluid,
                                     time_srs=time_srs,
                                     plot_prop=plot_prop,
                                     labels=label)

.. image:: /images/fracture_length_PKN.png
    :align:   center
    :scale: 80 %

Expectedly, the solution first follows the viscosity dominated radial fracture solution and then transitions to height contained regime for which the classical PKN \cite{PKN61} solution is applicable. The error introduced in the solution at about 2 seconds is due to re-meshing.

There are many fracture variables that we can plot now (you can see a list of variables that can be plotted in the Postprocessing and Visualization section). lets plot the footprint of the fracture in 3D and super impose the viscosity dominated and PKN analytical solutions. We will first load the saved fracture objects at the times at which we want to plot the footprint.

.. code-block:: python

   Fr_list, properties = load_fractures(address="./Data/confined_propagation",
                                        time_srs=np.asarray([1, 5, 20, 50, 80, 110, 140]))
   time_srs = get_fracture_variable(Fr_list,
                                    variable='time')

Note that the fractures closest to the given times are loaded as the solutions are available only at the time steps at which the fractures were saved. The exact times are obtained from the loaded fracture list, at which the analytical solutions will be evaluated.

.. code-block:: python

   plot_prop_mesh = PlotProperties(text_size=1.7, use_tex=True)
   Fig_Fr = plot_fracture_list(Fr_list,                           #plotting mesh
                               variable='mesh',
                               projection='3D',
                               backGround_param='sigma0',
                               mat_properties=properties[0],
                               plot_prop=plot_prop_mesh)

   Fig_Fr = plot_fracture_list(Fr_list,                           #plotting footprint
                               variable='footprint',
                               projection='3D',
                               fig=Fig_Fr)

   Fig_Fr = plot_analytical_solution('PKN',                       #plotting footprint analytical
                                     variable='footprint',
                                     mat_prop=Solid,
                                     inj_prop=Injection,
                                     fluid_prop=Fluid,
                                     fig=Fig_Fr,
                                     projection='3D',
                                     time_srs=time_srs[2:],
                                     h=7.0)
   plt_prop = PlotProperties(line_color_anal='b')
   Fig_Fr = plot_analytical_solution('M',
                                     variable='footprint',
                                     mat_prop=Solid,
                                     inj_prop=Injection,
                                     fluid_prop=Fluid,
                                     fig=Fig_Fr,
                                     projection='3D',
                                     time_srs=time_srs[:2],
                                     plot_prop=plt_prop)

   plot_prop = PlotProperties(alpha=0.2, text_size=5)             #plotting width
   Fig_Fr = plot_fracture_list(Fr_list,
                               variable='w',
                               projection='3D',
                               fig=Fig_Fr,
                               plot_prop=plot_prop)
   ax = Fig_Fr.get_axes()[0]
   ax.view_init(60, -114)

.. image:: /images/footprint_PKN.png
    :align:   center

