.. _run-a-simulation:

Running a Simulation
====================

Lets run a simple simulation of a radial fracture propagation. The first step towards running the simulation is to create a mesh describing our domain as a :py:class:`mesh.CartesianMesh` object (see the class documentation for details). PyFrac uses a rectangular mesh to discretize the domain.

.. code-block:: python

   from mesh import CartesianMesh

   Mesh = CartesianMesh(0.3, 0.3, 41, 41)

The above code will generate a rectangular mesh with 41 cells along both the x and y axes, having the dimensions of [x_min=-0.3, x_max=0.3, y_min=-0.3, y_max=0.3] meters. Next, we have to specify the parameters describing the material being fractured and the injected fluid. This is to be done by instantiating the properties classes. Below, we set up a material with the Poisson's ratio of :math:`0.4`, the Young's modulus of :math:`3.3\times10^{10}\,Pa`  and the fracture toughness of :math:`0.005\;Mpa\,\sqrt{m}` by instantiating a :py:class:`properties.MaterialProperties` object:

.. code-block:: python

   from properties import MaterialProperties

   # solid properties
   nu = 0.4                            # Poisson's ratio
   youngs_mod = 3.3e10                 # Young's modulus
   Eprime = youngs_mod / (1 - nu ** 2) # plain strain modulus
   K_Ic = 5e3                          # fracture toughness

   Solid = MaterialProperties(Mesh, Eprime, K_Ic)

The fluid properties are to be set up with a :py:class:`properties.FluidProperties` object. Below we set up a fluid with a viscosity of :math:`1.1\times 10^{-3}\;Pa\,s`:

.. code-block:: python

   from properties import FluidProperties

   # fluid properties
   Fluid = FluidProperties(viscosity=1.1e-3)

Next, we will set up injection parameters with the :py:class:`properties.InjectionProperties` object. For this simulation, we set the injection rate to be :math:`0.001\;m^3/s`:

.. code-block:: python

   from properties import  InjectionProperties

   # injection parameters
   Q0 = 0.001  # injection rate
   Injection = InjectionProperties(Q0, Mesh)

Simulation parameters such as the end time, the times at which the solution is required to be evaluated, the output folder to write data and many others can be set up using :py:class:`properties.SimulationProperties` object (See the class description for the available options). The parameters are read from a file at the time of instantiation. If no file is given, the default values are used. Below, we first instantiate the simulation parameters object with the default values and then set up the parameters that are required to be changed according to our simulation.

.. code-block:: python

   from properties import SimulationProperties

   # simulation properties
   simulProp = SimulationProperties()
   simulProp.finalTime = 1                      # the time at which the simulation stops
   simulProp.set_simulation_name("radial")      # name the simulation "radial"

After setting up of the properties, let us set up the initial state of the fracture that is to be propagated. It is done by creating a :py:class:`fracture.Fracture` object. For this simulation, we set the viscosity dominated analytical solution as the initial state of the fracture. This is done by first creating a :py:class:`fracture_initialization.Geometry` class object and setting up our initial shape as 'radial' and providing the initial radius. After that, we instantiate a :py:class:`fracture_initialization.InitializationParameters` class object and set the regime in which our initial fracture is propagating as "M" (specifying the viscosity dominated regime). This object along with the properties objects that we had instantiated before are passed to the constructor of the Fracture class. For a complete list of options and the ways a fracture can be initialized, see the documentation of the :py:class:`fracture_initialization.InitializationParameters` class. For this simulation, we start with a fracture with a radius of :math:`0.1\,m`.

.. code-block:: python

   from fracture import Fracture
   from fracture_initialization import Geometry, InitializationParameters

   # initialization parameters
   Fr_geometry = Geometry('radial', radius=0.15)
   init_param = InitializationParameters(Fr_geometry, regime='M')

   # creating fracture object
   Fr = Fracture(Mesh,
                 init_param,
                 Solid,
                 Fluid,
                 Injection,
                 simulProp)

After specifying all the properties and initializing the fracture, we will set up a controller and run the simulation.

.. code-block:: python

   from controller import Controller

   # create a Controller
   controller = Controller(Fr,
                           Solid,
                           Fluid,
                           Injection,
                           simulProp)

   # run the simulation
   controller.run()

The :py:func:`controller.Controller.run` function will advance the simulation according to the parameters set in the simulation properties. The state of the fracture is stored in the form of the fracture object in the output folder set up in the simulation properties. A new folder with the name of the simulation and the time stamp at which the simulation was run is created for each of the simulation. If a folder or name is not provided, the simulation will be saved in the default folder (_simulation_data_PyFrac) with the default name (simulation). After the simulation is finished, the results can be post-processed and visualized using the provided visualization routines.