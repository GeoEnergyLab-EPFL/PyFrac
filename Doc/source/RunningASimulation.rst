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

Meshing during simulations
-----------------------------------------

PyFrac is based on a planar structured rectangular mesh created at the beginning of all simulations. In the beginning of
this chapter you've seen how to generate a mesh centered on [0, 0]. It is, however, possible to generate a mesh centered
around any point you wish by executing the following:

.. code-block:: python

   from mesh import CartesianMesh

   Mesh = CartesianMesh(0.15, [-0.175, 0.05], 47, 71)

This will generate a mesh having dimensions of [x_min=-0.15, x_max=0.15, y_min=-0.175, y_max=0.05] meters. You can pass
an array of two variables defining the limits of your mesh instead of a half length. Combinations of the two options are
freely possible.

In a next step we need to decide on how the mesh should evolve during the simulation. The default
settings are such that we double the height and breadth of our cells, once the fracture reaches the boundary of our
mesh. For this re-meshing, the number of cells remains the same, so a doubling of the height and breadth results in a
doubling of the discretized domain size. For the mesh given above the dimensions after such a re-meshing would be
[x_min=-0.30, x_max=0.30, y_min=-0.35, y_max=0.1]. If you want to coarsen your mesh by a smaller factor, the re-meshing
factor can be adpated with:

.. code-block:: python

   from properties import SimulationProperties

   simulProp = SimulationProperties()
   simulProp.remeshFactor = 1.5

If you don't want the fracture to extend out of the original mesh. You can simply block re-meshing. In this case the
simulation will stop once you reach the boundary. This can be performed by setting:

.. code-block:: python

   simulProp.enableRemeshing = False

We will refer to this type re-meshing hereafter as the "mesh compression".

We implemented some additional features concerning the evolution of our discretized domain during the simulation. The
first is called a "mesh extension": This means that we add elements on the sides where the fracture is reaching the
boundary of the mesh. By default this feature is disabled to save memory and reduce computation time. It is, however,
possible to allow the extension of the mesh in given directions. This can be achieved by the following command

.. code-block:: python

   simulProp.set_mesh_extension_direction(['top'])

The possible options are 'top', 'bottom', 'left', 'right' meaning respectively that the mesh is extended in direction
of positive y, negative y, negative x, or positive x. Additionally, one can define an extension in 'horizontal',
'vertical' or 'all' directions. Any combination of two parameters is possible for example the line of code

.. code-block:: python

   simulProp.set_mesh_extension_direction(['top', 'left'])

will allow the mesh to extend towards positive y and negative x. The mesh extension factor tells you by which factor
the number of elements is multiplied when the boundary is reached.

.. code-block:: python

   simulProp.set_mesh_extension_factor(1.5)

The mesh extension factor can be chosen different for every direction. Passing a float value will set it to given value
value in all directions. Further options are the passing of a list with two entries (first stands for the factor in x
and second for the factor in y direction) or a list with four entries (respectively giving the factor in negative x,
positive x, negative y and positive y direction). Two important notes on the mesh extension are to be made:

1. Whenever the fracture reaches a boundary where mesh extension has not been turned on, a re-meshing by compressing the
   domain is performed.
2. Whenever the fracture reaches the boundary in all four directions simultaneously, a re-meshing by compression of
   the domain is done.

The second point can be disabled by setting

.. code-block:: python

   simulProp.meshExtensionAllDir = True

In this case the fracture will add elements in the specified directions even if three or four boundaries are reached
simultaneously.

For this last case the number of elements is growing rapidly and computational costs might explode. To counter this
problem we implemented the possibility to reduce the number of cells. The limiting parameter on the computational cost
is the number of elements inside the propagating fracture (respectively the maximum number inside a fracture for several
fractures). By default we set the number of elements allowed inside a fracture to infinity such that no mesh reduction
is performed. As not to coarsen the mesh to much, the user can set a maximum cell breadth (max hx). The initial aspect
ratio of the cells is then used to define the equivalent max cell height. If a mesh reduction would lead to a coarsening
with cell height or breadth bigger than the defined maximum we disable mesh reduction. The following code allows to set
the reduction factor (factor by which the number of cells in x and y will be divided), the maximum number of cells
inside the fracture as well as the max breadth of the cell.

.. code-block:: python

   simulProp.meshReductionFactor = 1.5
   simulProp.maxElementIn = 1000
   simulProp.maxCellSize = 50


Examples of different mesh extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We illustrate the different re-meshing options by several examples as to familiarize the user with the different
options. Our first example has its focus on a propagating dike. The following initial mesh is and re-meshing properties
are specified:

.. code-block:: python

   from mesh import CartesianMesh
   from properties import SimulationProperties

   # creating mesh
   Mesh = CartesianMesh(0.01, 0.01, 31, 31)

   # define the re-meshing parameters
   simulProp.set_mesh_extension_factor(1.5)
   simulProp.set_mesh_extension_direction(['top', 'horizontal'])

In words we start a simulation where we allow the fracture to extend in horizontal (positive and negative x-direction)
as well as to the top (in positive y-dircetion). The mesh extension factor is set to 1.5 and the mesh compression factor
remains at it's default value of 2.

.. image:: /images/width_contours.png
    :align:   center
    :scale: 80 %
