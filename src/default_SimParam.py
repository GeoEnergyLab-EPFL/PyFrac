### default simulation parameters ###

# tolerances
toleranceFractureFront = 1.0e-3         # tolerance for the fracture front position solver
toleranceEHL = 1.0e-5                   # tolerance for the elastohydrodynamic system solver
tol_projection = 2.5e-3                 # tolerance for the toughness iteration

# max iterations
max_front_itrs = 30                     # maximum iterations for the fracture front
max_solver_itrs = 80                    # maximum iterations for the elastohydrodynamic solver
max_toughness_Itrs = 10                 # maximum toughness iterations

# time and time stepping
tmStp_prefactor = 0.8                   # time step prefactor(pf) to calculate the time step (dt = pf*min(dx, dy)/max(v)
req_sol_at = None                       # times at which the solution is required
final_time = None                       # time to stop the propagation
maximum_steps = 2000                    # maximum time steps
timeStep_limit = None                   # limit for the time step
fixed_time_step = None                  # constant time step

# time step re-attempt
max_reattemps = 8                       # maximum reattempts in case of time step failure
reattempt_factor = 0.8                  # the factor by which time step is reduced on reattempts

# output
plot_figure = False                     # if True, figure will be plotted after the given time period
save_to_disk = True                     # if True, fracture will be saved after the given time period
output_folder = None                    # the address to save the output data
plot_analytical = False                 # if True, analytical solution will also be plotted
analytical_sol = None                   # the analytical solution to be ploted
bck_color = None                        # the parameter according to which background is color coded (see class doc.)
plot_eltType = False                    # plot the element type with color coded dots (channel, ribbon or tip)
output_time_period = None               # the time period after which the output is generated
sim_name = None                         # name given to the simulation
block_figure = False                    # if true, the simulation will proceed after the figure is closed
output_every_TS = None                  # the number of time steps after which the output is done

# type of solver
mech_loading = False                    # if True, the mechanical loading solver will be used
volume_control = False                  # if True, the volume control solver will be used
viscous_injection = True                # if True, the viscous fluid solver solver will be used
substitute_pressure = True              # if True, the pressure will be substituted with width to make the EHL system

# miscellaneous
tip_asymptote = 'U'                     # the tip_asymptote to be used (see class documentation for details)
save_regime = True                      # if True, the the regime of the ribbon cells will also be saved
verbosity = 2                           # the level of details about the ongoing simulation to be plotted
enable_remeshing = True                 # if true, computation domain will be remeshed after reaching end of the domain
remesh_factor = 2.                      # the factor by which the mesh is compressed
front_advancing = 'semi-implicit'       # possible options include 'implicit', 'explicit' and 'semi-implicit'
collect_perf_data = False               # if True, performance data will be collected in the form of a tree
param_from_tip = False                  # set the space dependant tip parameters to be taken from ribbon cell.
save_ReyNumb = False                    # if True, the Reynold's number at each edge will be saved.
save_fluid_flux = False                 # if True, the fluid flux at each edge will be saved.
save_fluid_vel = False                  # if True, the fluid vel at each edge will be saved.
gravity = False                         # if True, the effect of gravity will be taken into account.
TI_Kernel_exec_path = './TI_Kernel'     # the folder containing the executable to calculate TI elasticty matrix.
explict_projection = False              # if True, direction from last time step will be used to evaluate TI parameters
symmetric = False                       # if True, only positive quarter of the cartesian coordinates will be solved

# fracture geometry
height = None                           # fracture height to calculate the analytical solution for PKN or KGD geometry
aspect_ratio = None                     # fracture aspect ratio to calculate the analytical solution for TI case