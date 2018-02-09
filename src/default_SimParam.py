### default simulation parameters ###

# tolerances
toleranceFractureFront = 1.0e-3         # tolerance for the fracture front position solver
toleranceEHL = 1.0e-5                   # tolerance for the elastohydrodynamic system solver
tol_toughness = 2.5e-3                    # tolerance for the toughness iteration

# max iterations
maxfront_its = 30                       # maximum iterations for the fracture front
max_itr_solver = 100                    # maximum iterations for the elastohydrodynamic solver
max_toughnessItr = 20                   # maximum toughness iterations

# time and time stepping
tmStp_prefactor = 0.8                   # time step prefactor(pf) to calculate the time step (dt = pf*min(dx, dy)/max(v)
req_sol_at = None                       # times at which the solution is required
final_time = 1000                       # time to stop the propagation
maximum_steps = 2000                    # maximum time steps
timeStep_limit = None                   # limit for the time step
tmStp_fact_limit = 2.0                  # limit on to what factor the time step can increase between two successive
                                        # time steps

# time step re-attempt
max_reattemps = 8                       # maximum reattempts in case of time step failure
reattempt_factor = 0.8                  # the factor by which time step is reduced on reattempts

# output
plot_figure = True                     # if True, figure will be plotted after the given time period
save_to_disk = False                    # if True, fracture will be saved after the given time period
out_file_folder = "None"                # the address to save the output data
plot_analytical = False                 # if True, analytical solution will also be plotted
analytical_sol = None                   # the analytical solution to be ploted
bck_color = None                        # the parameter according to which background is color coded (see class doc.)
plot_eltType = False                    # plot the element type with color coded dots (channel, ribbon or tip)
output_time_period=None                 # the time period after which the output is generated

# type of solver
mech_loading = False                    # if True, the mechanical loading solver will be used
volume_control = False                  # if True, the volume control solver will be used
viscous_injection = True                # if True, the viscous fluid solver solver will be used

# miscellaneous
tip_asymptote = 'U'                     # the tip_asymptote to be used (see class documentation for details)
save_regime = True                      # if True, the the regime of the ribbon cells will also be saved
verbosity = 1                           # the level of details about the ongoing simulation to be plotted
remesh_factor = 2.                      # the factor by which the mesh is compressed