# default simulation parameters

# tolerances
toleranceFractureFront=1.0e-3
toleranceEHL=1.0e-5
tol_toughness=1e-3

# max iterations
maxfront_its=30
max_itr_solver=100
max_toughnessItr=60

# time and time steping
tmStp_prefactor=0.4
req_sol_at=None
final_time=10000.
maximum_steps=10000

# time step re-attempt
max_reattemps=8
reattempt_factor=0.8

# output
plot_figure=False
save_to_disk = False
out_file_folder = "None"
plot_analytical = False
analytical_sol = "M"
output_time_period=1e-10

# type of solver
mech_loading=False
volume_control=False
viscous_injection=True

# miscellaneous
tip_asymptote='U'