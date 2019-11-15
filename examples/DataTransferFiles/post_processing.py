####################
# plotting results #
####################

from visualization import *
from examples.DataTransferFiles.postprocess_AM import *

# define simulation name and time series to observe
#sim_name = "008_pc_2019-09-23__10_58_32"
sim_name = "015_pc_2019-09-24__08_12_45"
#sim_name = "014_pc_2019-09-23__15_20_54"
#sim_name = "030_pc_A_Qconst__2019-11-13__08_47_07"
#sim_name = "020_pc_A__2019-10-23__07_46_38"
#time_srs = np.asarray([15,3212427434])
#time_period = 1e3

# loading simulation results
if 'time_period' in locals():
    Fr_list, properties = load_fractures(address="./Data/neutral_buoyancy",
                                         sim_name=sim_name,
                                         time_srs=time_period)       # load all fractures
elif 'time_srs' in locals():
    Fr_list, properties = load_fractures(address="./Data/neutral_buoyancy",
                                         sim_name=sim_name,
                                         time_srs=time_srs)       # load all fractures
else:
    Fr_list, properties = load_fractures(address="./Data/neutral_buoyancy",
                                         sim_name=sim_name)       # load all fractures

Solid = properties[0]
Fluid = properties[1]
Injection = properties[2]
SimulationProperties = properties[3]

# exporting the variables
mesh_list = get_fracture_variable(Fr_list, 'mesh')

width_srs = get_fracture_variable(Fr_list,
                                 variable='w')

time_srs = get_fracture_variable(Fr_list,                     # list of times
                                 variable='time')

pressure_srs = get_fracture_variable(Fr_list,
                                       variable='pn')

front_velocity = get_fracture_variable(Fr_list,
                                       variable='v')

extremities = get_extremities_cells(Fr_list)

front_velocity_at_tip = np.empty([len(front_velocity),1])
for i in range(len(front_velocity)):
    front_velocity_at_tip[i] = front_velocity[i][extremities[i,3]]

# plotting normalized opening
sliced_width = get_fracture_variable_slice_cell_center(width_srs[-1], mesh_list[-1], point=[0,0], orientation='vertical')
sliced_pressure = get_fracture_variable_slice_cell_center(pressure_srs[-1], mesh_list[-1], point=[0,0], orientation='vertical')

values = get_bar_values_and_scales(properties)

dimless_width = sliced_width[0]/values[1][2][0]
dimless_pressure = sliced_pressure[0]/values[1][3][0]

lower_ext = mesh_list[-1].CenterCoor[extremities[-1,2]]
theoretical_p = np.sqrt(((sliced_width[1]-lower_ext[1])/values[1][1][0])/(3*0.249*time_srs[-1]/values[1][5][0]))

plt.plot((sliced_width[1]-lower_ext[1])/values[1][1][0],dimless_width)    # blue
plt.plot((sliced_width[1]-lower_ext[1])/values[1][1][0],dimless_pressure) #yellow
plt.plot((sliced_width[1]-lower_ext[1])/values[1][1][0],theoretical_p)

#plt.plot(sliced_width[1],sliced_width[0])
#plt.plot(sliced_width[1],sliced_pressure[0])

# plot footprint
Fig_FP = None
Fig_FP = plot_fracture_list(Fr_list[-2:],
                            variable='mesh',
                            projection='2D',
                            mat_properties=Solid,
                            backGround_param='confining stress')
plt_prop = PlotProperties(plot_FP_time=False)
Fig_FP = plot_fracture_list(Fr_list[-2:],
                            variable='footprint',
                            projection='2D',
                            fig=Fig_FP,
                            plot_prop=plt_prop)

print("Hello")
plt.show()