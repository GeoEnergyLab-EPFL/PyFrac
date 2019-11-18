####################
# plotting results #
####################

from visualization import *
from examples.DataTransferFiles.postprocess_AM import *

# define simulation name and time series to observe
#sim_name = "008_pc_2019-09-23__10_58_32"
#sim_name = "015_pc_2019-09-24__08_12_45"
#sim_name = "014_pc_2019-09-23__15_20_54"
#sim_name = "030_pc_A_Qconst__2019-11-13__08_47_07"
sim_name = "020_pc_A__2019-10-23__07_46_38"
#sim_name = "028_pc_A__2019-11-06__07_58_17"
#time_srs = np.asarray([15,3212427434])
time_period = 100

# loading simulation results
if 'time_period' in locals():
    Fr_list, properties = load_fractures(address="./Data/neutral_buoyancy",
                                         sim_name=sim_name,
                                         time_period=time_period)       # load all fractures
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


eval_step = -1
# Getting normalized opening
sliced_width = get_fracture_variable_slice_cell_center(width_srs[eval_step], mesh_list[eval_step], point=[0,0], orientation='vertical')
sliced_pressure = get_fracture_variable_slice_cell_center(pressure_srs[eval_step], mesh_list[eval_step], point=[0,0], orientation='vertical')

values = get_bar_values_and_scales(properties,2700) # getting the scales

dimless_width = sliced_width[0]/values[1][2][0]
dimless_pressure = sliced_pressure[0]/values[1][3][0]

lower_ext = mesh_list[eval_step].CenterCoor[extremities[eval_step,2]]
upper_ext = mesh_list[eval_step].CenterCoor[extremities[eval_step,3]]

# get the fracture length and the stable breadth

if upper_ext[1] >= 0:
    eval_point = [0,1/2*(upper_ext[1]+lower_ext[1])+0.25*upper_ext[1]]
else:
    eval_point = [0,1/2*(upper_ext[1]+lower_ext[1])]

intercepts = get_front_intercepts(Fr_list,eval_point)

b_stable = (intercepts[eval_step][3] - intercepts[eval_step][2])/2
print("The stable breadth is")
print(b_stable)
print(" evaluate at point: ")
print(eval_point)
l_frac = intercepts[eval_step][0] - intercepts[eval_step][1]
print(l_frac)

# evaluating the pressure gradient in the head
pressure_diff = np.diff(sliced_pressure[0])
ind_negp = np.where(pressure_diff < 0)
ind_max = np.where(sliced_pressure[0] == np.amax(sliced_pressure[0]))
test =np.where(ind_negp[0] < ind_max[0][0])
fit = np.polyfit(sliced_pressure[1][ind_negp[0][test[0][-1]]+1:ind_max[0][0]+1],
            sliced_pressure[0][ind_negp[0][test[0][-1]]+1:ind_max[0][0]+1], 1)

alpha_num = (b_stable/values[1][0][0])**(3/2)
print("alpha num is:")
print(alpha_num)

# theoretical pressure / opening
theoretical_p = np.sqrt(((sliced_width[1]-lower_ext[1])/values[1][1][0])/(3*0.249*time_srs[eval_step]/values[1][5][0]))


# plotting.
plt.figure(0)
plt.plot((sliced_width[1]-lower_ext[1])/values[1][1][0],dimless_width,'b.')  # blue
plt.plot((sliced_width[1]-lower_ext[1])/values[1][1][0],dimless_pressure,'r.') #yellow
plt.plot((sliced_width[1]-lower_ext[1])/values[1][1][0],theoretical_p,'k-')
plt.ylabel('$\Omega,\Pi$')
plt.xlabel('$\zeta$')
plt.grid(True)

plt.figure(1)
plt.plot(sliced_pressure[1][ind_negp[0][test[0][-1]]+1:ind_max[0][0]+1],
         sliced_pressure[1][ind_negp[0][test[0][-1]]+1:ind_max[0][0]+1]*fit[0]+fit[1])
plt.plot(sliced_pressure[1],sliced_pressure[0],'r.') #yellow
plt.ylabel('$p$')
plt.xlabel('$z$')
plt.grid(True)

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


plt.show(block=True)