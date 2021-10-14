import os
import numpy as np

from visualization import *
output_fol = "./Data/radial_VC_gmres_res"

# loading simulation results A
Fr_list_A, properties_A = load_fractures(address=output_fol, load_all=True)  # load all fractures
time_srs_A = get_fracture_variable(Fr_list_A, variable='time')  # list of times
Solid_A, Fluid_A, Injection_A, simulProp_A = properties_A

tA = np.asarray(time_srs_A)
tmax = tA.max()
pos = np.where(tA==tmax)[0]

# test
sorting = np.argsort(time_srs_A)
sum = 0.
for i in range(len(time_srs_A)):
    sum = sum + time_srs_A[i] - time_srs_A[sorting[i]]

sheck  = np.sum(sorting - np.arange(len(time_srs_A)))

basepath = '/home/carlo/Desktop/PyFrac/VC_gmres/Data/radial_VC_gmres_res/simulation__2021-08-04__11_45_32'
basename = '/simulation__2021-08-04__11_45_32_file_'

for i in range(len(time_srs_A)):
    ext_old = str(sorting[i])
    ext_new = str(i)+"_A"
    os.rename(r'/home/carlo/Desktop/PyFrac/VC_gmres/Data/radial_VC_gmres_res/simulation__2021-08-04__11_45_32'+basename+ext_old,r'/home/carlo/Desktop/PyFrac/VC_gmres/Data/radial_VC_gmres_res/simulation__2021-08-04__11_45_32'+basename+ext_new)


for i in range(len(time_srs_A)):
    ext_old = str(i)+"_A"
    ext_new = str(i)
    os.rename(r'/home/carlo/Desktop/PyFrac/VC_gmres/Data/radial_VC_gmres_res/simulation__2021-08-04__11_45_32'+basename+ext_old,r'/home/carlo/Desktop/PyFrac/VC_gmres/Data/radial_VC_gmres_res/simulation__2021-08-04__11_45_32'+basename+ext_new)