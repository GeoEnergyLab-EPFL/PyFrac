#!/usr/bin/python
import sys
sys.path.append('')
from coalescence_exp_db12_refined_only_for_colaescence import run_simulation
from postprocess_fracture import append_to_json_file

#discretization_list_x=[57,71,85,99,141,183,211]
#discretization_list_y=[30,40,50,60,70,80,90,100,110]
discretization_list_x=[71]
discretization_list_y=[41]

#211x40failed
done=[]
failed=[]
for mesh_discretiz_y in discretization_list_y:
    for mesh_discretiz_x in discretization_list_x:
        try:
            print('\n')
            print('\n')
            print('Running '+str(mesh_discretiz_x)+" x "+str(mesh_discretiz_y))
            run_simulation(mesh_discretiz_x, mesh_discretiz_y)
            print('\n')
            done.append(str(mesh_discretiz_x)+"x"+str(mesh_discretiz_y))
        except:
            failed.append(str(mesh_discretiz_x)+"x"+str(mesh_discretiz_y))
myJsonName = "/home/carlo/Desktop/PyFrac/Paper_OKeffe/Data/coalescence_exp_db3_mesh_study_failed.json"
append_to_json_file(myJsonName, failed, 'append2keyAND2list', key='failed')
