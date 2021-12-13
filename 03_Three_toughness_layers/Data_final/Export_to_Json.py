from utilities.utility import setup_logging_to_console
from utilities.visualization import *
from utilities.postprocess_fracture import append_to_json_file, load_fractures

# setting up the verbosity level of the log at console
setup_logging_to_console(verbosity_level='debug')


# post processing function
def get_info(Fr_list_A):  # get L(t) and x_max(t) and p(t)
    double_L_A = [];
    x_max_A = [];
    p_A = [];
    w_A=[];
    time_simul_A = [];

    for frac_sol in Fr_list_A:
        center_indx = frac_sol.mesh.locate_element(0., 0.)
        # we are at a give time step now,
        # I am getting double_L_A, x_max_A
        x_min_temp = 0.
        x_max_temp = 0.
        y_min_temp = 0.
        y_max_temp = 0.
        for i in range(frac_sol.Ffront.shape[0]):
            segment = frac_sol.Ffront[i]
            # to find the x_max at this time:
            if segment[0] > x_max_temp:
                x_max_temp = segment[0]
            if segment[2] > x_max_temp:
                x_max_temp = segment[2]
            # to find the n_min at this time:
            if segment[0] < x_min_temp:
                x_min_temp = segment[0]
            if segment[2] < x_min_temp:
                x_min_temp = segment[2]
            # to find the y_max at this time:
            if segment[1] > y_max_temp:
                y_max_temp = segment[1]
            if segment[3] > y_max_temp:
                y_max_temp = segment[3]
            # to find the y_min at this time:
            if segment[1] < y_min_temp:
                y_min_temp = segment[1]
            if segment[3] < y_min_temp:
                y_min_temp = segment[3]

        double_L_A.append(y_max_temp - y_min_temp)
        x_max_A.append(x_max_temp - x_min_temp)

        p_A.append(frac_sol.pFluid.max() / 1.e6)
        w_A.append(frac_sol.w[center_indx].tolist()[0])
        time_simul_A.append(frac_sol.time)
    return double_L_A, x_max_A, p_A, w_A


def export(myfolder, simulation_name, to_export, destination):
    ####################################
    # exporting to multiple json files #
    ####################################

    export_results =True

    if export_results:
        # 1) decide the names of the Json files:
        if 1 in to_export:
            myJsonName_1 = destination + "/TJ_"+simulation_name+"_export.json"           # I will export here most of the infos


        # 2) load the results:
        #
        # >>> Remember that you can select a subset of time steps <<<<
        # >>> otherwise you will export at all the time steps     <<<<
        #
        if 1 in to_export:
            print("\n 1) loading results")
            Fr_list, properties = load_fractures(address=myfolder,load_all=True) # or load_fractures(address=myfolder,time_srs=np.linspace(5., 8.0,600))
            Solid, Fluid, Injection, simulProp = properties
            print(" <-- DONE\n")

        # *) write to json the general informations
        if 2 in to_export:
            time_srs = get_fracture_variable(Fr_list, variable='time')
            time_srs = np.asarray(time_srs)
            double_L, x_max, p_A, w_A = get_info(Fr_list)

            simul_info = {'Eprime': Solid.Eprime,
                          'max_KIc': Solid.K1c.max(),
                          'min_KIc': Solid.K1c.min(),
                          'max_Sigma0': Solid.SigmaO.max(),
                          'min_Sigma0': Solid.SigmaO.min(),
                          'viscosity': Fluid.viscosity,
                          'total_injection_rate': Injection.injectionRate.max(),
                          'sources_coordinates_lastFR': Fr_list[-1].mesh.CenterCoor[Injection.sourceElem].tolist(),
                          't_max': time_srs.max(),
                          't_min': time_srs.min(),
                          '2L': double_L,
                          'xmax': x_max,
                          'times': time_srs.tolist(),
                          'p': p_A,
                          'w': w_A
                          }
            append_to_json_file(myJsonName_1, simul_info, 'append2keyASnewlist', key='simul_info',
                                delete_existing_filename=True)  # be careful: delete_existing_filename=True only the first time you call "append_to_json_file"

        # 3) write to json the coordinates of the points defining the fracture fronts at each time:
        if 3 in to_export:
            print("\n 2) writing fronts")
            time_srs = get_fracture_variable(Fr_list,variable='time') # get the list of times corresponding to each fracture object
            append_to_json_file(myJsonName_1, time_srs, 'append2keyASnewlist', key='time_srs_of_Fr_list')
            fracture_fronts = []
            numberof_fronts = [] #there might be multiple fracture fronts in general
            mesh_info = [] # if you do not make remeshing or mesh extension you can export it only once
            index = 0
            for fracture in Fr_list:
                fracture_fronts.append(np.ndarray.tolist(fracture.Ffront))
                numberof_fronts.append(fracture.number_of_fronts)
                mesh_info.append([Fr_list[index].mesh.Lx, Fr_list[index].mesh.Ly, Fr_list[index].mesh.nx, Fr_list[index].mesh.ny])
                index = index + 1
            append_to_json_file(myJsonName_1, fracture_fronts, 'append2keyASnewlist', key='Fr_list')
            append_to_json_file(myJsonName_1, numberof_fronts, 'append2keyASnewlist', key='Number_of_fronts')
            append_to_json_file(myJsonName_1,mesh_info,'append2keyASnewlist', key='mesh_info')
            print(" <-- DONE\n")



        # 4) get the fracture opening w(t) versus time (t) at a point of coordinates myX and myY
        if 4 in to_export:
            print("\n 3) get w(t) at a point... ")
            my_X = 0.00 ; my_Y = 0.00
            w_at_my_point, time_list_at_my_point = get_fracture_variable_at_point(Fr_list, variable='w', point=[my_X, my_Y])
            append_to_json_file(myJsonName_1, w_at_my_point, 'append2keyASnewlist', key='w_at_my_point')
            append_to_json_file(myJsonName_1, time_list_at_my_point, 'append2keyASnewlist', key='time_list_W_at_my_point')
            print(" <-- DONE\n")



        # 5) get the fluid pressure p(t) versus time (t) at a point of coordinates myX and myY
        if 5 in to_export:
            print("\n 4) get pf(t) at a point... ")
            my_X = 0.00 ; my_Y = 0.00
            pf_at_my_point, time_list_at_my_point = get_fracture_variable_at_point(Fr_list, variable='pf', point=[my_X, my_Y])
            append_to_json_file(myJsonName_1, pf_at_my_point, 'append2keyASnewlist', key='pf_at_my_point_A')
            append_to_json_file(myJsonName_1, time_list_at_my_point, 'append2keyASnewlist', key='time_list_pf_at_my_point_A')
            print(" <-- DONE\n")


        # 6) get the fluid pressure p(t) versus time (t) at a point of coordinates myX and myY
        if 6 in to_export:
            print("\n 5) get pf(t) at a point... ")
            my_X = 0.00 ; my_Y = 0.00
            pf_at_my_point, time_list_at_my_point = get_fracture_variable_at_point(Fr_list, variable='pf', point=[my_X, my_Y])
            append_to_json_file(myJsonName_1, pf_at_my_point, 'append2keyASnewlist', key='pf_at_my_point_B')
            append_to_json_file(myJsonName_1, time_list_at_my_point, 'append2keyASnewlist', key='time_list_pf_at_my_point_B')
            print(" <-- DONE\n")


        # 7) get w(y) along a vertical line passing through mypoint for different times
        if 7 in to_export:
            print("\n 6) get w(y) with y passing through a specific point for different times... ")
            my_X = 0.; my_Y = 0.
            ext_pnts = np.empty((2, 2), dtype=np.float64)
            fracture_list_slice = plot_fracture_list_slice(Fr_list,
                                                           variable='w',
                                                           projection='2D',
                                                           plot_cell_center=True,
                                                           extreme_points=ext_pnts,
                                                           orientation='vertical',
                                                           point1=[my_X , my_Y],
                                                           export2Json=True,
                                                           export2Json_assuming_no_remeshing=False)
            towrite = {'w_vert_slice_': fracture_list_slice}
            append_to_json_file(myJsonName_1, towrite, 'extend_dictionary')
            print(" <-- DONE\n")



        # 8) get w(x) along a horizontal line passing through mypoint for different times
        if 8 in to_export:
            print("\n 7) get w(x) with x passing through a specific point for different times... ")
            my_X = 0.; my_Y = 0.
            ext_pnts = np.empty((2, 2), dtype=np.float64)
            fracture_list_slice = plot_fracture_list_slice(Fr_list,
                                                           variable='w',
                                                           projection='2D',
                                                           plot_cell_center=True,
                                                           extreme_points=ext_pnts,
                                                           orientation='horizontal',
                                                           point1=[my_X , my_Y],
                                                           export2Json=True,
                                                           export2Json_assuming_no_remeshing=False)
            towrite = {'w_horiz_slice_': fracture_list_slice}
            append_to_json_file(myJsonName_1, towrite, 'extend_dictionary')
            print(" <-- DONE\n")



        # 9) get w(x,y,t) and pf(x,y,t)
        if 9 in to_export:
            print("\n 8) get w(x,y,t) and  pf(x,y,t)... ")
            wofxyandt = []
            pofxyandt = []
            info = []
            jump = True #this is used to jump the first fracture
            for frac in Fr_list:
                if not jump:
                    wofxyandt.append(np.ndarray.tolist(frac.w))
                    pofxyandt.append(np.ndarray.tolist(frac.pFluid))
                    info.append([frac.mesh.Lx,frac.mesh.Ly,frac.mesh.nx,frac.mesh.ny,frac.time])
                else:
                    jump = False

            append_to_json_file(myJsonName_1, wofxyandt, 'append2keyASnewlist', key='w')
            append_to_json_file(myJsonName_1, pofxyandt, 'append2keyASnewlist', key='p')
            append_to_json_file(myJsonName_1, info, 'append2keyASnewlist', key='info_for_w_and_p')
            print(" <-- DONE\n")

        print("DONE! in " + myJsonName_1)
        plt.show(block=True)



def check_exe(folders_list, simul_list, to_export, destination):
    #######################################################
    # loop on the files and export to multiple json files #
    #######################################################
    if len(simul_list) != len(simul_list):
        print("ERROR: {simul_list} does not contain the same number of elements as {folder_list} ")
    else:
        n_of_todo = len(simul_list)
        for i in range(n_of_todo):
            print("exporting "+ str(i)+" of "+str(n_of_todo-1)+" :")
            myfolder = folders_list[i]
            simulation_name = simul_list[i]
            export(myfolder, simulation_name, to_export, destination)
    print("---- DONE ----")


####################################
# MAIN                             #
####################################

#-------> SELECT WHAT TO EXPORT:

# 1) mandatory
# 2) mandatory
# 3) write to json the coordinates of the points defining the fracture fronts at each time:
# 4) get the fracture opening w(t) versus time (t) at a point of coordinates myX and myY
# 5) get the fluid pressure p(t) versus time (t) at a point of coordinates myX and myY
# 6) get the fluid pressure p(t) versus time (t) at a point of coordinates myX and myY
# 7) get w(y) along a vertical line passing through mypoint for different times
# 8) get w(x) along a horizontal line passing through mypoint for different times
# 9) get w(x,y,t) and pf(x,y,t)

to_export = [1,2,3,4,7,8]

#-------> SPECIFY OUT ID AND FOLDER NAME:
# simul_list = ["01", "02", "02bis", "03", "04"]#["02bis"]
# folders_list = ["/home/carlo/Desktop/PyFrac/VC_gmres/Data_final/01",
#                 "/home/carlo/Desktop/PyFrac/VC_gmres/Data_final/02",
#                 "/home/carlo/Desktop/PyFrac/VC_gmres/Data_final/02bis",
#                 "/home/carlo/Desktop/PyFrac/VC_gmres/Data_final/03",
#                 "/home/carlo/Desktop/PyFrac/VC_gmres/Data_final/04"]
#folders_list = ["/home/carlo/Desktop/PyFrac/VC_gmres/Data_final/02bis"]
# ["/home/carlo/Desktop/PyFrac/VC_gmres/Data_final/01",
#                 "/home/carlo/Desktop/PyFrac/VC_gmres/Data_final/02"]
#simul_list = ["02break"]
#simul_list = ["05mtoK", "06mtoK"]
simul_list = ["09mtoK"]
common_address = "/home/carlo/Desktop/PyFrac/03_Three_toughness_layers/Data_final/"
common_address = "/home/peruzzo/PycharmProjects/PyFrac/03_Three_toughness_layers/Data_final/"
# folders_list = [common_address + "05mtoK",
#                 common_address + "06mtoK"]
folders_list = [
                common_address + "09mtoK"]
#destination = "/home/carlo/Desktop/PyFrac/03_Three_toughness_layers/Data_final"
destination ="/home/peruzzo/PycharmProjects/PyFrac/03_Three_toughness_layers/Data_final"
# ---GO ---:
check_exe(folders_list, simul_list, to_export, destination)
