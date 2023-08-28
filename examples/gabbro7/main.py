"""get_fracture_variable
Created by Pedro Lima.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory".
All rights reserved. See the LICENSE.TXT file for more details.
"""

import argparse
import numpy as np
import subprocess

from mesh_obj.mesh import CartesianMesh

from solid.solid_prop import MaterialProperties
from fluid.fluid_prop import FluidProperties
from properties import InjectionProperties, SimulationProperties
from fracture_obj.fracture import Fracture
from fracture_obj.fracture_initialization import Geometry, InitializationParameters
from controller import Controller
from utilities.utility import setup_logging_to_console
from utilities.visualization import *
from utilities.postprocess_fracture import load_fractures

# import pandas as pd
# experimentPressure_MPa = pd.read_csv("/home/pedro/projects/gabbro7/experiment-data/RawData/175323.csv").loc[:," P1"]
# dtExp_sec = 0.2
# experimentTime_sec = [dtExp]

SMALL_NUMBER = 1e-5


class custom_factory:
    def __init__(self, xlabel, ylabel):
        self.number_of_plots = 1
        self.data = {
            "xlabel": xlabel,
            "ylabel": ylabel,
            "xdata": [],
            "p_in_crack": [],
            "p_in_line": [],
        }  # max value of x that can be reached during the simulation

    def custom_plot(self, plot_index, sim_prop, fig=None):
        if plot_index - 1 not in range(self.number_of_plots):
            print(
                f"check the variable number_of_plots in the custom class! \n you asked plot {plot_index} but declared number_of_plots={self.number_of_plots}"
            )
        if plot_index == 1:
            return self.plot_1(sim_prop, fig=fig)
        else:
            print(f" you did not code the plot function in the custom class ")

    def plot_1(self, sim_prop, fig=None):
        # this method is mandatory
        if fig is None:
            fig = plt.figure()
            ax = fig.gca()
        else:
            ax = fig.get_axes()[0]

        ax.scatter(self.data["xdata"], self.data["p_in_line"], color="b")
        ax.scatter(self.data["xdata"], self.data["p_in_crack"], color="r", marker="*")

        ax.set_xlabel(self.data["xlabel"])
        ax.set_ylabel(self.data["ylabel"])
        # ax.set_yscale('log')
        # ax.set_xscale('log')
        return fig

    def postprocess_fracture(
        self, sim_prop, solid_prop, fluid_prop, injection_prop, fr
    ):
        # this method is mandatory
        self.data["xdata"].append(fr.time)
        self.data["p_in_line"].append(fr.pInjLine / 1000000)
        ID = fr.mesh.locate_element(0.0, 0.0)
        self.data["p_in_crack"].append(fr.pFluid[ID[0]] / 1000000)  #
        fr.postprocess_info = self.data
        return fr


parser = argparse.ArgumentParser()
parser.add_argument("-r", "--runQ", type=bool, default=False)
parser.add_argument("-p", "--plotQ", type=bool, default=False)
parser.add_argument("-t", "--finaltime", type=float, default=180)
parser.add_argument("-j", "--jsonQ", type=bool, default=True)
args = parser.parse_args()

runQ = args.runQ
plotQ = args.plotQ
dataPath = "./Data/gabbro7"


# setting up the verbosity level of the log at console
setup_logging_to_console(verbosity_level="debug")

if runQ:
    nu = poissonRatio = 0.29
    youngMod_Pa = 99.7e9
    youngModPlane_Pa = youngMod_Pa / (1 - poissonRatio**2)
    toughness_PaSqrtMeters = (2.79 + 0.11) * 1e6  # Fracture toughness (Pa.m^1/2)
    equivToughness_PaSqrtMeters = np.sqrt(32 / np.pi) * toughness_PaSqrtMeters
    cartersLeakoff_m3perSqrtS = 0.0
    confinementStress_Pa = 5e6

    # initialWidth_m = 1e-8
    initialNetPressure_Pa = 40e6 - confinementStress_Pa
    notchRadius_m = 10.5e-3
    # initialRadius_m = 1.12 * notchRadius_m
    ### To force start at imminence of propagation
    initialRadius_m = (
        1.06
        * (
            np.pi
            * equivToughness_PaSqrtMeters
            / (8 * initialNetPressure_Pa * np.sqrt(2))
        )
        ** 2
    )
    # initialNetPressure_Pa = (
    #     np.pi * equivToughness_PaSqrtMeters / (8 * np.sqrt(2 * initialRadius_m))
    # ) * 0.98
    mesh = CartesianMesh(Lx=initialRadius_m * 5, Ly=initialRadius_m * 5, nx=61, ny=61)

    solid = MaterialProperties(
        Mesh=mesh,
        Eprime=youngModPlane_Pa,
        toughness=toughness_PaSqrtMeters,
        Carters_coef=cartersLeakoff_m3perSqrtS,
        minimum_width=1e-6,
        confining_stress=confinementStress_Pa,
    )

    # Note: For a constant injection use "Qo = your_injection_rate". For injection histories the number of entries in Qo
    # must match with the number of entries in t_change. First entry of Qo is injection for t between the first and second
    # entry of t_change and so on.
    t_change = [0, 13, 23, 73, 123, 173]
    # Qo = [6e-10, 2e-09, 2.7e-09, 2.05e-09, 1.5e-09, 1.1e-09]
    Qo = injectionRate_m3PerSec = 0.08 * 1e-6 / 60
    if type(Qo) == list:
        injectionRate_m3PerSec = np.asarray([t_change, Qo])

    #                                   mL/MPa     m3/mL  MPa/Pa
    lumpedCompressibility_m3perPa = 6.29e-2 * 1e-6 * 1e-6
    initialFluidP = initialNetPressure_Pa + confinementStress_Pa
    injection = InjectionProperties(
        rate=injectionRate_m3PerSec,
        mesh=mesh,
        model_inj_line=True,
        il_compressibility=lumpedCompressibility_m3perPa,
        il_volume=1,
        perforation_friction=0,
        initial_pressure=initialFluidP,
    )

    viscosity = 0.6
    fluidCompressibility_1perPa = 0
    fluid = FluidProperties(
        viscosity=viscosity, compressibility=fluidCompressibility_1perPa
    )

    simulation = SimulationProperties()
    # simulation.saveTSJump, simulation.plotTSJump = 1, 20
    simulation.set_outputFolder(dataPath)
    # simulation.frontAdvancing = "explicit"
    simulation.customPlotsOnTheFly = True
    simulation.custom = custom_factory("time [s]", "pressure [MPa]")
    simulation.plotVar = ["w", "pf", "custom"]
    simulation.plotFigure = True
    simulation.projMethod = "LS_continousfront"

    # Custom timesteps
    simulation.finalTime = 1.30 * args.finaltime  # Seconds
    tMarks_percentage = np.asarray([1, 80, 95, 98, 100, 104, 110]) / 100
    tMarks = args.finaltime * tMarks_percentage
    npoints = np.asarray([2] * (len(tMarks) - 1))
    npoints[-3] = 4
    npoints[-2] = 12
    npoints[-1] = 1
    dts = np.append(np.diff(tMarks) / npoints, 1.0)
    dts[-2] = 0.75
    tMarks[0] = 0
    # dts = np.append(np.diff(tMarks)/2,0.5)
    my_fixed_ts = np.asarray(
        [
            tMarks,
            dts,
        ]
    )
    # my_fixed_ts = np.asarray(
    #     [
    #         [0.0, 130, 150, 160, 164, 166],
    #         [30,  10, 5, 2, 0.5, 1],
    #     ]
    # )
    simulation.fixedTmStp = my_fixed_ts
    # simulation.set_solTimeSeries(
    #     np.asarray(
    #         [
    #             2.1,
    #             2.2,
    #             2.25,
    #             2.3,
    #             2.4,
    #             2.5756,
    #             2.85539,
    #             3.8,
    #             4.9,
    #             6.2,
    #             7.7,
    #             8.0,
    #         ]
    #     )
    # )

    # starting gfrom a prexisting fracture with net pressure specified and opening from elasticity
    from solid.elasticity_isotropic import load_isotropic_elasticity_matrix_toepliz

    Eprime = youngModPlane_Pa

    elasticityMatrix = load_isotropic_elasticity_matrix_toepliz(
        mesh,
        youngModPlane_Pa,
        C_precision=np.float64,
        useHMATdot=False,
        nu=poissonRatio,
    )

    Fr_geometry = Geometry("radial", radius=initialRadius_m)
    initialState = InitializationParameters(
        Fr_geometry,
        regime="static",
        net_pressure=initialNetPressure_Pa,
        # width=initialWidth_m,
        elasticity_matrix=elasticityMatrix,
        time=1,
    )

    fracture = Fracture(
        mesh=mesh,
        init_param=initialState,
        solid=solid,
        fluid=fluid,
        injection=injection,
        simulProp=simulation,
    )

    controller = Controller(
        Fracture=fracture,
        Solid_prop=solid,
        Fluid_prop=fluid,
        Injection_prop=injection,
        Sim_prop=simulation,
    )

    controller.run()


if args.plotQ or args.jsonQ:
    from utilities.postprocess_fracture import load_fractures

    Fr_list, properties = load_fractures(dataPath)
    # Fr_list, properties = load_fractures(address=dataPath, time_srs=np.linspace(0, args.finaltime, steps))
    solid, fluid, injection, simulation = properties


# -------------- exporting to json file -------------- #


# # 1) export general information to json
# # 2) export to json the coordinates of the points defining the fracture front at each time
# # 3) export the fracture opening w(t) versus time (t) at a point of coordinates myX and myY
# # 4) export the fluid pressure p(t) versus time (t) at a point of coordinates myX and myY
# # 5) export w(y) along a vertical line passing through mypoint for different times
# # 6) export pf(x) along a horizontal line passing through mypoint for different times
# # 7) export w(x,y,t) and pf(x,y,t)

to_export = [1, 2, 3, 4, 5, 6, 7, 8, 9]
jsonQ = args.jsonQ
if jsonQ:
    from utilities.postprocess_fracture import append_to_json_file

    # sim_name = simulation.get_simulation_name()
    sim_name = "simulation__2023-08-15__18_15_55"
    # decide the names of the Json files:
    myJsonName_1 = "./Data/Pyfrac_" + sim_name + "_export.json"

    # 1) export general information to json
    if 1 in to_export:
        print("\n 2) writing general info")
        time_srs = get_fracture_variable(Fr_list, variable="time")
        time_srs = np.asarray(time_srs)

        simul_info = {
            "Eprime": solid.Eprime,
            "max_KIc": solid.K1c.max(),
            "min_KIc": solid.K1c.min(),
            "max_Sigma0": solid.SigmaO.max(),
            "min_Sigma0": solid.SigmaO.min(),
            "viscosity": fluid.viscosity,
            "total_injection_rate": injection.injectionRate.max(),
            "sources_coordinates_lastFR": Fr_list[-1]
            .mesh.CenterCoor[injection.sourceElem]
            .tolist(),
            "t_max": time_srs.max(),
            "t_min": time_srs.min(),
        }
        append_to_json_file(
            myJsonName_1,
            simul_info,
            "append2keyASnewlist",
            key="simul_info",
            delete_existing_filename=True,
        )  # be careful: delete_existing_filename=True only the first time you call "append_to_json_file"

    #     # 2) export the coordinates of the points defining the fracture front at each time:
    if 2 in to_export:
        print("\n 2) writing fronts")
        time_srs = get_fracture_variable(
            Fr_list, variable="time"
        )  # get the list of times corresponding to each fracture object
        append_to_json_file(
            myJsonName_1, time_srs, "append2keyASnewlist", key="time_srs_of_Fr_list"
        )
        fracture_fronts = []
        numberof_fronts = []  # there might be multiple fracture fronts in general
        mesh_info = (
            []
        )  # if you do not make remeshing or mesh extension you can export it only once
        index = 0
        for fracture in Fr_list:
            fracture_fronts.append(np.ndarray.tolist(fracture.Ffront))
            numberof_fronts.append(fracture.number_of_fronts)
            mesh_info.append(
                [
                    Fr_list[index].mesh.Lx,
                    Fr_list[index].mesh.Ly,
                    Fr_list[index].mesh.nx,
                    Fr_list[index].mesh.ny,
                ]
            )
            index = index + 1
        append_to_json_file(
            myJsonName_1, fracture_fronts, "append2keyASnewlist", key="Fr_list"
        )
        append_to_json_file(
            myJsonName_1, numberof_fronts, "append2keyASnewlist", key="Number_of_fronts"
        )
        append_to_json_file(
            myJsonName_1, mesh_info, "append2keyASnewlist", key="mesh_info"
        )
        print(" <-- DONE\n")

    # 3) export the fracture opening w(t) versus time (t) at a point of coordinates myX and myY
    if 3 in to_export:
        print("\n 3) get fracture radius... ")
        radius = get_fracture_variable(Fr_list, variable="d_mean")
        append_to_json_file(
            myJsonName_1, radius, "append2keyASnewlist", key="fractureRadius"
        )
        print(" <-- DONE\n")

    # 4) export the fluid pressure p(t) versus time (t) at a point of coordinates myX and myY
    if 4 in to_export:
        print("\n 4) get pf(t) at a point... ")
        my_X = 0.0
        my_Y = 0.0
        pf_at_my_point, time_list_at_my_point = get_fracture_variable_at_point(
            Fr_list, variable="pf", point=[[my_X, my_Y]]
        )
        append_to_json_file(
            myJsonName_1, pf_at_my_point, "append2keyASnewlist", key="pf_at_my_point_A"
        )
        append_to_json_file(
            myJsonName_1,
            time_list_at_my_point,
            "append2keyASnewlist",
            key="time_list_pf_at_my_point_A",
        )
        print(" <-- DONE\n")

    # 5) export w(y) along a vertical line passing through mypoint for different times
    if 5 in to_export:
        print(
            "\n 5) get w(y) with y passing through a specific point for different times... "
        )
        my_X = 0.0
        my_Y = 0.0
        ext_pnts = np.empty((2, 2), dtype=np.float64)
        fracture_list_slice = plot_fracture_list_slice(
            Fr_list,
            variable="w",
            projection="2D",
            plot_cell_center=True,
            extreme_points=ext_pnts,
            orientation="horizontal",
            point1=[my_X, my_Y],
            export2Json=True,
            export2Json_assuming_no_remeshing=False,
        )
        towrite = {"w_vert_slice_": fracture_list_slice}
        append_to_json_file(myJsonName_1, towrite, "extend_dictionary")
        print(" <-- DONE\n")

    # 6) export pf(x) along a horizontal line passing through mypoint for different times
    if 6 in to_export:
        print(
            "\n 6) get pf(x) with x passing through a specific point for different times... "
        )
        my_X = 0.0
        my_Y = 0.0
        ext_pnts = np.empty((2, 2), dtype=np.float64)
        fracture_list_slice = plot_fracture_list_slice(
            Fr_list,
            variable="pf",
            projection="2D",
            plot_cell_center=True,
            extreme_points=ext_pnts,
            orientation="horizontal",
            point1=[my_X, my_Y],
            export2Json=True,
            export2Json_assuming_no_remeshing=False,
        )
        towrite = {"pf_horiz_slice_": fracture_list_slice}
        append_to_json_file(myJsonName_1, towrite, "extend_dictionary")
        print(" <-- DONE\n")

    # 7) export w(x,y,t) and pf(x,y,t)
    if 7 in to_export:
        print("\n 7) get w(x,y,t) and  pf(x,y,t)... ")
        wofxyandt = []
        pofxyandt = []
        info = []
        jump = True  # this is used to jump the first fracture
        for frac in Fr_list:
            if not jump:
                wofxyandt.append(np.ndarray.tolist(frac.w))
                pofxyandt.append(np.ndarray.tolist(frac.pFluid))
                info.append(
                    [frac.mesh.Lx, frac.mesh.Ly, frac.mesh.nx, frac.mesh.ny, frac.time]
                )
            else:
                jump = False

        append_to_json_file(myJsonName_1, wofxyandt, "append2keyASnewlist", key="w")
        append_to_json_file(myJsonName_1, pofxyandt, "append2keyASnewlist", key="p")
        append_to_json_file(
            myJsonName_1, info, "append2keyASnewlist", key="info_for_w_and_p"
        )
        print(" <-- DONE\n")

    subprocess.run(["cp", myJsonName_1, "./Data/latest_exported_simulation.json"])
    print("DONE! in " + myJsonName_1)

if plotQ:
    # loading simulation results
    time_srs = get_fracture_variable(Fr_list, variable="time")  # list of times

    # plot fracture radius
    plot_prop = PlotProperties()
    plot_prop.lineStyle = "."  # setting the line style to point
    # plot_prop.graphScaling = 'loglog'       # setting to log log plot
    Fig_R = plot_fracture_list_at_point(Fr_list, variable="pf", plot_prop=plot_prop)
    Fig_R = plot_fracture_list_at_point(
        Fr_list, variable="pn", plot_prop=plot_prop, fig=Fig_R
    )
    # Fig_R = plot_fracture_list_at_point(Fr_list,
    #                            variable='ir',
    #                            plot_prop=plot_prop)
    # Fig_R = plot_fracture_list_at_point(Fr_list,
    #                            variable='tir',
    #                            plot_prop=plot_prop)
    # plot analytical radius
    # Fig_R = plot_analytical_solution(regime='M',
    #                                  variable='d_mean',
    #                                  mat_prop=solid,
    #                                  inj_prop=injection,
    #                                  fluid_prop=fluid,
    #                                  time_srs=time_srs,
    #                                  fig=Fig_R)
    # plot analytical radius
    # Fig_R = plot_analytical_solution(regime='K',
    #                                  variable='d_mean',
    #                                  mat_prop=solid,
    #                                  inj_prop=injection,
    #                                  fluid_prop=fluid,
    #                                  time_srs=time_srs,
    #                                  fig=Fig_R)
    plt.show(block=True)
