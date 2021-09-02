from Hdot import gmres_counter
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import lgmres
from visualization import *
from elasticity import load_isotropic_elasticity_matrix_toepliz

######################################################
# post processing functions
######################################################
def get_info(Fr_list_A):  # get L(t) and x_max(t) and p(t)
    double_L_A = [];
    x_max_A = [];
    p_A = [];
    w_A=[];
    time_simul_A = [];
    center_indx = (Fr_list_A[0]).mesh.locate_element( 0., 0.)
    for frac_sol in Fr_list_A:
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
        w_A.append(frac_sol.w[center_indx])
        time_simul_A.append(frac_sol.time)
    return double_L_A, x_max_A, p_A,w_A,  time_simul_A
#--------------------------------
def getW(fr):
    #
    # compute the fracture opening due to unitary pressure
    #
    p = 1. # unitary pressure

    # prepare the elasticity matrix
    C = load_isotropic_elasticity_matrix_toepliz(fr.mesh, Solid_A.Eprime)
    C._set_domain_IDX(fr.EltCrack)
    C._set_codomain_IDX(fr.EltCrack)

    # filling fraction correction for element in the tip region
    r = fr.FillF - .25
    indx = np.where(np.less(r, 0.1))[0]
    r[indx] = 0.1
    ac = (1 - r) / r
    correction_val = ac * np.pi / 4.
    C._set_tipcorr(correction_val, fr.EltTip)

    # solving the system using left or right preconditioner
    left_prec = True
    if left_prec:
        rhs = np.full(len(fr.EltCrack),p)
        rhs = C._precJm1dotvec(rhs)
        C.left_precJ = True #enable the left preconditioner
        counter = gmres_counter()  # to obtain the number of iteration and residual
        sol_GMRES = gmres(C, rhs, tol = 1e-14, maxiter = 1000, callback = counter)
        final_sol = sol_GMRES[0]
    else:
        rhs = np.full(len(fr.EltCrack),p)
        C.right_precJ = True #enable the left preconditioner
        counter = gmres_counter()  # to obtain the number of iteration and residual
        sol_GMRES = gmres(C, rhs, tol = 1e-14, maxiter = 1000, callback = counter)
        final_sol = C._precJm1dotvec(sol_GMRES[0])
        C.right_precJ = False

    # check convergence
    if sol_GMRES[1] > 0:
        print("WARNING: Volume control system did NOT converge after " + str(sol_GMRES[1]) + " iterations!")
        rel_err = np.linalg.norm(C._matvec(final_sol) - (rhs)) / np.linalg.norm(rhs)
        print("         error of the solution: " + str(rel_err))
    elif sol_GMRES[1] == 0:
        rel_err = np.linalg.norm(C._matvec(final_sol) - (rhs)) / np.linalg.norm(rhs)
        print(" --> GMRES BOUNDARY EFF. converged after " + str(counter.niter) + " iter. & rel err is " + str(rel_err))

        # check if the solution obtained with the good left preconditoner is also good for the right one
        # y = C._precJdotvec(final_sol)
        # C.right_precJ = True  # enable the left preconditioner
        # rhs_rightprec = np.full(len(fr.EltCrack),p)
        # rel_err_right = np.linalg.norm(C._matvec(y)-rhs_rightprec)/ np.linalg.norm(rhs)
        # print("         error of the solution of the right prec system: " + str(rel_err_right))
        # C.right_precJ = False

    w_ = np.zeros(fr.mesh.NumberOfElts)
    w_[fr.EltCrack] = final_sol

    # to check the solution:
    # from utility import plot_as_matrix
    # plot_as_matrix(w_, fr.mesh)
    return w_
#--------------------------------
def loop_on_Aiwi(data_i):
    [w_i, A_i, FillF_i, tip_i] = data_i
    sumAi = 0.
    sumAi_wi = 0.

    for i in range(len(w_i)):
        if i in tip_i:
            ind = np.where(tip_i == i)[0]
            FillF = FillF_i[ind]
        else:
            FillF = 1.

        if w_i[i] == 0.:
            FillF = 0.

        sumAi = sumAi + A_i * FillF
        sumAi_wi = sumAi_wi + A_i * FillF * w_i[i]

    return sumAi, sumAi_wi
#--------------------------------
def getG(data_i, data_im1):
    sumAi, sumAi_wi = loop_on_Aiwi(data_i)
    sumAim1, sumAi_wi_m1 = loop_on_Aiwi(data_im1)
    return 0.5 * ((sumAi_wi - sumAi_wi_m1) / (sumAi - sumAim1))[0]
#--------------------------------

######################################################
# main
######################################################
#output_fol = "./Data/temp"
#output_fol ="./Data/1p45_noHmat"
#output_fol ="./Data/1p45_noHmat_0p05regul"
output_fol ="./Data/1p42_noHmat_0p05regul"
simulation_name = "G_1p42_noHmat_0p05regul"
output_fol ="./Data/1p37_noHmat_0p005regul"
simulation_name = "G_1p37_noHmat_0p005regul"
output_fol = "./Data/1p47_noHmat_0p0005regul"
simulation_name = "G_1p47_noHmat_0p0005regul"
#simulation_name = "G_1p45_noHmat_coarse_tol"
#output_fol ="./Data/1p40_noHmat_0p01regul"
#simulation_name = "G_1p40_noHmat_0p01regul"
#output_fol = "./Data/noHmat_DisTime_fastQ"
#simulation_name = "1p9_noHmat_DisTime_fastQ"
compute_G = True
save_res = True
myJsonName_1 = "./Data/TJ_"+simulation_name+"_export.json"



if compute_G:
    # loading simulation results A
    Fr_list_A_total, properties_A = load_fractures(address=output_fol, load_all=True)  # load all fractures
    time_srs_A = get_fracture_variable(Fr_list_A_total, variable='time')  # list of times
    Solid_A, Fluid_A, Injection_A, simulProp_A = properties_A
    double_L_A, x_max_A, p_A, w_A, time_simul_A = get_info(Fr_list_A_total)

    # taking a subset of the solution
    endpoint = len(Fr_list_A_total)
    #endpoint = 60
    step = 4

    Fr_list_A = []
    selection = []
    for i in range(0,endpoint,step):
        selection.append(i)
        Fr_list_A.append(Fr_list_A_total[i])

    double_L_A_sel = []
    for i in selection:
        double_L_A_sel.append(double_L_A[i])
    double_L_A_sel.pop(-1)

    # plot the different fronts
    plot_prop = PlotProperties()
    Fig_R = plot_fracture_list(Fr_list_A,
                               variable='footprint',
                               plot_prop=plot_prop)
    Fig_R = plot_fracture_list(Fr_list_A,
                               fig=Fig_R,
                               variable='mesh',
                               mat_properties=properties_A[0],
                               backGround_param='K1c',
                               plot_prop=plot_prop)
    plt.show()

    Glist = []
    NoT = len(Fr_list_A)
    for i in range(1,NoT):
        print("\n Processing: " + str(int(100.*i/(NoT-1))) +' %')

        fr_i = Fr_list_A[i]
        fr_im1 = Fr_list_A[i-1]

        w_i = getW(fr_i)
        w_im1 = getW(fr_im1)

        A_i = fr_i.mesh.hx * fr_i.mesh.hy
        A_im1 = fr_im1.mesh.hx * fr_im1.mesh.hy

        FillF_i = fr_i.FillF
        FillF_im1 = fr_im1.FillF

        tip_i = fr_i.EltTip
        tip_im1 = fr_im1.EltTip

        data_i = [w_i, A_i, FillF_i, tip_i]
        data_im1 = [w_im1, A_im1, FillF_im1, tip_im1]

        Glist.append(getG(data_i, data_im1))

    # making it non dimensional
    H = 1.48 * 2
    for i in range(len(double_L_A_sel)):
        double_L_A_sel[i] = double_L_A_sel[i] / H

    Eprime = Solid_A.Eprime
else:
    import json

    # Opening JSON file
    f = open(myJsonName_1, )

    # returns JSON object as
    # a dictionary
    data = json.load(f)
    double_L_A_sel = data['simul_info']['double_L_A/H']
    Glist = data['simul_info']['G']
    Eprime = data['simul_info']['Eprime']
    H = data['simul_info']['H']

######################################################
# plot w_cemter vs time
######################################################
xlabel = 'L/H [-]'
ylabel = 'G*Eprime [...]'
fig, ax = plt.subplots()

if compute_G:
    for i in range(len(Glist)):
        Glist[i] = np.sqrt(Glist[i] * Eprime)
ax.scatter(double_L_A_sel, Glist, color='k')

G_PS = []
for i in range(len(double_L_A_sel)):
    #G_PS.append(np.pi*H/(4.*Eprime))
    G_PS.append(np.sqrt(np.pi * H / (4.)))
ax.plot(double_L_A_sel, G_PS, color='b')

G_Limit = []
for i in range(len(double_L_A_sel)):
    #G_PS.append(np.pi*H/(4.*Eprime))
    G_Limit.append((0.5e6)**2/ Eprime)
ax.plot(double_L_A_sel, G_Limit, color='r')

G_Radial = []
for i in np.arange(0.5,1.5,0.3):
    G_Radial.append(np.sqrt(4./np.pi * (i * H/2.) ))
ax.plot(np.arange(0.5,1.5,0.3), G_Radial, color='b')

ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.set_yscale('log')
ax.set_xscale('log')
plt.show()

######################################################
# export to Json
######################################################
if compute_G and save_res :
    simul_info = {'double_L_A/H': double_L_A_sel,
                  'G': Glist,
                  'Eprime': Eprime,
                  'H': H}

    append_to_json_file(myJsonName_1, simul_info, 'append2keyASnewlist', key='simul_info',
                        delete_existing_filename=True)  # be careful: delete_existing_filename=True only the first time you call "append_to_json_file"
