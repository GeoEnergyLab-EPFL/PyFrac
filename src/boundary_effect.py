
#external
import numpy as np
import copy
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import lgmres
import logging

#local
from Hdot import *

class BoundaryEffect:
    """
    Class defining the Material properties of the solid.

    Arguments:


    Attributes:

    """

    def __init__(self, Mesh, Eprime, Poissonratio, path2boundaryMesh,
                 preconditioner = True,
                 lgmres = False,
                 maxLeafSizeTrK = 300,
                 etaTrK = 10,
                 maxLeafSizeDispK = 1000,
                 etaDispL = 100,
                 epsACA=0.001):
        """
        The constructor function:
        - check the integrity of the mesh with the one created in pyfrac
        - build the Hmatrix

        Variables:
             coordinates   - const std::vector<double>
             connectivity  - const std::vector<int64_t>
             kernel        - const std::string
             properties    - const std::vector<double>
             max_leaf_size - const int
             eta           - const double
             eps_aca       - const double
        """
        # HMAT parameters
        self.use_preconditioner = preconditioner
        self.use_lgmres = lgmres
        self.max_leaf_size_tr = maxLeafSizeTrK
        self.eta_tr = etaTrK
        self.max_leaf_size_disp = maxLeafSizeDispK
        self.eta_disp = etaDispL
        self.eps_aca = epsACA

        # Load the mesh from the file
        import json
        with open(path2boundaryMesh) as json_file:
            boundarymesh = json.load(json_file)
            print("json file loaded")

        self.bndryMesh = boundarymesh

        coor_bndry = np.asarray(boundarymesh["pts_total"])

        conn_bndry = np.asarray(boundarymesh["conn_Total"])

        properties = [Eprime * (1 - Poissonratio ** 2), Poissonratio]  # Young Modulus , Poisson's ratio

        ### Check that the input mesh is coherent with the one of PyFrac ###
        #
        # check the mesh size
        # reldiff = abs(boundarymesh["hx"] - Mesh.hx) / Mesh.hx
        # if reldiff > 0.05:
        #     raise SystemExit('The size hx is too different (' + str(reldiff) + ' >5%) from the one of the mesh created by PyFrac. \n Expected loss of accuracy ')
        #
        # reldiff = abs(boundarymesh["hy"] - Mesh.hy) / Mesh.hy
        # if reldiff > 0.05:
        #     raise SystemExit('The size hy is too different (' + str(reldiff) + ' >5%) from the one of the mesh created by PyFrac. \n Expected loss of accuracy ')
        #
        # reldiff = abs(boundarymesh["hz"] - Mesh.hx) / Mesh.hx
        # if reldiff > 0.05:
        #     raise SystemExit('The size hz is too different (' + str(reldiff) + ' >5%) from hx of the mesh created by PyFrac. \n Expected loss of accuracy ')
        #
        # reldiff = abs(boundarymesh["hz"] - Mesh.hy) / Mesh.hy
        # if reldiff > 0.05:
        #     raise SystemExit('The size hz is too different (' + str(reldiff) + ' >5%) from hy of the mesh created by PyFrac. \n Expected loss of accuracy ')
        #
        # check that the mesh from PyFrac is inside the boundary
        # this check is valid for parallelepiped
        #
        if coor_bndry[:, 0].max() < Mesh.VertexCoor[:,0].max():
            raise SystemExit('Max x of the mesh in PyFrac is larger than the bounding mesh. ')
        if coor_bndry[:, 0].min() > Mesh.VertexCoor[:,0].min():
            raise SystemExit('Min x of the mesh in PyFrac is smaller than the bounding mesh. ')
        if coor_bndry[:, 1].max() < Mesh.VertexCoor[:,1].max():
            raise SystemExit('Max y of the mesh in PyFrac is larger than the bounding mesh. ')
        if coor_bndry[:, 1].min() > Mesh.VertexCoor[:,1].min():
            raise SystemExit('Min y of the mesh in PyFrac is smaller than the bounding mesh. ')

        ### Add the mesh of PyFrac to the one of the boundary ###
        n_of_vert_fp = Mesh.VertexCoor.shape[0]
        coor_fp_3D = np.zeros([n_of_vert_fp,3])
        for i in range(n_of_vert_fp):
            coor_fp_3D[i, 0] = Mesh.VertexCoor[i, 0]
            coor_fp_3D[i, 1] = Mesh.VertexCoor[i, 1]
            coor_fp_3D[i, 2] = 0.

        #first we have got the coordinates of the boundary and then the one of the fracture plane
        coor = np.concatenate((coor_bndry,coor_fp_3D),axis=0)
        #fisrt we set the connectivity of the boundary and then the one of the fracture plane
        conn = np.concatenate((conn_bndry, (Mesh.Connectivity + sum(boundarymesh["pts_len"]))),axis=0)

        ### DoF indexes in the numeration of the global array of unknwowns (cosidering all the elements of the plane)###
        # number of elements fp
        self.n_of_Elts_fp = Mesh.NumberOfElts

        # number of elements boudary
        self.n_of_Elts_bndry = sum(boundarymesh["conn_len"])

        # total number of DoF fracture plane
        self.n_of_unknowns_fp = Mesh.NumberOfElts * 3

        # total number of DoF boundary
        self.n_of_unknowns_bndry = self.n_of_Elts_bndry * 3

        # total number of DoF
        self.n_of_unknowns_tot = self.n_of_unknowns_fp + self.n_of_unknowns_bndry

        # indexes of the fracture plane DoF indexes
        self.fpINDX = np.arange(2, self.n_of_unknowns_fp, 3) + self.n_of_unknowns_bndry

        # indexes of the boundary and indexes of the crack plane DoF
        self.bndry_and_shear_fpINDX = np.setdiff1d(np.arange(self.n_of_unknowns_tot), self.fpINDX, assume_unique=True)

        ### equation type indexes ###
        # The equation type is:
        #   0 for a traction boundary condition
        #   1 for a displacement boundary condition
        # assuming no displacement BC on the fracture plane
        equationtype = np.asarray(boundarymesh["equation_Type_Face"]).flatten()
        displacemIDX = np.where(equationtype == 1)[0]

        if len(displacemIDX) == 0:
            raise SystemExit('You must fix at least one displacement of the bounding box in order to prevent any rigid body movement ')

        # with the following operation we will implicitly assume a traction boundary condition on the fracture plane
        tractionIDX = np.setdiff1d(np.arange(self.n_of_unknowns_tot),displacemIDX, assume_unique=True)

        # pack the data
        data = (coor.flatten(), conn.flatten(), properties,
                self.max_leaf_size_tr,   self.eta_tr,
                self.max_leaf_size_disp, self.eta_disp, self.eps_aca,
                tractionIDX, displacemIDX,
                self.use_preconditioner)

        #plot to check
        #------------------------
        # from mpl_toolkits.mplot3d import Axes3D
        # from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        # import matplotlib.pyplot as plt
        # import matplotlib.colors as colors
        # import scipy as sp
        #
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # #for ind in range(sum(boundarymesh["conn_len"]),conn.shape[0]):
        # for ind in range(conn.shape[0]):
        #     el = conn[ind]
        #
        #     x = [ coor[el][0][0], coor[el][1][0], coor[el][2][0], coor[el][3][0] ]
        #     y = [ coor[el][0][1], coor[el][1][1], coor[el][2][1], coor[el][3][1] ]
        #     z = [ coor[el][0][2], coor[el][1][2], coor[el][2][2], coor[el][3][2] ]
        #     verts = [list(zip(x, y, z))]
        #     poly = Poly3DCollection(verts,linewidths=0.5, alpha=0.2)
        #     poly.set_color(colors.rgb2hex(sp.rand(3)))
        #     poly.set_edgecolor('k')
        #     ax.add_collection3d(poly)
        #
        # ax.set_xlim3d(-0.08, 0.08)
        # ax.set_ylim3d(-0.08, 0.08)
        # ax.set_zlim3d(-0.08, 0.08)
        # plt.show()
        #------------------------

        #plot to check
        #------------------------
        #from mpl_toolkits import mplot3d
        #import matplotlib.pyplot as plt
        # nop = coor.shape[0] #number of points
        # x = np.zeros(nop)
        # y = np.zeros(nop)
        # z = np.zeros(nop)
        # for pt in range(coor.shape[0]):
        #     x[pt] = coor[pt, 0]
        #     y[pt] = coor[pt, 1]
        #     z[pt] = coor[pt, 2]
        # ax = plt.axes(projection='3d')
        # ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5);
        #------------------------

        # some memory statistics
        getMemUse()

        cost_hmat = round(self.n_of_unknowns_tot * self.n_of_unknowns_tot * 8 / 1024 / 1024 / 1024, 2) #GiB
        print("   -> KERNEL: 3DR0 cost: " + str(cost_hmat) + " GiB")
        print("   -> KERNEL: 3DR0_displ cost: " + str(cost_hmat) + " GiB")
        print("                               ------------")
        print("   -> Total KERNEL cost: " + str(2*cost_hmat) + " GiB")
        print("   -> Total KERNEL + PREC. cost: " + str(3 * cost_hmat) + " GiB")

        #create the Hdot and Mdot (preconditioner)
        self.Hdot = Hdot()
        blockHmat_iLU = self.Hdot.set(data)

        if self.use_preconditioner :
            self.Mdot = Mdot(blockHmat_iLU)

        # set boundary condition values (BCs)
        # - note that we assume 0 as BC on the fracture plane, notably we want to impose 0 traction on the fracture plane
        self.Pu = np.concatenate((np.asarray(boundarymesh["bc_Values"],dtype=float).flatten(),np.zeros(Mesh.NumberOfElts * 3, dtype=float)))

        # to keep memory of the DD on the boundary
        self.all_DD = np.zeros(self.n_of_unknowns_tot)
        self.last_traction = None

        # testing that the preconditioner works
        # initial_vec = np.ones(self.n_of_unknowns_tot)
        # self.Hdot._setRhsOUTindx(np.arange(self.n_of_unknowns_tot))
        # self.Mdot._setRhsOUTindx(np.arange(self.n_of_unknowns_tot))
        # final_vec = self.Mdot._matvec(self.Hdot._matvec(initial_vec))
        # reldiff = np.linalg.norm(initial_vec-final_vec)/self.n_of_unknowns_tot
        # print("------CHECK-------")
        # print("The relative difference between the initial and the final vec is:"+str(reldiff))
        # print("------------------")


    # ------------------------------------------------------------------------------------------------------------------

    def active(self):
        return self.active

    def getTraction(self, w, EltCrack):
        """
        This function updates the confining stress based on the elastic effect of the boundaries due to the current value of
        the fracture opening wk
        Arguments:
             wk (array):        -- the current value of fracture opening.

        Note:
                - For "boundary" we mean both the external boundaries and the tangential displacement discontinuities at the crack plane
        """
        log = logging.getLogger('PyFrac.boundary_effect.getTraction')
        log_only_to_logfile = logging.getLogger('PyFrac_LF.boundary_effect.getTraction')

        # *** get the influence of the crack onto the boundary ***
        # - build an opening array for the whole fracture plane with 0 opening where there is no fracture
        all_w = np.zeros(self.n_of_unknowns_tot)
        all_w[self.fpINDX] = w

        # - set the output indexes
        # here we consider the boundary indexes and the crack-only shear-only indexes
        # indexes of the fracture plane DoF indexes
        fpINDX_shear1 = np.arange(0, self.n_of_unknowns_fp, 3) + self.n_of_unknowns_bndry
        fpINDX_shear2 = np.arange(1, self.n_of_unknowns_fp, 3) + self.n_of_unknowns_bndry
        crackINDX_shear1 = fpINDX_shear1[EltCrack]
        crackINDX_shear2 = fpINDX_shear2[EltCrack]
        bndry_and_shear_crackINDX = np.sort(np.concatenate((np.arange(self.n_of_unknowns_bndry),crackINDX_shear1,crackINDX_shear2)))

        RhsOUTindx = bndry_and_shear_crackINDX
        self.Hdot._setRhsOUTindx(RhsOUTindx)
        if self.use_preconditioner:
            self.Mdot._setRhsOUTindx(RhsOUTindx)

        # - multiply HMAT * [0,0,0,0,..,wi,...,0,0,0]
        rhs = self.Hdot._matvec_full(all_w)

        # *** get the displacement discontinuities at the boundaries ***
        # - set the output indexes
        # The output indexes are already set to be self.boundaryINDX

        # - solve for the boundary displacement discontinuities
        counter = gmres_counter()

        rhs = - rhs + self.Pu[RhsOUTindx]
        maxiter = 500
        restart = 50
        tol = 2.e-7
        if self.use_preconditioner:
            self.Mdot.setHdot(self.Hdot)
            if self.use_lgmres:
                u = lgmres(self.Hdot, rhs,
                          x0=self.all_DD[RhsOUTindx],
                          tol=tol,
                          maxiter=maxiter,
                          callback=counter)
                # u = lgmres(self.Hdot, rhs,
                #           x0=self.all_DD[RhsOUTindx],
                #           tol=tol,
                #           maxiter=maxiter,
                #           M=self.Mdot,
                #           callback=counter)
            else:
                u = gmres(self.Mdot, rhs,
                          x0=self.all_DD[RhsOUTindx],
                          tol=tol,
                          maxiter=maxiter,
                          callback=counter,
                          restart=restart)
                # u = gmres(self.Hdot, rhs,
                #           x0=self.all_DD[RhsOUTindx],
                #           tol=tol,
                #           maxiter=maxiter,
                #           callback=counter,
                #           restart=restart,
                #           M=self.Mdot)

                utemp = self.Mdot._precvec(u[1])
                u = (u[0],utemp)
        else:
            if self.use_lgmres:
                u = lgmres(self.Hdot, rhs,
                          x0=self.all_DD[RhsOUTindx],
                          tol=tol,
                          maxiter=maxiter,
                          callback=counter)
            else:
                u = gmres(self.Hdot, rhs,
                          x0=self.all_DD[RhsOUTindx],
                          tol=tol,
                          maxiter=maxiter,
                          callback=counter,
                          restart=restart)

        # check convergence
        if u[1]>0:
            log.warning("WARNING: BOUNDARY EFF. did NOT converge after "+ str(u[1]) + " iterations!")
            rel_err = np.linalg.norm(self.Hdot._matvec(u[0]) - (rhs))/np.linalg.norm(rhs)
            log.warning("         error of the solution: " + str(rel_err))
        elif u[1]==0:
            rel_err = np.linalg.norm(self.Hdot._matvec(u[0]) - (rhs)) / np.linalg.norm(rhs)
            log.debug(" --> GMRES BOUNDARY EFF. converged after " + str(counter.niter) + " iter. & rel err is " + str(rel_err))

        # *** get the influence of the boundary onto the crack plane ***
        # - make the vector u to fit the whole number of DDs
        all_u = np.zeros(self.n_of_unknowns_tot)
        all_u[bndry_and_shear_crackINDX] = u[0]

        # - set the output indexes
        RhsOUTindx = self.fpINDX
        self.Hdot._setRhsOUTindx(RhsOUTindx)

        # - multiply the matrix for the displacement discontinuities on the boundary
        traction = self.Hdot._matvec_full(all_u )

        # save the solution for u!
        self.all_DD = all_u

#        if self.last_traction is not None:
#            residual = 100*np.linalg.norm(self.last_traction - traction)/np.linalg.norm(self.last_traction )
#            log.info(" Boundary eff. residual:"+str(residual))
        self.last_traction = traction

        return traction

    #-----------------------------------------------------------------------------------------------------------------------

    def getSystemError(self, wk, rhs_crack, EltCrack):
        """

        Args:
            wk: fracture opening on all the cells on the fracture plane
            rhs_crack: traction on all the cells on the fracture plane
            EltCrack: indexes of the element

        Returns:
            error
        """
        # - boundary and shear crack indexes
        fpINDX_shear1 = np.arange(0, self.n_of_unknowns_fp, 3) + self.n_of_unknowns_bndry
        fpINDX_shear2 = np.arange(1, self.n_of_unknowns_fp, 3) + self.n_of_unknowns_bndry
        crackINDX_shear1 = fpINDX_shear1[EltCrack]
        crackINDX_shear2 = fpINDX_shear2[EltCrack]
        bndry_and_shear_crackINDX = np.sort(
            np.concatenate((np.arange(self.n_of_unknowns_bndry), crackINDX_shear1, crackINDX_shear2)))


        # - set the output indexes to be the full matrix minus the dof of the element on the crack plane but not in the crack
        RhsOUTindx = np.sort(np.concatenate((bndry_and_shear_crackINDX,self.fpINDX[EltCrack])))
        self.Hdot._setRhsOUTindx(RhsOUTindx)

        # update the solution for w
        # note that the DD outside of the crack but in the crack plane are all 0
        self.all_DD[self.fpINDX[EltCrack]] = wk[EltCrack]

        # plot solution
        #import matplotlib.pyplot as plt
        #plt.scatter(range(self.n_of_unknowns_tot),self.all_DD)

        # - multiply HMAT * [0,0,ui,0,0,..,wi,...,0,0,0]
        rhs_k = self.Hdot._matvec(self.all_DD[RhsOUTindx])
        rhs = copy.deepcopy(self.Pu)
        rhs[self.fpINDX[EltCrack]] =  rhs_crack[EltCrack]

        error = np.linalg.norm(rhs_k - rhs[RhsOUTindx])/np.linalg.norm(rhs[RhsOUTindx])

        #--------
        # - solve for the boundary displacement discontinuities
        #u = gmres(self.Hdot, rhs[RhsOUTindx], x0=np.zeros(RhsOUTindx.size), tol=1e-12, maxiter=5000)
        #rhs_k = self.Hdot._matvec(u[0])
        #error = np.linalg.norm(rhs_k - rhs[RhsOUTindx]) / np.linalg.norm(rhs[RhsOUTindx])
        #-------
        return error