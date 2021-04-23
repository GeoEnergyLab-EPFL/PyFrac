
#external
import numpy as np
import copy
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spilu
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import LinearOperator
import logging


from boundary_effect_mesh import boundarymesh

import pypart
from pypart import Bigwhamio
from pypart import pyGetFullBlocks

def getMemUse():
    # some memory statistics
    import os, psutil

    process = psutil.Process(os.getpid())
    byte_use = process.memory_info().rss  # byte
    GiByte_use = byte_use / 1024 / 1024 / 1024  # GiB
    print("Current memory use: " + str(GiByte_use) + " GiB")
    return GiByte_use

def getPermutation(HMATobj):
    permut = HMATobj.getPermutation()
    pdof = []
    for i in range(len(permut)):
        pdof.extend([3 * permut[i], 3 * permut[i] + 1, 3 * permut[i] + 2])
    return pdof

def applyPermutation(HMATobj, row_ind, col_ind):
    pdof = np.asarray(getPermutation(HMATobj))
    col_ind = np.asarray(pdof)[col_ind]
    row_ind = np.asarray(pdof)[row_ind]
    del pdof
    return row_ind, col_ind

def deleteNonUsedRows(row_ind, col_ind, values, toBeDeletedROWs): #todo: this method looks too fill up the memory (~HMAT)
    toBeSavedRows = np.in1d(row_ind, toBeDeletedROWs, assume_unique=False)
    row_ind=row_ind[toBeSavedRows]
    col_ind=col_ind[toBeSavedRows]
    values=values[toBeSavedRows]
    del toBeSavedRows
    return row_ind, col_ind, values

def checkOvelappingEntries(row_ind_tract, row_ind_displ):
    row_ind_tract = np.unique(row_ind_tract)
    row_ind_displ = np.unique(row_ind_displ)
    commonRows = np.intersect1d(row_ind_tract,row_ind_displ,assume_unique=True)
    if len(commonRows) > 0:
        print("ERROR: common Rows have been detected")
        import time
        time.sleep(1.)
        exit()

##############################
# Mdot operator for GMRES    #
##############################
class Mdot(LinearOperator):
  def __init__(self, blockHmat_iLU):
    self.dtype_ = float
    self.shape_ = blockHmat_iLU.shape
    self.blockHmat_iLU = blockHmat_iLU
    super().__init__(self.dtype_, self.shape_)

  def _matvec(self, v):
    """
    This function implements the dot product.
    :param v: vector expected to be of size unknowns_number_
    :return: HMAT.v, where HMAT is a matrix obtained by selecting equations from either HMATtract or HMATdispl
    """
    all_v = np.zeros(self.HMAT_size_)
    all_v[self.rhsOUTindx] = v
    Rhs = self.blockHmat_iLU.solve(v)
    return Rhs[self.rhsOUTindx]

  def _setRhsOUTindx(self, RhsOUTindx):
    """
    This function sets the index to be in output from the multiplication in _matvet
    :param RhsOUTindx: indexes to output
    :return: -
    """
    self.rhsOUTindx = RhsOUTindx
    self._changeShape(RhsOUTindx.size)

  def _changeShape(self, shape_):
    self.shape_ = (shape_,shape_)
    super().__init__(self.dtype_, self.shape_)

  @property
  def _init_shape(self):
    return self.shape_

  def _init_dtype(self):
    return self.dtype_

##############################
# Hdot operator for GMRES    #
##############################
class Hdot(LinearOperator):
  """
    This is a special Hdot operator.
    When it is instantiated for the first time, it builds two Hmatrices. These are related respectively to a traction kernel and a displacement kernel
    In this contest, a traction kernel is a linear operator that links the displacement discontinuities (DDs) to traction, while
    a displacement kernel links DDs to displacements.
    Then the Hdot provides the multiplication of a vector of DD over a set of equation taken from the two Kernels pending some restrictions.
    Bearing in mind that the equations are selected by row indexes, the restrictions are:
        1-You can not take the same index (or equation) form both the kernel
        2-The final set of equation selected must be equal to the number of rows equal to the one of each kernel

  """
  def __init__(self):
      self.unknowns_number_ = None
      self.matvec_size_ = None
      self.HMAT_size_ = None
      self.shape_ = None
      self.dtype_ = float
      self.HMATtract = Bigwhamio()
      self.HMATdispl = Bigwhamio()

  def set(self, data):
    import pypart
    from pypart import Bigwhamio
    # instantiating the objects and variables

    # unpaking the data
    coor, conn, properties, \
    max_leaf_size_tr, eta_tr, \
    max_leaf_size_disp, eta_disp, \
    eps_aca, \
    tractionIDX, displacemIDX, use_preconditioner = data

    # checks
    nodes_per_element_ = 4
    n_of_elts_ = int(len(conn) / nodes_per_element_)
    if len(conn) % nodes_per_element_ != 0 :
        print(" ERROR: \n ")
        print(" wrong connectivity dimension \n ")

    # define the HMAT size
    # define the total number of unknowns to be output by the matvet method
    unknowns_per_element_ = 3
    self.HMAT_size_ = int(n_of_elts_ * unknowns_per_element_)
    self.matvec_size_ = self.HMAT_size_

    # it is mandatory to define shape and dtype of the dot product
    self.shape_ = (self.matvec_size_, self.matvec_size_)
    super().__init__(self.dtype_, self.shape_)

    # set the equation indexes to make the mixed traction-displacement system
    self._setEquationIDX(tractionIDX, displacemIDX)

    # set the objects

    self.HMATtract.set(coor.tolist(),
                       conn.tolist(),
                       "3DR0",
                       properties,
                       max_leaf_size_tr,
                       eta_tr,
                       eps_aca)
    print("The compression ratio for 3DR0 kernel is: "+ str(self.HMATtract.getCompressionRatio()))

    if use_preconditioner:
        # ---> the memory here consist mainly of the Hmat
        myget = pyGetFullBlocks()
        myget.set(self.HMATtract)
        # ---> the memory here consist at most of 3 * Hmat
        col_ind_tract = myget.getColumnN()
        row_ind_tract = myget.getRowN()
        # here we need to permute the rows and columns
        # ---> the memory here consist at most of 3 * Hmat
        [row_ind_tract, col_ind_tract] = applyPermutation(self.HMATtract, row_ind_tract, col_ind_tract )
        # ---> the memory here consist at most of 5 * Hmat for a while and it goes back to 4 Hmat
        values_tract = myget.getValList()
        del myget
        # the following methow quickly fills up the memory leaving it unchanged before and after its application
        [row_ind_tract, col_ind_tract, values_tract] = deleteNonUsedRows(row_ind_tract, col_ind_tract, values_tract,
                                                                         displacemIDX)

    self.HMATdispl.set(coor,
                       conn,
                       "3DR0_displ", #kernel
                       properties,
                       max_leaf_size_disp,
                       eta_disp,
                       eps_aca)
    print("The compression ratio for 3DR0_displ kernel is: "+ str(self.HMATdispl.getCompressionRatio()))

    if use_preconditioner:
        # ---> the memory here consist mainly of 2 Hmat
        myget = pyGetFullBlocks()
        myget.set(self.HMATdispl)
        # ---> the memory here consist at most of 4 Hmat
        col_ind_displ = myget.getColumnN()
        row_ind_displ = myget.getRowN()
        # here we need to permute the rows and columns
        # ---> the memory here consist at most of 4 Hmat
        [row_ind_displ, col_ind_displ] = applyPermutation(self.HMATdispl, row_ind_displ, col_ind_displ )
        # ---> the memory here consist at most of 5 Hmat
        values_displ = myget.getValList()
        del myget
        # ---> the memory here consist at most of 5 times the Hmat
        # the following methow quickly fills up the memory
        [row_ind_displ, col_ind_displ, values_displ] = deleteNonUsedRows(row_ind_displ, col_ind_displ, values_displ,
                                                                         tractionIDX)
        # ---> the memory here consist at most of 5 times the Hmat
        # the following method uses 1 times the Hmat of memory to apply
        checkOvelappingEntries(row_ind_tract, row_ind_displ)
        # ---> the memory here consist at most of 5 times the Hmat
        # the following concatenations are slow and fills the memory up
        row_ind_tract = np.concatenate((row_ind_tract,row_ind_displ))
        del row_ind_displ
        col_ind_tract = np.concatenate((col_ind_tract,col_ind_displ))
        del col_ind_displ
        values_tract = np.concatenate((values_tract,values_displ))
        del values_displ
        # ---> the memory here consist at most of 3 times the Hmat -> can not be less
        #blockHmat = csr_matrix((values_tract, (row_ind_tract, col_ind_tract)), shape=self.shape_)
        blockHmat = csc_matrix((values_tract, (row_ind_tract, col_ind_tract)), shape=self.shape_)
        del values_tract, row_ind_tract, col_ind_tract


        ### Compute an incomplete LU decomposition for a sparse, square matrix. ###
        # The resulting object is an approximation to the inverse of blockHmat.
        # Drop tolerance (0 <= tol <= 1) for an incomplete LU decomposition. (default: 1e-4)
        # Specifies the fill ratio upper bound (>= 1.0) for ILU. (default: 10)
        # To improve the better approximation to the inverse, you may need to increase fill_factor AND decrease drop_tol.
        blockHmat_iLU = spilu(blockHmat, drop_tol=1e-5, fill_factor=9)
        del blockHmat
        return blockHmat_iLU
    else :
        return None

  def _matvec(self, v):
    """
    This function implements the dot product.
    :param v: vector expected to be of size unknowns_number_
    :return: HMAT.v, where HMAT is a matrix obtained by selecting equations from either HMATtract or HMATdispl
    """
    all_v = np.zeros(self.HMAT_size_)
    all_v[self.rhsOUTindx] = v
    Rhs_t = self.HMATtract.hdotProduct(all_v)
    Rhs_d = self.HMATdispl.hdotProduct(all_v)
    Rhs = self._selectEquations(Rhs_t,Rhs_d)
    return Rhs[self.rhsOUTindx]

  def _matvec_full(self, v):
    """
    This function implements the dot product.
    :param v: vector expected to be of size self.HMAT_size_
    :return: HMAT.v, where HMAT is a matrix obtained by selecting equations from either HMATtract or HMATdispl
    """
    Rhs_t = self.HMATtract.hdotProduct(v)
    Rhs_d = self.HMATdispl.hdotProduct(v)
    Rhs = self._selectEquations(Rhs_t,Rhs_d)
    return Rhs[self.rhsOUTindx]

  def _selectEquations(self, Rhs_t, Rhs_d):
    """
    This functions select the results from two arrays according to the choice expressed by Rhs_t, Rhs_d
    :param Rhs_t: vector of indexes of the equations from the traction HMAT to be considered
    :param Rhs_d: vector of indexes of the equations from the displacement HMAT to be considered
    :return: Rhs is a vector with the required values and size unknowns_number_
    """
    Rhs = np.zeros(self.HMAT_size_)
    Rhs[self.tractionIDX] = np.asarray(Rhs_t)[self.tractionIDX]
    Rhs[self.displacemIDX] = np.asarray(Rhs_d)[self.displacemIDX]
    return Rhs

  def _setRhsOUTindx(self, RhsOUTindx):
    """
    This function sets the index to be in output from the multiplication in _matvet
    :param RhsOUTindx: indexes to output
    :return: -
    """
    self.rhsOUTindx = RhsOUTindx
    self._changeShape(RhsOUTindx.size)

  def _setEquationIDX(self, tractionIDX, displacemIDX):
      # this function sets the indexes in order to assembly the final system according to the boundary conditions
      # all the crack plane is assumed to have a traction BC
      self.displacemIDX = displacemIDX
      self.tractionIDX = tractionIDX

  def _changeShape(self, shape_):
      self.shape_ = (shape_,shape_)
      super().__init__(self.dtype_, self.shape_)

  @property
  def _init_shape(self):
    return self.shape_

  def _init_dtype(self):
    return self.dtype_

#--------------------------------


class BoundaryEffect:
    """
    Class defining the Material properties of the solid.

    Arguments:


    Attributes:

    """

    def __init__(self, Mesh, Eprime, Poissonratio):
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

        # Load the mesh from the file
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

        # HMAT parameters
        self.max_leaf_size_tr = 5000
        self.eta_tr = 0.
        self.max_leaf_size_disp = 5000
        self.eta_disp = 0.
        self.eps_aca = 0.001
        self.use_preconditioner = True
        ### equation type indexes ###
        # The equation type is:
        #   0 for a traction boundary condition
        #   1 for a displacement boundary condition

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

        cost_hmat = self.n_of_unknowns_tot * self.n_of_unknowns_tot * 8 / 1024 / 1024 / 1024 #GiB
        print("traction kernel cost: " + str(cost_hmat) + " GiB")

        cost_hmat = self.n_of_unknowns_tot * self.n_of_unknowns_tot * 8 / 1024 / 1024 / 1024 #GiB
        print("total boundary cost: " + str(2*cost_hmat) + " GiB")
        print("total boundary cost with preconditioner: " + str(3 * cost_hmat) + " GiB")

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
        self.Mdot._setRhsOUTindx(RhsOUTindx)

        # - multiply HMAT * [0,0,0,0,..,wi,...,0,0,0]
        rhs = self.Hdot._matvec_full(all_w)

        # *** get the displacement discontinuities at the boundaries ***
        # - set the output indexes
        # The output indexes are already set to be self.boundaryINDX

        # - solve for the boundary displacement discontinuities
        rhs = - rhs + self.Pu[RhsOUTindx]
        maxiter = 5000
        tol = 1e-11
        if self.use_preconditioner:
            u = gmres(self.Hdot, rhs, x0=self.all_DD[RhsOUTindx], tol=tol, maxiter=maxiter, M=self.Mdot)
        else:
            u = gmres(self.Hdot, rhs, x0=self.all_DD[RhsOUTindx], tol=tol, maxiter=maxiter)

        # check convergence
        if u[1]>0:
            log.warning("WARNING: gmres did not converge after "+ str(u[1]) + " iterations!")
            rel_err = np.linalg.norm(self.Hdot._matvec(u[0]) - (rhs))/np.linalg.norm(rhs)
            log.warning("         error of the solution: " + str(rel_err))
#        elif u[1]==0:
#            rel_err = np.linalg.norm(self.Hdot._matvec(u[0]) - (rhs)) / np.linalg.norm(rhs)
#            log.info(" Boundary eff. GMRES:" + str(rel_err))

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