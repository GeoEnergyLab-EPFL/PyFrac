
import numpy as np
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import LinearOperator

import pypart
from pypart import Bigwhamio
from boundary_effect_mesh import boundarymesh

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
  def __init__(self, data):
    # instantiating the objects and variables
    self.HMATdispl = Bigwhamio()
    self.HMATtract = Bigwhamio()
    self.unknowns_number_ = None

    # unpaking the data
    coor, conn, properties, max_leaf_size, eta, eps_aca, tractionIDX, displacemIDX = data

    # set the objects
    self.HMATdispl.set(coor,
                       conn,
                       "3DR0_displ", #kernel
                       properties,
                       max_leaf_size,
                       eta,
                       eps_aca)

    self.HMATtract.set(coor,
                       conn,
                       "3DR0",
                       properties,
                       max_leaf_size,
                       eta,
                       eps_aca)
    # checks
    nodes_per_element_ = 4
    number_of_elements_ = len(conn) / nodes_per_element_
    if len(conn) % nodes_per_element_ != 0 :
        print(" ERROR: \n ")
        print(" wrong connectivity dimension \n ")

    # define the total number of unknowns
    unknowns_per_element_ = 3
    self.unknowns_number_ = int(number_of_elements_ * unknowns_per_element_)

    # it is mandatory to define shape and dtype
    self.shape_ = (self.unknowns_number_, self.unknowns_number_)
    self.dtype_ = float
    super().__init__(self.dtype_, self.shape_)

    # set the equation indexes
    self._setEquationIDX(tractionIDX, displacemIDX)

  def _matvec(self, v):
    """
    This function implements the dot product.
    :param v: vector expected to be of size unknowns_number_
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
    Rhs = np.zeros(self.unknowns_number_)
    Rhs[self.tractionIDX] = Rhs_t[self.tractionIDX]
    Rhs[self.displacemIDX] = Rhs_d[self.displacemIDX]
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
      self.displacemIDX = displacemIDX
      self.tractionIDX = tractionIDX

  def _changeShape(self, shape_):
      self.shape_ = shape_
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

    def __init__(self, active, Mesh, Eprime, Poissonratio):
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
        if active:
            # Load the mesh from the file
            self.boundarymesh = boundarymesh

            coor = np.asarray(boundarymesh["pts_total"])

            conn = np.asarray(boundarymesh["conn_Total"])

            properties = [Eprime, Poissonratio]  # Young Modulus , Poisson's ratio

            ### Check that the input mesh is coherent with the one of PyFrac ###
            #
            # check the mesh size
            reldiff = abs(boundarymesh["hx"] - Mesh.hx) / Mesh.hx
            if reldiff > 0.05:
                raise SystemExit('The size hx is too different (' + str(reldiff) + ' >5%) from the one of the mesh created by PyFrac. \n Expected loss of accuracy ')

            reldiff = abs(boundarymesh["hy"] - Mesh.hy) / Mesh.hy
            if reldiff > 0.05:
                raise SystemExit('The size hy is too different (' + str(reldiff) + ' >5%) from the one of the mesh created by PyFrac. \n Expected loss of accuracy ')

            reldiff = abs(boundarymesh["hz"] - Mesh.hx) / Mesh.hx
            if reldiff > 0.05:
                raise SystemExit('The size hz is too different (' + str(reldiff) + ' >5%) from hx of the mesh created by PyFrac. \n Expected loss of accuracy ')

            reldiff = abs(boundarymesh["hz"] - Mesh.hy) / Mesh.hy
            if reldiff > 0.05:
                raise SystemExit('The size hz is too different (' + str(reldiff) + ' >5%) from hy of the mesh created by PyFrac. \n Expected loss of accuracy ')
            #
            # check that the mesh from PyFrac is inside the boundary
            # this check is valid for parallelepiped
            #
            if coor[:, 0].max() < Mesh.VertexCoor[:,0].max():
                raise SystemExit('Max x of the mesh in PyFrac is larger than the bounding mesh. ')
            if coor[:, 0].min() > Mesh.VertexCoor[:,0].min():
                raise SystemExit('Min x of the mesh in PyFrac is smaller than the bounding mesh. ')
            if coor[:, 1].max() < Mesh.VertexCoor[:,1].max():
                raise SystemExit('Max y of the mesh in PyFrac is larger than the bounding mesh. ')
            if coor[:, 1].min() > Mesh.VertexCoor[:,1].min():
                raise SystemExit('Min y of the mesh in PyFrac is smaller than the bounding mesh. ')

            ### Add the mesh of PyFrac to the one of the boundary ###
            number_of_vertexes = Mesh.VertexCoor.shape[0]
            coor_fr_plane_3D = np.zeros([number_of_vertexes,3])
            for i in range(number_of_vertexes):
                coor_fr_plane_3D[i, 0] = Mesh.VertexCoor[i, 0]
                coor_fr_plane_3D[i, 1] = Mesh.VertexCoor[i, 1]
                coor_fr_plane_3D[i, 2] = 0.

            coor = np.concatenate((coor,coor_fr_plane_3D),axis=0)

            conn = np.concatenate((conn, (Mesh.Connectivity + sum(boundarymesh["pts_len"]))),axis=0)

            ### fracture plane indexes in the numeration of the ###
            self.numberofUnknowns = (Mesh.NumberOfElts + sum(boundarymesh["conn_len"])) * 3
            self.fpINDX = np.arange(0, Mesh.NumberOfElts, 3) + sum(boundarymesh["conn_len"] * 3)
            self.boundaryINDX = np.setdiff1d(np.arange(self.numberofUnknowns), self.fpINDX, assume_unique=True)

            # HMAT parameters
            self.max_leaf_size = 1000000
            self.eta = 1000.
            self.eps_aca = 0.001

            ### equation type indexes ###
            # The equation type is:
            #   0 for a traction boundary condition
            #   1 for a displacement boundary condition

            equationtype = np.asarray(boundarymesh["equation_Type_Face"]).flatten()
            displacemIDX = np.where(equationtype == 1)[0]

            if len(displacemIDX) == 0:
                raise SystemExit('You must fix at least one displacement of the bounding box in order to prevent any rigid body movement ')

            # in the following we will implicitly assume a traction boundary condition on the fracture plane
            tractionIDX = np.setdiff1d(np.arange(self.numberofUnknowns),displacemIDX, assume_unique=True)

            # pack the data
            data = (coor.flatten(), conn.flatten(), properties, self.max_leaf_size, self.eta, self.eps_aca, tractionIDX, displacemIDX)

            self.Hdot = Hdot(data)

            # set boundary conditions (BCs)
            # - note that we assume 0 as BC on the fracture plane, notably we want to impose 0 traction on the fracture plane
            self.Pu = np.concatenate((np.asarray(boundarymesh["bc_Values"].flatten()),np.zeros(Mesh.NumberOfElts * 3, dtype=float)))


        self.active = active


    # ------------------------------------------------------------------------------------------------------------------

    def active(self):
        return self.active

    def getTraction(self, w):
        """
        This function updates the confining stress based on the elastic effect of the boundaries due to the current value of
        the fracture opening wk
        Arguments:
             wk (array):        -- the current value of fracture opening.

        Note:
                - For "boundary" we mean both the external boundaries and the tangential displacement discontinuities at the crack plane
        """
        # *** get the influence of the crack onto the boundary ***
        # - build an opening array for the whole fracture plane with 0 opening where there is no fracture
        all_w = np.zeros(self.numberofUnknowns)
        all_w[self.fpINDX] = w

        # - set the output indexes
        RhsOUTindx = self.boundaryINDX
        self.Hdot._setRhsOUTindx(RhsOUTindx)

        # - multiply HMAT * [0,0,0,0,..,wi,...,0,0,0]
        rhs = self.Hdot_matvec(all_w)

        # *** get the displacement discontinuities at the boundaries ***
        # - set the output indexes
        # The output indexes are already set to be self.boundaryINDX

        # - solve for the boundary displacement discontinuities
        u = gmres(self.Hdot, self.Pu + rhs)

        # *** get the influence of the boundary onto the crack plane ***
        # - make the vector u to fit the whole number of DDs
        all_u = np.zeros(self.numberofUnknowns)
        all_u[self.boundaryINDX] = u

        # - set the output indexes
        RhsOUTindx = self.fpINDX
        self.Hdot._setRhsOUTindx(RhsOUTindx)

        # - multiply the matrix for the displacement discontinuities on the boundary
        traction = self.Hdot._matvec(all_u )
        return traction

    #-----------------------------------------------------------------------------------------------------------------------
