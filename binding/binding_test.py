# import numpy as np
# import pypart
#
# from pyparty import Bigwhamio
#
#
# # Defining the variables:
#
# # coordinates   - const std::vector<double>
# # connectivity  - const std::vector<int64_t>
# # kernel        - const std::string
# # properties    - const std::vector<double>
# # max_leaf_size - const int
# # eta           - const double
# # eps_aca       - const double
#
# coor =[-1., -1., 0.,
#        1., -1., 0.,
#        1., 1., 0.,
#        -1., 1., 0.,
#        -1., 2., 0.,
#        1., 2., 0.]
#
# conn =[0, 1, 2, 3, 3, 2, 5, 4]
#
# properties = [100,0.2] # Young Modulus , Poisson's ratio
#
# max_leaf_size = 1
# eta = 0.
# eps_aca = 0.001
#
# displacementKernel = "3DR0_displ"
# displacementHMAT = Bigwhamio()
# # set the object
# displacementHMAT.set(coor,
#                  conn,
#                  displacementKernel,
#                  properties,
#                  max_leaf_size,
#                  eta,
#                  eps_aca)
#
#
# tractionKernel = "3DR0_traction"
# tractionHMAT = Bigwhamio()
# # set the object
# tractionHMAT.set(coor,
#       conn,
#       tractionKernel,
#       properties,
#       max_leaf_size,
#       eta,
#       eps_aca)
#
# # flattened collocation points
# mycollp = tractionHMAT.getCollocationPoints()
# print(mycollp)
# print("\n")
#
# # hdot product
# print("Testing the Hdot product for the tractionHMAT \n")
# tractions = tractionHMAT.hdotProduct([1.,2.,3.,4.,5.,6.])
# print(tractions)
#
# print("Testing the Hdot product for the displacementHMAT \n")
# tractions = displacementHMAT.hdotProduct([1.,2.,3.,4.,5.,6.])
# print(tractions)
#
# mysol = [1.,1.,1.,1.,1.,1.]
# obsPoints = [-10.,-10.,0., #point 1
#              20.,-20.,0.]  #point 2
#
# stresses = tractionHMAT.computeStresses(mysol, obsPoints, 2, properties, coor, conn, True)
# print("point 1 ")
# print(stresses[1:6])
# print("point 2 ")
# print(stresses[7:12])
#
# x,y,z,a,b,G,nu = [0.,0.,0.,1.,1.,200.,0.3]
# mystress = tractionHMAT.getInfluenceCoe(x,y,z,a,b,G,nu)
# print("\n ------------------ \n ")
# print(" Stress: \n ")
# print("DDx (shear)  -> | sxx, syy, szz, sxy, sxz, syz  | ")
# print("DDy (shear)  -> | sxx, syy, szz, sxy, sxz, syz  | ")
# print("DDy (normal) -> | sxx, syy, szz, sxy, sxz, syz  | ")
# print(mystress[0:6])
# print(mystress[6:12])
# print(mystress[12:18])
#
# x,y,z,a,b,G,nu = [1.5,1.5,0.,2.5,2.,200.,0.3]
# mystress = tractionHMAT.getInfluenceCoe(x,y,z,a,b,G,nu)
# print("\n ------------------ \n ")
# print(" Stress: \n ")
# print("DDx (shear)  -> | sxx, syy, szz, sxy, sxz, syz  | ")
# print("DDy (shear)  -> | sxx, syy, szz, sxy, sxz, syz  | ")
# print("DDy (normal) -> | sxx, syy, szz, sxy, sxz, syz  | ")
# print(mystress[0:6])
# print(mystress[6:12])
# print(mystress[12:18])
# print(mystress)
#
# mydisplacements = displacementHMAT.computeDisplacements(mysol, obsPoints, 2, properties, coor, conn, True)
# print("point 1 ")
# print(mydisplacements[0:3])
# print("point 2 ")
# print(mydisplacements[3:7])


import numpy as np
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import LinearOperator
import pypart
from pypart import Bigwhamio

class Hdot(LinearOperator):

  def __init__(self, kernel, data):
    self.HMAT = Bigwhamio()

    coor, conn, properties, max_leaf_size, eta, eps_aca = data

    if kernel == "3DR0_displ" or kernel == "3DR0":
      # set the object
      self.HMAT.set(coor,
                   conn,
                   kernel,
                   properties,
                   max_leaf_size,
                   eta,
                   eps_aca)

      nodes_per_element_ = 4
      number_of_elements_ = len(conn) / nodes_per_element_
      if len(conn) % nodes_per_element_ != 0 :
        print(" ERROR: \n ")
        print(" wrong connectivity dimension \n ")
      unknowns_per_element_ = 3
      unknowns_number_ = int(number_of_elements_ * unknowns_per_element_)
      self.shape_ = (unknowns_number_, unknowns_number_)
      self.dtype_ = float
      # we must define shape and dtype
      super().__init__(self.dtype_, self.shape_)
    else:
       print(" ERROR: \n ")
       print(" Unknown kernel \n ")

  def _matvec(self, v):
    return self.HMAT.hdotProduct(v)

  @property
  def _init_shape(self):
    return self.shape_

  def _init_dtype(self):
    return self.dtype_

#--------------------------------


# Defining the variables:

# coordinates   - const std::vector<double>
# connectivity  - const std::vector<int64_t>
# kernel        - const std::string
# properties    - const std::vector<double>
# max_leaf_size - const int
# eta           - const double
# eps_aca       - const double

coor =[-1., -1., 0.,
       1., -1., 0.,
       1., 1., 0.,
       -1., 1., 0.,
       -1., 2., 0.,
       1., 2., 0.]

conn =[0, 1, 2, 3, 3, 2, 5, 4]

properties = [100,0.2] # Young Modulus , Poisson's ratio

max_leaf_size = 1
eta = 0.
eps_aca = 0.001

#kernel = "3DR0_displ"
kernel = "3DR0"

data = (coor, conn, properties, max_leaf_size, eta, eps_aca )

B = Hdot(kernel, data)
v = np.asarray([1.,1.,1.,1.,1.,1.])
print(B.matvec(v))

print(B * v)

res = gmres(B,v)
if res[1] == 0:
  print("successful gmres \n")
  print("the result \n")
  print(res[0])
elif res[1] > 0:
  print("convergence to tolerance not achieved, number of iterations ")
elif res[1] < 0:
  print("illegal input or breakdown ")
