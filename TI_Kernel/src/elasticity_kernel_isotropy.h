//
// Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
// Geo-Energy Laboratory, 2016-2019.  All rights reserved.
// See the LICENSE.TXT file for more details.
//
// Contributors:
// Weihan Zhang
// Fatima-Ezzahra Moukhtari


#ifndef INC_HFPX3D_ELAST_KER_ISO_H
#define INC_HFPX3D_ELAST_KER_ISO_H


#include <il/Array2D.h>
#include <src/mesh.h>

namespace hfp3d {

  //  Kernel elasticity for the isotropic case
    il::Array2D<double> CIMatrix(Mesh mesh,double Ep);
}
#endif
