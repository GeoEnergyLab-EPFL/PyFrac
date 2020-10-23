//
// Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
// Geo-Energy Laboratory, 2016-2019.  All rights reserved.
// See the LICENSE.TXT file for more details.
//
// Contributors:
// Weihan Zhang


#ifndef HFPX3D_VC_SELFCORR_H
#define HFPX3D_VC_SELFCORR_H

#include "src/mesh.h"
#include <fstream>
#include<iostream>
#include <il/math.h>
#include<il/linearAlgebra.h>
#include <il/Array.h>
#include <il/Array2D.h>

namespace hfp3d{

    // Self influence correction
    il::Array2D<double> Self_corr(il::Array2D<double> global);


   //local matrix for the crack
    il::Array2D<double> K_local(il::Array2D<double> global);

    //correction of the dislocation (0 for the point out of the crack)
    //which is just a test that is unused in the code
    il::Array<double> dd_corr(il::Array<double> dd);

}

#endif
