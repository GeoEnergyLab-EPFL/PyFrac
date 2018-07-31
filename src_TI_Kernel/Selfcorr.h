//
// Created by Student on 6/7/2017.
//

#ifndef HFPX3D_VC_SELFCORR_H
#define HFPX3D_VC_SELFCORR_H

#include "mesh.h"
#include <fstream>
#include<iostream>
#include <vector>
namespace hfp3d{

    // Self influence correction
    double Self_corr[][] (vector<vector<double>> global);


   //local matrix for the crack
    double K_local[][](double global[][]);

    //correction of the dislocation (0 for the point out of the crack)
    //which is just a test that is unused in the code
    double dd_corr[](double dd[]);



}



#endif //HFPX3D_VC_SELFCORR_H
