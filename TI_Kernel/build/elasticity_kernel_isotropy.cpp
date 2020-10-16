//
// Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
// Geo-Energy Laboratory, 2016-2019.  All rights reserved.
// See the LICENSE.TXT file for more details.
//
// Contributors:
// Weihan Zhang
// Fatima-Ezzahra Moukhtari

#include <il/math.h>
#include <src/elasticity_kernel_isotropy.h>


namespace hfp3d {

   ////////////////////////////////////Elasticity kernel for isotropic case
    il::Array2D<double> CIMatrix(Mesh mesh,double Ep) {
        double a = mesh.dx / 2;
        double b = mesh.dy / 2;
        int nelts = mesh.nelts();

        il::Array2D<double> CI{nelts, nelts};
        for (int i = 0; i < nelts; i++) {

            for (int j = 0; j < nelts; j++) {

                double x = mesh.node(i)[0] - mesh.node(j)[0];
                double y = mesh.node(i)[1] - mesh.node(j)[1];
                double amx = a - x;
                double apx = a + x;
                double bmy = b - y;
                double bpy = b + y;
                CI(i, j) = (Ep / (8 * (il::pi))) * (sqrt(pow(amx, 2) + pow(bmy, 2)) / (amx * bmy)+ sqrt(pow(apx, 2) + pow(bmy, 2)) / (apx * bmy) + sqrt(pow(amx, 2) + pow(bpy, 2)) / (amx * bpy) +sqrt(pow(apx, 2) + pow(bpy, 2)) / (apx * bpy));
            }
        }

        return CI;


    }
}


