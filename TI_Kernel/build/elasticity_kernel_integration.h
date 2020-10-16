//
// Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
// Geo-Energy Laboratory, 2016-2019.  All rights reserved.
// See the LICENSE.TXT file for more details.
//
// Contributors:
// Weihan Zhang
// Fatima-Ezzahra Moukhtari


#ifndef INC_HFPX3D_ELAST_KER_INT_H
#define INC_HFPX3D_ELAST_KER_INT_H

#include <complex>
#include <il/StaticArray.h>
#include <il/StaticArray3D.h>
#include <il/StaticArray4D.h>
#include <il/StaticArray2D.h>

namespace hfp3d {

 //In this script, J01,I43_I21,I43_J23,I65_I43_J23,I21 are just functions of integrals which serve to calculate the dislocation
 //  kernel. The only function that is called in AssemblyDDM is stress. The elasticity property and the inclined angle de
 // are also defined in the cpp file. This kernel is only for the transversely isotropic material,if one needs to
 //evaluate the solution for the isotropic material, just define the Lamé coefficients and add a small perturbation showed
 //in the commented code in the cpp file.



    std::complex<double> delta(int p,int q);



    std::complex<double> J01(int p, std::complex<double> gamaj,int k,il::StaticArray<double,3> l,il::StaticArray<double,3> V,il::StaticArray<double,3> W,
                             std::complex<double> Vj, std::complex<double> Lj, std::complex<double> RjA, std::complex<double> RjB,
                             std::complex<double> TjA, std::complex<double> TjB);

    std::complex<double> I43_I21(int p,std::complex<double> gamaj,il::StaticArray<double,3> l,il::StaticArray<double,3> dA,
                                 il::StaticArray<double,3> dB,il::StaticArray<double,3> V,il::StaticArray<double,3> MA,
                                 il::StaticArray<double,3> MB,std::complex<double> Vbar,std::complex<double> Vj);

    std::complex<double> I43_J23(int p, int q,  std::complex<double> gamaj, int k,il::StaticArray<double,3> l,il::StaticArray<double,3> dA,
                                 il::StaticArray<double,3> dB,il::StaticArray<double,3> V,il::StaticArray<double,3> W,il::StaticArray<double,3> MA,
                                 il::StaticArray<double,3> MB, std::complex<double> Vbar, std::complex<double> Vj, std::complex<double> Lbar, std::complex<double> Lj,
                                 std::complex<double> RjA, std::complex<double> RjB,  std::complex<double> TjA, std::complex<double> TjB);

    std::complex<double> I65_I43_J23(int P,int Q, int T,  std::complex<double> gamaj,il::StaticArray<double,3> l,il::StaticArray<double,3> dA,
                                     il::StaticArray<double,3> dB,il::StaticArray<double,3> V,il::StaticArray<double,3> MA,
                                     il::StaticArray<double,3> MB, std::complex<double> Vbar, std::complex<double> Vj, std::complex<double> Lbar, std::complex<double> Lj, std::complex<double> RjA,
                                     std::complex<double> RjB,  std::complex<double> TjA, std::complex<double> TjB);

    std::complex<double> I21( std::complex<double> gamaj,int k,il::StaticArray<double,3> l,il::StaticArray<double,3> dA,
                              il::StaticArray<double,3> dB,il::StaticArray<double,3> V,il::StaticArray<double,3> MA,
                              il::StaticArray<double,3> MB, std::complex<double> Vbar, std::complex<double> Vj);

    int e(int p,int q,int s);

    double SI(int p,int q,int t, il::StaticArray<double,3> xA,il::StaticArray<double,3> xB,il::StaticArray<double,3> x);

    double sL(int P,int Q,int T,il::StaticArray<double,3> xA,il::StaticArray<double,3> xB,il::StaticArray<double,3> x);

    double strain(int P,int Q,int T, il::StaticArray<double,3> xA,il::StaticArray<double,3> xB,il::StaticArray<double,3> xC,
                  il::StaticArray<double,3> xD,il::StaticArray<double,3> x);




    //Q=0, strike-slip fault;  Q=1,dip-slip fault; Q=2, Tensile fracture
    // return the stress component [11,22,33,23,13,12] at x due to a rectangular dislocation loop {xA,xB,xC,xD] in
    //clockwise direction
    il::StaticArray<double,6> Stress(int Q,il::StaticArray<double,3> xA,il::StaticArray<double,3> xB,il::StaticArray<double,3> xC,
                                     il::StaticArray<double,3> xD,il::StaticArray<double,3> x);



    //stress at the crack plan due to Q=1,2,3
    il::StaticArray2D<double,3,3> Normal_Shear_Stress(il::StaticArray<double,3> xA,il::StaticArray<double,3> xB,il::StaticArray<double,3> xC,
                                                      il::StaticArray<double,3> xD,il::StaticArray<double,3> x);


}
#endif
