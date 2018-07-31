

#ifndef INC_HFPX3D_ELAST_KER_INT_H
#define INC_HFPX3D_ELAST_KER_INT_H

#include <complex>
#include "inputE.h"
using namespace std;
namespace hfp3d {

 //In this script, J01,I43_I21,I43_J23,I65_I43_J23,I21 are just functions of integrals which serve to calculate the dislocation
  //  kernel. The only function that is called in AssemblyDDM is stress. The elasticity property and the inclined angle de
    // are also defined in the cpp file. This kernel is only for the transversely isotropic material,if one needs to
    //evaluate the solution for the isotropic material, just define the Lamé coefficients and add a small perturbation showed
    //in the commented code in the cpp file.

    std::complex<double> delta(int p,int q);



    std::complex<double> J01(int p, std::complex<double> gamaj,int k, double l[3],  std::complex<double> V[3],  std::complex<double> W[3],
                             std::complex<double> Vj, std::complex<double> Lj, std::complex<double> RjA, std::complex<double> RjB,
                             std::complex<double> TjA, std::complex<double> TjB);

    std::complex<double> I43_I21(int p,std::complex<double> gamaj, double l[3],  std::complex<double> dA[3],
                                 std::complex<double> dB[3],  std::complex<double> V[3],  std::complex<double> MA[3],
                                 std::complex<double> MB[3],std::complex<double> Vbar,std::complex<double> Vj);

    std::complex<double> I43_J23(int p, int q,  std::complex<double> gamaj, int k, double l[3],  std::complex<double> dA[3],
                                 std::complex<double> dB[3],  std::complex<double> V[3],  std::complex<double> W[3],  std::complex<double> MA[3],
                                 std::complex<double> MB[3], std::complex<double> Vbar, std::complex<double> Vj, std::complex<double> Lbar, std::complex<double> Lj,
                                 std::complex<double> RjA, std::complex<double> RjB,  std::complex<double> TjA, std::complex<double> TjB);

    std::complex<double> I65_I43_J23(int P,int Q, int T,  std::complex<double> gamaj,double l[3], std::complex<double> dA[3],
                                     std::complex<double> dB[3],  std::complex<double> V[3],  std::complex<double> MA[3],
                                     std::complex<double> MB[3], std::complex<double> Vbar, std::complex<double> Vj, std::complex<double> Lbar, std::complex<double> Lj, std::complex<double> RjA,
                                     std::complex<double> RjB,  std::complex<double> TjA, std::complex<double> TjB);

    std::complex<double> I21( std::complex<double> gamaj,int k,double l[3], std::complex<double> dA[3],
                              std::complex<double> dB[3],  std::complex<double> V[3],  std::complex<double> MA[3],
                              std::complex<double> MB[3], std::complex<double> Vbar, std::complex<double> Vj);

    int e(int p,int q,int s);

    double SI(int p,int q,int t, double xA[3], double xB[3], double x[3]);

    double SC(int p,int q,int t, double xA[3],double xB[3],double x[3]);

    double sL(int P,int Q,int T,double xA[3],double xB[3],double x[3], bool FS);

    double strain(int P,int Q,int T,double xA[3],double xB[3],double xC[3],
                           double xD[3],double x[3], bool FS);




    //Q=0, strike-slip fault;  Q=1,dip-slip fault; Q=2, Tensile fracture
    // return the stress component [11,22,33,23,13,12] at x due to a rectangular dislocation loop {xA,xB,xC,xD] in
    //clockwise direction
    void Stress(int Q,double xA[3],double xB[3],double xC[3],
                              double xD[3],double x[3], double Result[6], bool FS);



    //stress at the crack plan due to Q=1,2,3
    void Normal_Shear_Stress(double xA[3],double xB[3],double xC[3],
                                              double xD[3],double x[3], double Result[3][3], bool FS);


}
#endif //INC_HFPX3D_ELAST_KER_INT_H
