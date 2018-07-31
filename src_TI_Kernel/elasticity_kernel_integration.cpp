
#include<iostream>
#include <complex>
#include <cmath>
#include <math.h>
#include "elasticity_kernel_integration.h"
using namespace std;



namespace hfp3d {
    const double pi = 3.141592653589793238463;

    double Error=1e-10;

    void cross(double x[], double y[], double res[]){

        res[0] = ( (x[1]*y[2]) - (x[2]*y[1]));
        res[1] = ( -(x[0]*y[2]) + (x[2]*y[0]));
        res[2] = ( (x[0]*y[1]) - (x[1]*y[0]));
    }

    void cross(std::complex<double> x[], std::complex<double> y[], std::complex<double> res[]){

        res[0] = ( (x[1]*y[2]) - (x[2]*y[1]));
        res[1] = ( -(x[0]*y[2]) + (x[2]*y[0]));
        res[2] = ( (x[0]*y[1]) - (x[1]*y[0]));
    }

    void cross(double x[], std::complex<double> y[], std::complex<double> res[]){

        res[0] = ( (x[1]*y[2]) - (x[2]*y[1]));
        res[1] = ( -(x[0]*y[2]) + (x[2]*y[0]));
        res[2] = ( (x[0]*y[1]) - (x[1]*y[0]));
    }

    void cross(std::complex<double> x[], double y[], std::complex<double> res[]){

        res[0] = ( (x[1]*y[2]) - (x[2]*y[1]));
        res[1] = ( -(x[0]*y[2]) + (x[2]*y[0]));
        res[2] = ( (x[0]*y[1]) - (x[1]*y[0]));

    }

    //delta and phi defined in (Pan,2014), delta is the angle inclined with the isotropic plan. phi is a rotation angle
    // in the isotropic plan which can be considered as 0.

    //phi=0, and if delta=0, dislocation loop in the isotropic plan, delta=pi/2, dislocation loop perpendicular to
    //the isotropic plan
    double de=pi/180*0;
    double phi=pi/180*0;

    ////////////////////////////////////Material property
//    la and mu are the Lamé parameters of the Material
    vector<double> Ce = hfp3d::Cmatrix("TI_parameters.json");
    double c11=Ce[0];
    double c12= Ce[1];
    double c13=Ce[2];
    double c33=Ce[3];
    double c44=Ce[4];
    double c66=(c11-c12)/2;
    double a = c44 * (c13 + c44);
    double b = pow((c13 + c44), 2) + pow(c44, 2) - c11 * c33;
    double c = c44 * (c13 + c44);

    // Coefficients m0 and m1 (defined as m1 and m2 in Pan2014)
    const std::complex<double> m0 = (-b + sqrt(std::complex<double>(pow(b, 2) - 4 * a * c))) / 2. / a;
    const std::complex<double> m1 = 1. / m0;
    const std::complex<double> m[2] = {m0, m1};
    //std::complex<double> m0,m1,mm1;

    //  double m0=(-b+sqrt(pow(b,2)-4*a*c))/2/a;
    // double m1=1/m0;

    const std::complex<double> gama0 = sqrt(std::complex<double>((c44 + m0 * (c13 + c44)) / c11));
    const std::complex<double> gama1 = sqrt(std::complex<double>((c44 + m1 * (c13 + c44)) / c11));
    const std::complex<double> gama2 = sqrt(c44 / c66);
    std::complex<double> gama[3] = {gama0, gama1, gama2};
    const std::complex<double> theta = c11 * (gama0 + gama1) / (c13 + c44);
    const std::complex<double> f0 = m1 / (m0 - m1);
    const std::complex<double> f1 = m0 / (m1 - m0);
    const double f2 = 1;
    std::complex<double> f[3] = {f0, f1, f2};
    const std::complex<double> g0 = (m1 + 1.) / (m0 - m1);
    const std::complex<double> g1 = (m0 + 1.) / (m1 - m0);
    const std::complex<double> g2 = 1;
    std::complex<double> g[3] = {g0, g1, g2};
    const complex<double> FF = theta*(gama[0]+gama[1])/(m[0]+m[1]+2.)-1.;
    std::complex<double> F[2][2] = {{(-1.) * (m[1] + 1.) * (m[1] + 1.) * (gama[1] + gama[0])
                                     / theta / (m[0] + m[1] + 2.) / pow((gama[0] - gama[1]), 2), 1.*(m[1] + 1.) * (m[0] + 1.) * (gama[1] + gama[1])
                                                                                                 / theta / (m[0] + m[1] + 2.) / pow((gama[0] - gama[1]), 2)},{1.* (m[0] + 1.) * (m[1] + 1.) * (gama[0] + gama[0])
                                                                                                                                                              / theta / (m[0] + m[1] + 2.) / pow((gama[0] - gama[1]), 2),(-1.) * (m[0] + 1.) * (m[0] + 1.) * (gama[0] + gama[1])
                                                                                                                                                                                                                         / theta / (m[0] + m[1] + 2.) / pow((gama[0] - gama[1]), 2)}};
    std::complex<double> G[2][2]= {{(m[0] + 1.) * F[0][0],(m[0] + 1.) * F[0][1]},{(m[1] + 1.) * F[1][0],(m[1] + 1.) * F[1][1]}};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // function coding the Kroenecker symbol
    std::complex<double> delta(int p,int q){
        std::complex<double> Result;
        if(p==q){
            Result=1;
        }
        else{
            Result=0;
        }
        return Result;
    }


////////////////////////////////////////////////////////////////////////////////////

    std::complex<double> J01(int p, std::complex<double> gamaj,int k, double l[3], std::complex<double> V[3], std::complex<double> W[3],
                             std::complex<double> Vj, std::complex<double> Lj, std::complex<double> RjA, std::complex<double> RjB,
                             std::complex<double> TjA, std::complex<double> TjB){
        std::complex<double> Result;
        if(p==0 || p==1){
            Result = l[k]/pow(Lj,2)*(-l[p]*(1./RjB-1./RjA)
                                     -pow((-1),p+1)*(l[1-p]*V[2]-l[2]*V[1-p]/pow(gamaj,2))/pow(Vj,2)*(TjB/RjB-TjA/RjA));
        }
        else  {
            Result = l[k] / pow(Lj,2) * (-l[p] * (1. / RjB - 1. / RjA) + W[2] / pow(Vj,2) * (TjB / RjB - TjA / RjA));
        }


        return Result;
    }
    ////////////////////////////////////////////////////////////////////////////////////////
    std::complex<double> I43_I21(int p,std::complex<double> gamaj, double l[3], std::complex<double> dA[3],
                                 std::complex<double> dB[3], std::complex<double> V[3], std::complex<double> MA[3],
                                 std::complex<double> MB[3],std::complex<double> Vbar,std::complex<double> Vj){
        std::complex<double> Result;
        if ((abs(Vbar)<=Error) && (abs(V[2])>Error)) {
            Result = 0.;
        }
        else if((abs(V[2]) <= Error) && (abs(Vbar) > Error) && (abs(MA[2]) <= Error) && (abs(MB[2]) > Error)){
            Result = l[2] / Vbar * pow((-1),p+1) * V[1- p] / Vj * ((dB[2] * Vj * sqrt(pow((dB[2] * Vj),2)
                                                                                      + pow(MB[2],2)) / (pow((dB[2] * V[2]),2) + pow(MB[2],2)))
                                                                   -(dA[2] * Vj * sqrt(pow((dA[2] * Vj),2) + pow(MA[2],2)) / (pow((dA[2] * V[2]),2) + pow(MA[2],2))));
        }
        else if((abs(V[2]) <= Error) && (abs(Vbar) > Error) && (abs(MA[2]) > Error) && (abs(MB[2]) <= Error)){
            Result = l[2] / Vbar * pow((-1),p+1) * V[1- p] / Vj * ((dB[2] * Vj * sqrt(pow((dB[2] * Vj),2)
                                                                                      + pow(MB[2],2)) / (pow((dB[2] * V[2]),2) + pow(MB[2],2)))
                                                                   -(dA[2] * Vj * sqrt(pow((dA[2] * Vj),2) + pow(MA[2],2)) / (pow((dA[2] * V[2]),2) + pow(MA[2],2))));
        }
        else if((abs(V[2]) <= Error) && (abs(Vbar) > Error) && (abs(MA[2]) > Error) && (abs(MB[2]) > Error)){
            Result = l[2] / Vbar * pow((-1),p+1) * V[1- p] / Vj * ((dB[2] * Vj * sqrt(pow((dB[2] * Vj),2)
                                                                                      + pow(MB[2],2)) / (pow((dB[2] * V[2]),2) + pow(MB[2],2)))
                                                                   -(dA[2] * Vj * sqrt(pow((dA[2] * Vj),2) + pow(MA[2],2)) / (pow((dA[2] * V[2]),2) + pow(MA[2],2))));
        }
        else {
            Result = l[2] / Vbar * (-V[p] / gamaj * Vbar / pow(V[2],2)* (atan(MB[2] * Vbar / gamaj / V[2]
                                                                              / sqrt(pow((dB[2] * Vj),2) + pow(MB[2],2)))
                                                                         -atan(MA[2] * Vbar / gamaj / V[2] / sqrt(pow((dA[2] * Vj),2) + pow(MA[2],2))))
                                    -V[p] / V[2] * ((MB[2] * sqrt(pow((dB[2] * Vj),2) + pow(MB[2],2)) / (pow((dB[2] * V[2]),2) + pow(MB[2],2)))
                                                    -(MA[2]*sqrt(pow((dA[2] * Vj),2) + pow(MA[2],2)) / (pow((dA[2] * V[2]),2) + pow(MA[2],2))))
                                    +pow((-1),p+1)*V[1-p]/Vj*((dB[2]*Vj*sqrt(pow((dB[2]*Vj),2)+ pow(MB[2],2))/(pow((dB[2]*V[2]),2)+pow(MB[2],2)))
                                                              -(dA[2]* Vj* sqrt(pow((dA[2] * Vj),2) + pow(MA[2],2)) / (pow((dA[2] * V[2]),2) + pow(MA[2],2)))));


        }
        return Result;
    }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::complex<double> I43_J23(int p, int q,  std::complex<double> gamaj, int k, double l[3], std::complex<double> dA[3],
                                 std::complex<double> dB[3],std::complex<double> V[3],std::complex<double> W[3],std::complex<double> MA[3],
                                 std::complex<double> MB[3], std::complex<double> Vbar, std::complex<double> Vj, std::complex<double> Lbar, std::complex<double> Lj,
                                 std::complex<double> RjA, std::complex<double> RjB,  std::complex<double> TjA, std::complex<double> TjB){
        std::complex<double> Result,Cjpq,RbarA;
        if (p==q) {
            Cjpq = pow((-1.),p+1) * (2. * pow(Lj,2) * l[0] * l[1] * l[2] * V[2] - pow((-1.),p+1) * pow(l[p],4)* W[2]
                                     - l[0] * l[1] * (pow(l[p],2) * l[2] * V[2] + pow(Lbar,2) * l[p] * V[p]));
        }
        else {
            Cjpq = pow(Lj,2) * (pow(l[0],2) - pow(l[1],2)) * l[2] * V[2] - pow(Lbar,2) * l[0] * l[1] * W[2];
        }
        RbarA = sqrt(pow(dA[0],2)+pow(dA[1],2)) ;

        if ((abs(Lbar)<=Error) && (abs(RbarA)>Error)) {
            Result = delta(k, 2)*pow(gamaj,2)*dA[p]*dA[q]/pow(RbarA,2)*(2. /pow(RbarA,2)*(RjB - RjA)-(1./RjB-1./RjA));
        }
        else if ((abs(Vbar)<=Error) && (abs(V[2])>Error)) {
            Result = 0;
        }
        else if ((abs(Lbar)>Error) && (abs(V[2])<=Error) && (abs(Vbar)>Error) && (abs(MA[2])<=Error) && (abs(MB[2])>Error)) {
            Result = -l[k] * l[p] * l[q] * l[2] / pow(Lbar,2) / pow(Lj,2) * (1. / RjB - 1. / RjA) - l[k] * Cjpq / pow(Lbar,4) / pow(Lj,2)/
                                                                                                    pow(Vj,2) * (TjB / RjB - TjA / RjA) -delta(p, q) * l[k] * gamaj *
                                                                                                                                         ((MB[2] * sqrt(pow((dB[2] * Vj),2) + pow(MB[2],2)) / (pow((dB[2] * V[2]),2) + pow(MB[2],2)))) * gamaj / Vbar
                     +l[k] * pow(gamaj,2) / pow(Vbar,3) * (V[p] * V[q] - pow((-1),p+q) * V[1 - p] * V[1 - q]) *
                      ((MB[2] * sqrt(pow((dB[2] * Vj),2) + pow(MB[2],2)) /
                        (pow((dB[2] * V[2]),2) + pow(MB[2],2))));
        }
        else if ((abs(Lbar)>Error) && (abs(V[2])<=Error) && (abs(Vbar)>Error) && (abs(MA[2])>Error) && (abs(MB[2])<=Error)) {


            Result = -l[k] * l[p] * l[q] * l[2] / pow(Lbar,2) / pow(Lj,2) * (1. / RjB - 1. / RjA) - l[k] * Cjpq / pow(Lbar,4) / pow(Lj,2)/
                                                                                                    pow(Vj,2) * (TjB / RjB - TjA / RjA) -delta(p, q) * l[k] * gamaj *
                                                                                                                                         (-(MA[2]* sqrt(pow((dA[2] * Vj),2) + pow(MA[2],2)) / (pow((dA[2] * V[2]),2) + pow(MA[2],2)))) * gamaj / Vbar
                     +l[k] * pow(gamaj,2) / pow(Vbar,3) * (V[p] * V[q] - pow((-1),p+q) * V[1 - p] * V[1 - q]) *
                      (-(MA[2] * sqrt(pow((dA[2] * Vj),2) + pow(MA[2],2)) /
                         (pow((dA[2] * V[2]),2) + pow(MA[2],2))));
        }
        else if ((abs(Lbar)>Error) && (abs(V[2])<=Error) && (abs(Vbar)>Error) && (abs(MA[2])>Error) && (abs(MB[2])>Error)) {
            Result = -l[k] * l[p] * l[q] * l[2] / pow(Lbar,2) / pow(Lj,2) * (1. / RjB - 1. / RjA) - l[k] * Cjpq / pow(Lbar,4) / pow(Lj,2)/
                                                                                                    pow(Vj,2) * (TjB / RjB - TjA / RjA) -delta(p, q) * l[k] * gamaj *
                                                                                                                                         ((MB[2]* sqrt(pow((dB[2] * Vj),2) + pow(MB[2],2)) / (pow((dB[2] * V[2]),2) + pow(MB[2],2)))
                                                                                                                                          -(MA[2] * sqrt(pow((dA[2] * Vj),2) + pow(MA[2],2)) / (pow((dA[2] * V[2]),2) + pow(MA[2],2))))*gamaj / Vbar
                     -l[k] * V[2] * pow(gamaj,2) / pow(Vbar,2) / Vj * (pow((-1),p+1) * V[1 - p] * V[q] + pow((-1),q+1) * V[p] * V[1- q]) *
                      ((dB[2] * Vj * sqrt(pow((dB[2] * Vj),2) + pow(MB[2],2)) / (pow((dB[2] * V[2]),2) + pow(MB[2],2)))
                       -(dA[2] * Vj * sqrt(pow((dA[2] * Vj),2) + pow(MA[2],2)) / (pow((dA[2] * V[2]),2) + pow(MA[2],2))))
                     +l[k] * pow(gamaj,2) / pow(Vbar,3) * (V[p] * V[q] - pow((-1),p+q) * V[1 - p] * V[1 - q]) *
                      ((MB[2] * sqrt(pow((dB[2] * Vj),2) + pow(MB[2],2)) /
                        (pow((dB[2] * V[2]),2) + pow(MB[2],2)))
                       -(MA[2] * sqrt(pow((dA[2] * Vj),2) + pow(MA[2],2)) / (pow((dA[2] * V[2]),2) + pow(MA[2],2))))
                     +l[k] * V[2] / Vbar / Vj *
                      (pow(l[2],2) / pow(Lbar,4) * (pow((-1),p+1) * l[1 - p] * l[q] + pow((-1),q+1) * l[p] * l[1 - q]) + pow(gamaj,2) / pow(Vbar,2)*
                                                                                                                         (pow((-1),p+1) * V[1 - p] * V[q] + pow((-1),q+1) * V[p] * V[1 - q]))
                      *((dB[2] * Vj / sqrt(pow((dB[2] * Vj),2) + pow(MB[2],2))) - (dA[2] * Vj / sqrt(pow((dA[2] * Vj),2) + pow(MA[2],2))))
                     +l[k] * V[2] / Vbar / pow(Vj,2) * (l[2] / pow(Lbar,4) * (pow((-1),p+q) * pow(l[p],2)* (l[p] * V[q] - 3. * l[1 - q] * V[1 - p])
                                                                              - pow(l[1 - p],2)* (l[1 - p] * V[1 - q] - 3. * l[q] * V[p]))
                                                        -V[2] * pow(gamaj,2) / pow(Vbar,2) * (V[p] * V[q] - pow((-1),p+q) * V[1 - p] * V[1 - q]))*(
                              (MB[2] / sqrt(pow((dB[2] * Vj),2) + pow(MB[2],2))) - (MA[2] / sqrt(pow((dA[2] * Vj),2) + pow(MA[2],2))));
        }
        else {
            Result = -l[k] * l[p] * l[q] * l[2] / pow(Lbar,2) / pow(Lj,2) * (1. / RjB - 1. / RjA) - l[k] * Cjpq / pow(Lbar,4) /
                                                                                                    pow(Lj,2) / pow(Vj,2) * (TjB / RjB - TjA / RjA) +delta(p, q) * l[k] * gamaj / V[2] *
                                                                                                                                                     (atan(MB[2] * Vbar / gamaj / V[2] / sqrt(pow((dB[2] * Vj),2) + pow(MB[2],2)))
                                                                                                                                                      -atan(MA[2] * Vbar / gamaj / V[2] / sqrt(pow((dA[2] * Vj),2) + pow(MA[2],2))))
                     -l[k] * V[2] * pow(gamaj,2) / pow(Vbar,3) / Vj * (pow((-1),p+1) * V[1 - p] * V[q] + pow((-1),q+1) * V[p] * V[1 - q]) *
                      ((dB[2] * Vj * sqrt(pow((dB[2] * Vj),2) + pow(MB[2],2)) /
                        (pow((dB[2] * V[2]),2) + pow(MB[2],2)))
                       -(dA[2] * Vj * sqrt(pow((dA[2] * Vj),2) + pow(MA[2],2)) / (pow((dA[2] * V[2]),2) + pow(MA[2],2))))
                     +l[k] * pow(gamaj,2) / pow(Vbar,3) * (V[p] * V[q] - pow((-1),p+q)* V[1 - p] * V[1 - q]) *
                      ((MB[2] * sqrt(pow((dB[2] * Vj),2) + pow(MB[2],2)) /
                        (pow((dB[2] * V[2]),2) + pow(MB[2],2)))
                       -(MA[2] * sqrt(pow((dA[2] * Vj),2) + pow(MA[2],2)) / (pow((dA[2] * V[2]),2) + pow(MA[2],2))))
                     +l[k] * V[2] / Vbar / Vj *
                      (pow(l[2],2) / pow(Lbar,4) * (pow((-1),p+1) * l[1 - p] * l[q] + pow((-1),q+1) * l[p] * l[1 - q]) + pow(gamaj,2) /
                                                                                                                         pow(Vbar,2) * (pow((-1),p+1) * V[1 - p] * V[q] + pow((-1),q+1) * V[p] * V[1 - q]))
                      *((dB[2] * Vj / sqrt(pow((dB[2] * Vj),2) + pow(MB[2],2))) - (dA[2] * Vj / sqrt(pow((dA[2] * Vj),2) + pow(MA[2],2))))
                     +l[k] * V[2] / Vbar / pow(Vj,2) * (l[2] / pow(Lbar,4) * (pow((-1),p+q) * pow(l[p],2) *
                                                                              (l[p] * V[q] - 3. * l[1 - q] * V[1 - p]) - pow(l[1 - p],2) * (l[1 - p] * V[1 - q] - 3. * l[q] * V[p]))
                                                        -V[2] * pow(gamaj,2) / pow(Vbar,2) * (V[p] * V[q] - pow((-1),p+q) * V[1 - p] * V[1 - q]))*(
                              (MB[2] / sqrt(pow((dB[2] * Vj),2) + pow(MB[2],2))) - (MA[2] / sqrt(pow((dA[2] * Vj),2) + pow(MA[2],2))));
        }

        return Result;
    }
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::complex<double> I65_I43_J23(int P,int Q, int T,  std::complex<double> gamaj,double l[3],std::complex<double> dA[3],
                                     std::complex<double> dB[3], std::complex<double> V[3], std::complex<double> MA[3],
                                     std::complex<double> MB[3], std::complex<double> Vbar, std::complex<double> Vj, std::complex<double> Lbar, std::complex<double> Lj, std::complex<double> RjA,
                                     std::complex<double> RjB,  std::complex<double> TjA, std::complex<double> TjB){

        int p,q;
        std::complex<double> Result,RbarA;
        if (P==Q) {
            p = P;
            q = T;
        }
        else if (P==T) {
            p = P;
            q = Q;
        }
        else if (Q==T) {
            p = Q;
            q = P;
        }

        RbarA = sqrt(pow(dA[0],2)+pow(dA[1],2)) ;
        if ((abs(Lbar)<=Error) && (abs(RbarA)>Error)) {
            Result= pow(dA[p],2)*dA[q]/pow(RbarA,4)*(4./pow(RbarA,2)*(dB[2]*RjB-dA[2]*RjA)-(dB[2]/RjB-dA[2]/RjA));
        }
        else if ((abs(Vbar)<=Error) && (abs(V[2])>Error)) {
            Result = 0;
        }
        else if ((abs(Lbar)>Error) && (abs(V[2])<=Error) && (abs(Vbar)>Error) && (abs(MA[2])<=Error) && (abs(MB[2])>Error)){
            Result = l[2]*pow(l[p],2)*l[q]/pow(Lbar,2)/pow(Lj,2)*(1./RjB-1./RjA)
                     +l[2]/pow(Lbar,4)/pow(Lj,2)/pow(Vj,2)*pow((-1),p+1)*l[p]*((2*l[q]*l[1-p]+pow((-1),p+q)*l[1-q]*l[p])*pow(Lbar,2)*V[2]
                                                                               +l[2]/pow(gamaj,2)*((delta(p,q)+1.)*l[q]*l[1-p]*l[2]*V[2]-pow((-1),p+q)*pow(l[p],3)*V[1-q]-pow(l[1-p],2)*l[q]*V[1-p]))*(TjB/RjB-TjA/RjA)
                     +l[2]*V[1-q]/pow(Vbar,3)/Vj*(4.*pow((-1.),q+1)*pow(V[1-p],2)+pow((-1.),p+1)*pow(V[2],2)*(pow(V[1-q],2)-3.*pow(V[q],2))*pow(gamaj,2)/pow(Vbar,2))
                      *((dB[2]*Vj*sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2))/(pow((dB[2]*V[2]),2)+pow(MB[2],2)))-(dA[2]*Vj*sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))/(pow((dA[2]*V[2]),2)+pow(MA[2],2))))
                     +l[2]*V[q]/Vbar*(3.*pow((-1.),p+q)*V[2]*(pow(V[q],2)-3.*pow(V[1-q],2))*pow(gamaj,2)/pow(Vbar,4))
                      *((MB[2]*sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2))/(pow((dB[2]*V[2]),2)+pow(MB[2],2)))-(MA[2]*sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))/(pow((dA[2]*V[2]),2)+pow(MA[2],2))))
                     -pow((-1.),p+1)*2.*l[2]*pow(V[2],2)*V[1-q]*(pow(V[1-q],2)-3.*pow(V[q],2))/pow(Vbar,3)/pow(Vj,3)*((pow((dB[2]*Vj),3)*sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2))/pow((pow((dB[2]*V[2]),2)+pow(MB[2],2)),2))
                                                                                                                      -(pow((dA[2]*Vj),3)*sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))/pow((pow((dA[2]*V[2]),2)+pow(MA[2],2)),2)))
                     -pow((-1.),p+q)*2.*l[2]*V[2]*V[q]*(pow(V[q],2)-3.*pow(V[1-q],2))*pow(gamaj,2)/pow(Vbar,5)*((MB[2]*pow((sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2))),3)/pow((pow((dB[2]*V[2]),2)+pow(MB[2],2)),2))
                                                                                                                -(MA[2]*pow((sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))),3)/pow((pow((dA[2]*V[2]),2)+pow(MA[2],2)),2)))
                     -pow((-1.),p+1)*l[2]*pow(V[2],2)/pow(Vbar,3)/Vj*(V[1-q]*(pow(V[1-q],2)-3.*pow(V[q],2))*pow(gamaj,2)/pow(Vbar,2)-pow(l[2],2)/pow(Lbar,4)*(pow((-1.),q+1)*V[1-q]*(pow(l[1],2)-pow(l[0],2))+2*l[0]*l[1]*V[q]))
                      *((dB[2]*Vj/sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2)))-(dA[2]*Vj/sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))))
                     -pow((-1.),p+q)*l[2]*pow(V[2],2)/pow(Vbar,3)/pow(Vj,2)*(V[2]*V[q]*(pow(V[q],2)-3.*pow(V[1-q],2))*pow(gamaj,2)/pow(Vbar,2)
                                                                             +l[2]/pow(Lbar,4)*(pow(l[q],3)*(2.*pow(V[1-q],2)-pow(V[q],2))+3.*l[1-q]*(pow(Lbar,2)*V[0]*V[1]-l[0]*l[1]*pow(V[q],2))))
                      *((MB[2]/sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2)))-(MA[2]/sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2)))) ;

        }
        else if ((abs(Lbar)>Error) && (abs(V[2])<=Error) && (abs(Vbar)>Error) && (abs(MA[2])>Error) && (abs(MB[2])<=Error)){
            Result = l[2]*pow(l[p],2)*l[q]/pow(Lbar,2)/pow(Lj,2)*(1./RjB-1./RjA)
                     +l[2]/pow(Lbar,4)/pow(Lj,2)/pow(Vj,2)*pow((-1.),p+1)*l[p]*((2.*l[q]*l[1-p]+pow((-1.),p+q)*l[1-q]*l[p])*pow(Lbar,2)*V[2]
                                                                                +l[2]/pow(gamaj,2)*((delta(p,q)+1.)*l[q]*l[1-p]*l[2]*V[2]-pow((-1.),p+q)*pow(l[p],3)*V[1-q]-pow(l[1-p],2)*l[q]*V[1-p]))*(TjB/RjB-TjA/RjA)
                     +l[2]*V[1-q]/pow(Vbar,3)/Vj*(4.*pow((-1.),q+1)*pow(V[1-p],2)+pow((-1.),p+1)*pow(V[2],2)*(pow(V[1-q],2)-3.*pow(V[q],2))*pow(gamaj,2)/pow(Vbar,2))
                      *((dB[2]*Vj*sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2))/(pow((dB[2]*V[2]),2)+pow(MB[2],2)))-(dA[2]*Vj*sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))/(pow((dA[2]*V[2]),2)+pow(MA[2],2))))
                     +l[2]*V[q]/Vbar*(3.*pow((-1),p+q)*V[2]*(pow(V[q],2)-3.*pow(V[1-q],2))*pow(gamaj,2)/pow(Vbar,4))
                      *((MB[2]*sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2))/(pow((dB[2]*V[2]),2)+pow(MB[2],2)))-(MA[2]*sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))/(pow((dA[2]*V[2]),2)+pow(MA[2],2))))
                     -pow((-1.),p+1)*2*l[2]*pow(V[2],2)*V[1-q]*(pow(V[1-q],2)-3.*pow(V[q],2))/pow(Vbar,3)/pow(Vj,3)*((pow((dB[2]*Vj),3)*sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2))/pow((pow((dB[2]*V[2]),2)+pow(MB[2],2)),2))
                                                                                                                     -(pow((dA[2]*Vj),3)*sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))/pow((pow((dA[2]*V[2]),2)+pow(MA[2],2)),2)))
                     -pow((-1.),p+q)*2.*l[2]*V[2]*V[q]*(pow(V[q],2)-3.*pow(V[1-q],2))*pow(gamaj,2)/pow(Vbar,5)*((MB[2]*pow((sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2))),3)/pow((pow((dB[2]*V[2]),2)+pow(MB[2],2)),2))
                                                                                                                -(MA[2]*pow((sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))),3)/pow((pow((dA[2]*V[2]),2)+pow(MA[2],2)),2)))
                     -pow((-1.),p+1)*l[2]*pow(V[2],2)/pow(Vbar,3)/Vj*(V[1-q]*(pow(V[1-q],2)-3.*pow(V[q],2))*pow(gamaj,2)/pow(Vbar,2)-pow(l[2],2)/pow(Lbar,4)*(pow((-1.),q+1)*V[1-q]*(pow(l[1],2)-pow(l[0],2))+2*l[0]*l[1]*V[q]))
                      *((dB[2]*Vj/sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2)))-(dA[2]*Vj/sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))))
                     -pow((-1.),p+q)*l[2]*pow(V[2],2)/pow(Vbar,3)/pow(Vj,2)*(V[2]*V[q]*(pow(V[q],2)-3.*pow(V[1-q],2))*pow(gamaj,2)/pow(Vbar,2)
                                                                             +l[2]/pow(Lbar,4)*(pow(l[q],3)*(2.*pow(V[1-q],2)-pow(V[q],2))+3.*l[1-q]*(pow(Lbar,2)*V[0]*V[1]-l[0]*l[1]*pow(V[q],2))))
                      *((MB[2]/sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2)))-(MA[2]/sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2)))) ;

        }
        else if ((abs(Lbar)>Error) && (abs(V[2])<=Error) && (abs(Vbar)>Error) && (abs(MA[2])>Error) && (abs(MB[2])>Error)){


            Result = l[2]*pow(l[p],2)*l[q]/pow(Lbar,2)/pow(Lj,2)*(1./RjB-1./RjA)
                     +l[2]/pow(Lbar,4)/pow(Lj,2)/pow(Vj,2)*pow((-1.),p+1)*l[p]*((2.*l[q]*l[1-p]+pow((-1),p+q)*l[1-q]*l[p])*pow(Lbar,2)*V[2]
                                                                                +l[2]/pow(gamaj,2)*((delta(p,q)+1.)*l[q]*l[1-p]*l[2]*V[2]-pow((-1.),p+q)*pow(l[p],3)*V[1-q]-pow(l[1-p],2)*l[q]*V[1-p]))*(TjB/RjB-TjA/RjA)
                     +l[2]*V[1-q]/pow(Vbar,3)/Vj*(4.*pow((-1),q+1)*pow(V[1-p],2)+pow((-1),p+1)*pow(V[2],2)*(pow(V[1-q],2)-3.*pow(V[q],2))*pow(gamaj,2)/pow(Vbar,2))
                      *((dB[2]*Vj*sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2))/(pow((dB[2]*V[2]),2)+pow(MB[2],2)))-(dA[2]*Vj*sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))/(pow((dA[2]*V[2]),2)+pow(MA[2],2))))
                     +l[2]*V[q]/Vbar*(3.*pow((-1),p+q)*V[2]*(pow(V[q],2)-3.*pow(V[1-q],2))*pow(gamaj,2)/pow(Vbar,4))
                      *((MB[2]*sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2))/(pow((dB[2]*V[2]),2)+pow(MB[2],2)))-(MA[2]*sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))/(pow((dA[2]*V[2]),2)+pow(MA[2],2))))
                     -pow((-1.),p+1)*2.*l[2]*pow(V[2],2)*V[1-q]*(pow(V[1-q],2)-3.*pow(V[q],2))/pow(Vbar,3)/pow(Vj,3)*((pow((dB[2]*Vj),3)*sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2))/pow((pow((dB[2]*V[2]),2)+pow(MB[2],2)),2))
                                                                                                                      -(pow((dA[2]*Vj),3)*sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))/pow((pow((dA[2]*V[2]),2)+pow(MA[2],2)),2)))
                     -pow((-1.),p+q)*2.*l[2]*V[2]*V[q]*(pow(V[q],2)-3.*pow(V[1-q],2))*pow(gamaj,2)/pow(Vbar,5)*((MB[2]*pow((sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2))),3)/pow((pow((dB[2]*V[2]),2)+pow(MB[2],2)),2))
                                                                                                                -(MA[2]*pow((sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))),3)/pow((pow((dA[2]*V[2]),2)+pow(MA[2],2)),2)))
                     -pow((-1.),p+1)*l[2]*pow(V[2],2)/pow(Vbar,3)/Vj*(V[1-q]*(pow(V[1-q],2)-3.*pow(V[q],2))*pow(gamaj,2)/pow(Vbar,2)-pow(l[2],2)/pow(Lbar,4)*(pow((-1.),q+1)*V[1-q]*(pow(l[1],2)-pow(l[0],2))+2*l[0]*l[1]*V[q]))
                      *((dB[2]*Vj/sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2)))-(dA[2]*Vj/sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))))
                     -pow((-1.),p+q)*l[2]*pow(V[2],2)/pow(Vbar,3)/pow(Vj,2)*(V[2]*V[q]*(pow(V[q],2)-3.*pow(V[1-q],2))*pow(gamaj,2)/pow(Vbar,2)
                                                                             +l[2]/pow(Lbar,4)*(pow(l[q],3)*(2.*pow(V[1-q],2)-pow(V[q],2))+3.*l[1-q]*(pow(Lbar,2)*V[0]*V[1]-l[0]*l[1]*pow(V[q],2))))
                      *((MB[2]/sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2)))-(MA[2]/sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2)))) ;
        }
        else{
            Result = l[2]*pow(l[p],2)*l[q]/pow(Lbar,2)/pow(Lj,2)*(1./RjB-1./RjA)
                     +l[2]/pow(Lbar,4)/pow(Lj,2)/pow(Vj,2)*pow((-1.),p+1)*l[p]*((2*l[q]*l[1-p]+pow((-1),p+q)*l[1-q]*l[p])*pow(Lbar,2)*V[2]
                                                                                +l[2]/pow(gamaj,2)*((delta(p,q)+1.)*l[q]*l[1-p]*l[2]*V[2]-pow((-1.),p+q)*pow(l[p],3)*V[1-q]-pow(l[1-p],2)*l[q]*V[1-p]))*(TjB/RjB-TjA/RjA)
                     -(2.*delta(p,q)+1.)*l[2]*V[q]/gamaj/pow(V[2],2)*(atan(MB[2]*Vbar/gamaj/V[2]/sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2)))
                                                                      -atan(MA[2]*Vbar/gamaj/V[2]/sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))))
                     +l[2]*V[1-q]/pow(Vbar,3)/Vj*(4.*pow((-1),q+1)*pow(V[1-p],2)+pow((-1),p+1)*pow(V[2],2)*(pow(V[1-q],2)-3.*pow(V[q],2))*pow(gamaj,2)/pow(Vbar,2))
                      *((dB[2]*Vj*sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2))/(pow((dB[2]*V[2]),2)+pow(MB[2],2)))-(dA[2]*Vj*sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))/(pow((dA[2]*V[2]),2)+pow(MA[2],2))))
                     +l[2]*V[q]/Vbar*(-(2.*delta(p,q)+1.)/V[2]+3.*pow((-1),p+q)*V[2]*(pow(V[q],2)-3.*pow(V[1-q],2))*pow(gamaj,2)/pow(Vbar,4))
                      *((MB[2]*sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2))/(pow((dB[2]*V[2]),2)+pow(MB[2],2)))-(MA[2]*sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))/(pow((dA[2]*V[2]),2)+pow(MA[2],2))))
                     -pow((-1),p+1)*2.*l[2]*pow(V[2],2)*V[1-q]*(pow(V[1-q],2)-3.*pow(V[q],2))/pow(Vbar,3)/pow(Vj,3)*((pow((dB[2]*Vj),3)*sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2))/pow((pow((dB[2]*V[2]),2)+pow(MB[2],2)),2))
                                                                                                                     -(pow((dA[2]*Vj),3)*sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))/pow((pow((dA[2]*V[2]),2)+pow(MA[2],2)),2)))
                     -pow((-1),p+q)*2.*l[2]*V[2]*V[q]*(pow(V[q],2)-3.*pow(V[1-q],2))*pow(gamaj,2)/pow(Vbar,5)*((MB[2]*pow((sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2))),3)/pow((pow((dB[2]*V[2]),2)+pow(MB[2],2)),2))
                                                                                                               -(MA[2]*pow((sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))),3)/pow((pow((dA[2]*V[2]),2)+pow(MA[2],2)),2)))
                     -pow((-1),p+1)*l[2]*pow(V[2],2)/pow(Vbar,3)/Vj*(V[1-q]*(pow(V[1-q],2)-3.*pow(V[q],2))*pow(gamaj,2)/pow(Vbar,2)-pow(l[2],2)/pow(Lbar,4)*(pow((-1),q+1)*V[1-q]*(pow(l[1],2)-pow(l[0],2))+2*l[0]*l[1]*V[q]))
                      *((dB[2]*Vj/sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2)))-(dA[2]*Vj/sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))))
                     -pow((-1),p+q)*l[2]*pow(V[2],2)/pow(Vbar,3)/pow(Vj,2)*(V[2]*V[q]*(pow(V[q],2)-3.*pow(V[1-q],2))*pow(gamaj,2)/pow(Vbar,2)
                                                                            +l[2]/pow(Lbar,4)*(pow(l[q],3)*(2.*pow(V[1-q],2)-pow(V[q],2))+3.*l[1-q]*(pow(Lbar,2)*V[0]*V[1]-l[0]*l[1]*pow(V[q],2))))
                      *((MB[2]/sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2)))-(MA[2]/sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2)))) ;
        }
        return Result;

    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::complex<double> I21( std::complex<double> gamaj,int k,double l[3],std::complex<double> dA[3],
                              std::complex<double> dB[3], std::complex<double> V[3], std::complex<double> MA[3],
                              std::complex<double> MB[3], std::complex<double> Vbar, std::complex<double> Vj){
        std::complex<double> Result;
        if ((abs(Vbar)<=Error) && (abs(V[2])>Error)) {
            Result = 0;
        }
        else if((abs(V[2]) <= Error) && (abs(Vbar) > Error) && (abs(MA[2]) <= Error) && (abs(MB[2]) > Error)) {
            Result = -l[k] * gamaj * ((MB[2] * sqrt(pow((dB[2] * Vj),2) + pow(MB[2],2)) / (pow((dB[2] * V[2]),2)  + pow(MB[2],2)))) *
                     gamaj / Vbar;
        }
        else if((abs(V[2]) <= Error) && (abs(Vbar) > Error) && (abs(MA[2]) > Error) && (abs(MB[2]) <= Error)) {
            Result = -l[k] * gamaj * (-(MA[2] * sqrt(pow((dA[2] * Vj),2)  + pow(MA[2],2)) / (pow((dA[2] * V[2]),2) + pow(MA[2],2)))) *
                     gamaj / Vbar;
        }
        else if((abs(V[2]) <= Error) && (abs(Vbar) > Error) && (abs(MA[2]) > Error) && (abs(MB[2]) > Error)){
            Result = -l[k] * gamaj * ((MB[2] * sqrt(pow((dB[2] * Vj),2)  + pow(MB[2],2)) / (pow((dB[2] * V[2]),2)  + pow(MB[2],2)))
                                      -(MA[2] * sqrt(pow((dA[2] * Vj),2)  + pow(MA[2],2)) / (pow((dA[2] * V[2]),2) + pow(MA[2],2))))*gamaj / Vbar;
        }
        else {
            Result = l[k] * gamaj / V[2] * (atan(MB[2] * Vbar / gamaj / V[2] / sqrt(pow((dB[2] * Vj),2) + pow(MB[2],2)))
                                            -atan(MA[2] * Vbar / gamaj / V[2] / sqrt(pow((dA[2] * Vj),2) + pow(MA[2],2))));
        }
        return Result;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::complex<double> L64_L42(int P, int Q, int T, double l[3], std::complex<double> dA[3], std::complex<double> dB[3], std::complex<double> V[3], std::complex<double> W[3],
                                 std::complex<double> Lbar, std::complex<double> RbarA, std::complex<double> RbarB, std::complex<double> TbarA, std::complex<double> TbarB){
        int p,q;
        std::complex<double> result;
        if (P == Q) {
            p = P;
            q = T;
        }
        else if (P == T){
            p = P;
            q = Q;
        }
        else if (Q == T){
            p = Q;
            q = P;
        }

        if ((abs(Lbar)<=Error) && (abs(RbarA)>Error)){
            result = dA[q]/2./pow(RbarA,4.)*(2.*delta(p, q) + 1. - 4.*pow(dA[p], 2.)/pow(RbarA, 2.))*(pow(dB[2], 2.) - pow(dA[2], 2.));
        }
        else if ((abs(Lbar)>Error) && (abs(RbarA)<=Error) && (abs(RbarB)> Error)){
            result = l[2]/pow(Lbar, 6.)*((2.*(pow((-1), p + q)*3*(pow(l[0], 2.)- pow(l[1], 2.))
                                              + 2.*e(p, q, 2)*pow(Lbar, 2))*l[1-q]*l[2]*V[2] + pow((-1), q+1)*(2.*delta(p,q) + 1.) * pow(Lbar,4)*V[1-q] + 4.*pow(l[p],2)*l[q]*W[2])/2.*(1./pow(RbarB,2)-1./pow(RbarA, 2))
                                         - l[q]*l[2]*((2.*delta(p,q) + 1.)*pow(Lbar, 2.) - 4.*pow(l[p], 2.))*(TbarB/pow(RbarB,2.)-TbarA/pow(RbarA, 2.))
                                         + pow((-1), p+1)*pow(V[2], 2)/pow(Lbar, 2)*((4*pow(l[q], 2)-pow(l[1-q], 2))*l[1-q]*l[2]*V[2]
                                                                                     + (pow(l[q], 2) - 2*pow(l[1-q], 2))*pow(l[q], 2)*V[1-q] + 3.*l[0]*l[1]*pow(l[1-q], 2)*V[q])*(1./pow(RbarB,4)-1./pow(RbarA, 4))
                                         + pow((-1), p + q)*V[2]/pow(Lbar, 2)*(V[q]*pow(pow(l[1], 2)-pow(l[0], 2),2)+4*l[0]*l[1]*(l[1-q]*l[2]*V[2]+pow(l[q], 2)*V[1-q]))*(TbarB/pow(RbarB,4)-TbarA/pow(RbarA,4)));
        }
        else if ((abs(Lbar)>Error) && (abs(RbarA)>Error) && (abs(RbarB)<= Error)){
            result = l[2]/pow(Lbar, 6)*((2.*(pow((-1), p + q)*3.*(pow(l[0], 2)- pow(l[1], 2))
                                             + 2.*e(p, q, 2)*pow(Lbar, 2))*l[1-q]*l[2]*V[2] + pow((-1), q+1)*(2.*delta(p,q) + 1.) * pow(Lbar,4)*V[1-q] + 4.*pow(l[p],2)*l[q]*W[2])/2.*(1./pow(RbarB,2)-1./pow(RbarA, 2))
                                        - l[q]*l[2]*((2.*delta(p,q) + 1.)*pow(Lbar, 2)-4.*pow(l[p], 2))*(TbarB/pow(RbarB,2)-TbarA/pow(RbarA, 2))
                                        + pow((-1), p+1)*pow(V[2], 2)/pow(Lbar, 2)*((4.*pow(l[q], 2)-pow(l[1-q], 2))*l[1-q]*l[2]*V[2]
                                                                                    + (pow(l[q], 2)-2.*pow(l[1-q], 2))*pow(l[q], 2)*V[1-q]+3.*l[0]*l[1]*pow(l[1-q], 2)*V[q])*(1./pow(RbarB,4)-1./pow(RbarA, 4))
                                        + pow((-1), p + q)*V[2]/pow(Lbar, 2)*(V[q]*pow(pow(l[1], 2)-pow(l[0], 2),2)+4.*l[0]*l[1]*(l[1-q]*l[2]*V[2]+pow(l[q], 2)*V[1-q]))*(TbarB/pow(RbarB,4)-TbarA/pow(RbarA,4)));
        }
        else {
            result = l[2]/pow(Lbar, 6)*((2.*(pow((-1), p + q)*3.*(pow(l[0], 2)- pow(l[1], 2))
                                             + 2.*e(p, q, 2)*pow(Lbar, 2))*l[1-q]*l[2]*V[2] + pow((-1), q+1)*(2.*delta(p,q) + 1.) * pow(Lbar,4)*V[1-q] + 4.*pow(l[p],2)*l[q]*W[2])/2.*(1./pow(RbarB,2)-1./pow(RbarA, 2))
                                        - l[q]*l[2]*((2.*delta(p,q) + 1.)*pow(Lbar, 2)-4.*pow(l[p], 2))*(TbarB/RbarB/pow(RbarB,2)-TbarA/pow(RbarA, 2))
                                        + pow((-1), p+1)*pow(V[2], 2)/pow(Lbar, 2)*((4.*pow(l[q], 2)-pow(l[1-q], 2))*l[1-q]*l[2]*V[2]
                                                                                    + (pow(l[q], 2)-2.*pow(l[1-q], 2))*pow(l[q], 2)*V[1-q]+3.*l[0]*l[1]*pow(l[1-q], 2)*V[q])*(1./pow(RbarB,4)-1./pow(RbarA, 4))
                                        + pow((-1), p + q)*V[2]/pow(Lbar, 2)*(V[q]*pow(pow(l[1], 2)-pow(l[0], 2),2)+4.*l[0]*l[1]*(l[1-q]*l[2]*V[2]+pow(l[q], 2)*V[1-q]))*(TbarB/pow(RbarB,4)-TbarA/pow(RbarA,4)));
        }
        return result;
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::complex<double> L42_L20(int p, int q, double l[3], std::complex<double> dA[3], std::complex<double> dB[3], std::complex<double> V[3],
                                 std::complex<double> Lbar, std::complex<double> RbarA, std::complex<double> RbarB, std::complex<double> TbarA, std::complex<double> TbarB){
        std::complex<double> result;
        if ((abs(Lbar) <= Error) && (abs(RbarA) > Error)){
            result = (2.*dA[p]*dA[q]/pow(RbarA, 2)-delta(p, q))*l[2]/pow(RbarA, 2);
        }
        else if ((abs(Lbar) > Error) && (abs(RbarA) <= Error) && (abs(RbarB) > Error)){
            result = -l[2]/pow(Lbar, 4)*((pow((-1), p + q)*l[1-p]*l[1-q] - l[p]*l[q])*(TbarB/pow(RbarB, 2)));
        }
        else if ((abs(Lbar) > Error) && (abs(RbarA) > Error) && (abs(RbarB) <= Error)){
            result = -l[2]/pow(Lbar, 4)*((pow((-1), p + q)*l[1-p]*l[1-q] - l[p]*l[q])*(-TbarA/pow(RbarA, 2)));
        }
        else{
            result = -l[2]/pow(Lbar, 4)*(V[2]*(pow((-1), p+1)*l[1-p]*l[q] + pow((-1), q+1)*l[p]*l[1-q])*(1./pow(RbarB, 2)-1./pow(RbarA, 2))
                                         + (pow((-1), p + q)*l[1-p]*l[1-q] - l[p]*l[q]) * (TbarB/pow(RbarB, 2) - TbarA/pow(RbarA, 2)));
        }
        return result;
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    int e(int p,int q,int s){
        int Result;
        if((p==0 && q==1 && s==2)||(p==1 && q==2 && s==0)||(p==2 && q==0 && s==1)){
            Result=1;
        }
        else if((p==2 && q==1 && s==0)||(p==1 && q==0 && s==2)||(p==0 && q==2 && s==1)){
            Result=-1;
        }
        else{
            Result=0;
        }
        return Result;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    double SI(int p,int q,int t, double xA[3], double xB[3], double x[3]){
        std::complex<double> Result;
        double l[3];
        std::complex<double> dA[3],dB[3],V[3],W[3],MA[3],MB[3] ;
        std::complex<double> Vbar,Lbar,RbarA,RbarB,TbarA,TbarB,V3,L3,R3A,R3B,T3A,T3B;
        l[0]=xB[0]-xA[0];
        l[1]=xB[1]-xA[1];
        l[2]=xB[2]-xA[2];
        dA[0] = x[0]-xA[0] ;
        dA[1] = x[1]-xA[1] ;
        dA[2] = x[2]-xA[2] ;
        dB[0] = x[0]-xB[0] ;
        dB[1] = x[1]-xB[1] ;
        dB[2] = x[2]-xB[2] ;

        cross(dA, l, V);
        cross(l,V,W);
        cross(dA,V,MA);
        cross(dB,V,MB);
        Vbar = sqrt(pow(V[0],2)+pow(V[1],2)) ;
        Lbar = sqrt(pow(l[0],2)+pow(l[1],2)) ;
        RbarA = sqrt(pow(dA[0],2)+pow(dA[1],2)) ;
        RbarB = sqrt(pow(dB[0],2)+pow(dB[1],2)) ;
        TbarA = dA[0]*l[0]+dA[1]*l[1] ;
        TbarB = dB[0]*l[0]+dB[1]*l[1] ;

        V3 = sqrt(pow(Vbar,2)/pow(gama2,2)+pow(V[2],2)) ;
        L3 = sqrt(pow(Lbar,2)+pow(l[2],2)/pow(gama2,2)) ;
        R3A = sqrt(pow(RbarA,2)+pow(dA[2],2)/pow(gama2,2)) ;
        R3B = sqrt(pow(RbarB,2)+pow(dB[2],2)/pow(gama2,2)) ;
        T3A = TbarA+dA[2]*l[2]/pow(gama2,2) ;
        T3B = TbarB+dB[2]*l[2]/pow(gama2,2) ;

        std::complex<double> Vj,Lj,RjA,RjB,TjA,TjB,gamaj,fj,gj,mmj;
        int it,ks;

        if ((p==0 || p==1) && (q==0 || q==1) && (t==0 || t==1)) {
            it = p;
            ks = 1 - q;
            Result = e(it, ks, 2) * pow((-1.),t+1) / gama2 * (J01(2, gama2, 1 - t, l, V, W, V3, L3, R3A, R3B, T3A, T3B)
                                                              -J01(1 - t, gama2, 2, l, V, W, V3, L3, R3A, R3B, T3A, T3B))
                     +delta(it, ks) / gama2 * J01(t, gama2, 2, l, V, W, V3, L3, R3A, R3B, T3A, T3B)
                     -delta(it, t) * (2. / pow(gama2,2) * gama2 * I43_I21(ks, gama2, l, dA, dB, V, MA, MB, Vbar, V3)
                                      +1. / gama2 * I21(gama2, ks, l, dA, dB, V, MA, MB, Vbar, V3))
                     -2. / pow(gama2,2) * gama2 * (delta(ks, t) * I43_I21(it, gama2, l, dA, dB, V, MA, MB, Vbar, V3)
                                                   +delta(it, ks) * I43_I21(t, gama2, l, dA, dB, V, MA, MB, Vbar, V3))
                     +2. / pow(gama2,2)* gama2 * I65_I43_J23(it, ks, t, gama2, l, dA, dB, V, MA, MB, Vbar, V3, Lbar, L3, R3A, R3B, T3A, T3B)
                     +1. / gama2 * I43_J23(it, t, gama2, ks, l, dA, dB, V, W, MA, MB, Vbar, V3, Lbar, L3, R3A, R3B, T3A, T3B);



            for (int j=0;j<2;j++) {
                gj = g[j];
                fj = f[j];
                gamaj = gama[j];
                Vj = sqrt(pow(Vbar,2)  / pow(gamaj,2)  + pow(V[2],2) );
                Lj = sqrt(pow(Lbar,2) + pow(l[2],2) / pow(gamaj,2));
                RjA = sqrt(pow(RbarA,2) + pow(dA[2],2)  / pow(gamaj,2));
                RjB = sqrt(pow(RbarB,2) + pow(dB[2],2) / pow(gamaj,2));
                TjA = TbarA + dA[2] * l[2] / pow(gamaj,2);
                TjB = TbarB + dB[2] * l[2] / pow(gamaj,2);
                Result = Result -
                         delta(it, t) * (2./ pow(gama2,2) * fj * gamaj * I43_I21(ks, gamaj, l, dA, dB, V, MA, MB, Vbar, Vj)
                                         +gj / gamaj * I21(gamaj, ks, l, dA, dB, V, MA, MB, Vbar, Vj))
                         -2. / pow(gama2,2) * fj * gamaj * (delta(ks, t) * I43_I21(it, gamaj, l, dA, dB, V, MA, MB, Vbar, Vj)
                                                            +delta(it, ks) * I43_I21(t, gamaj, l, dA, dB, V, MA, MB, Vbar, Vj))
                         +2. / pow(gama2,2) * fj * gamaj *
                          I65_I43_J23(it, ks, t, gamaj, l, dA, dB, V, MA, MB, Vbar, Vj, Lbar, Lj, RjA, RjB, TjA,
                                      TjB)
                         +gj / gamaj *
                          I43_J23(it, t, gamaj, ks, l, dA, dB, V, W, MA, MB, Vbar, Vj, Lbar, Lj, RjA, RjB, TjA, TjB);

            }



            Result = Result / 4. / pi * pow((-1),ks+1);
        }

        else if ((p==0 || p==1) && (q==0 || q==1) && t==2) {
            it = p;
            ks = 1 - q;
            Result = delta(it, ks) / pow(gama2,3) * J01(2, gama2, 2, l, V, W, V3, L3, R3A, R3B, T3A, T3B)
                     -1. / gama2 * J01(ks, gama2, it, l, V, W, V3, L3, R3A, R3B, T3A, T3B)
                     -2. / pow(gama2,2)/ gama2 * (I43_J23(it, ks, gama2, 2, l, dA, dB, V, W, MA, MB, Vbar, V3, Lbar, L3, R3A, R3B, T3A, T3B)
                                                  -delta(it, ks) * I21(gama2, 2, l, dA, dB, V, MA, MB, Vbar, V3));
            for(int j=0;j<2;j++) {

                gj = g[j];
                fj = f[j];
                gamaj = gama[j];
                Vj = sqrt(pow(Vbar,2) / pow(gamaj,2) + pow(V[2],2));
                Lj = sqrt(pow(Lbar,2)  + pow(l[2],2) / pow(gamaj,2));
                RjA = sqrt(pow(RbarA,2) + pow(dA[2],2) / pow(gamaj,2));
                RjB = sqrt(pow(RbarB,2) + pow(dB[2],2) / pow(gamaj,2));
                TjA = TbarA + dA[2] * l[2] / pow(gamaj,2);
                TjB = TbarB + dB[2] * l[2] / pow(gamaj,2);
                Result = Result - gj / gamaj * J01(it, gamaj, ks, l, V, W, Vj, Lj, RjA, RjB, TjA, TjB)
                         -2. / pow(gama2,2) * fj / gamaj *
                          (I43_J23(it, ks, gamaj, 2, l, dA, dB, V, W, MA, MB, Vbar, Vj, Lbar, Lj, RjA, RjB, TjA,
                                   TjB)
                           -delta(it, ks) * I21(gamaj, 2, l, dA, dB, V, MA, MB, Vbar, Vj));

            }
            Result = Result / 4. / pi * pow((-1),ks+1);
        }
        else if (p==2 && (q==0 || q==1) && (t==0 || t==1)) {
            ks = 1- q;
            Result = 0;
            for(int j=0;j<2;j++) {

                mmj = m[j];
                gj = g[j];
                fj = f[j];
                gamaj = gama[j];
                Vj = sqrt(pow(Vbar,2) / pow(gamaj,2) + pow(V[2],2));
                Lj = sqrt(pow(Lbar,2)  + pow(l[2],2) / pow(gamaj,2));
                RjA = sqrt(pow(RbarA,2) + pow(dA[2],2) / pow(gamaj,2));
                RjB = sqrt(pow(RbarB,2) + pow(dB[2],2) / pow(gamaj,2));
                TjA = TbarA + dA[2] * l[2] / pow(gamaj,2);
                TjB = TbarB + dB[2] * l[2] / pow(gamaj,2);
                Result = Result - mmj * gj / gamaj * J01(t, gamaj, ks, l, V, W, Vj, Lj, RjA, RjB, TjA, TjB)
                         -2. / pow(gama2,2) * mmj * fj / gamaj *
                          (I43_J23(ks, t, gamaj, 2, l, dA, dB, V, W, MA, MB, Vbar, Vj, Lbar, Lj, RjA, RjB, TjA,
                                   TjB)
                           -delta(ks, t) * I21(gamaj, 2, l, dA, dB, V, MA, MB, Vbar, Vj));

            }
            Result = Result / 4. / pi * pow((-1),ks+1);
        }
        else if (p==2 && (q==0 || q==1) && t==2) {
            ks = 1 - q;
            Result = 0;
            for(int j=0;j<2;j++) {

                mmj = m[j];
                gj = g[j];
                fj = f[j];
                gamaj = gama[j];
                Vj = sqrt(pow(Vbar,2) / pow(gamaj,2)  + pow(V[2],2));
                Lj = sqrt(pow(Lbar,2) + pow(l[2],2) / pow(gamaj,2));
                RjA = sqrt(pow(RbarA,2)  + pow(dA[2],2)  / pow(gamaj,2));
                RjB = sqrt(pow(RbarB,2)  + pow(dB[2],2) / pow(gamaj,2));
                TjA = TbarA + dA[2] * l[2] / pow(gamaj,2);
                TjB = TbarB + dB[2] * l[2] / pow(gamaj,2);
                Result = Result - mmj / pow(gamaj,2)  * (gj / gamaj * J01(2, gamaj, ks, l, V, W, Vj, Lj, RjA, RjB, TjA, TjB)
                                                         -2. / pow(gama2,2) * fj * gamaj * J01(ks, gamaj, 2, l, V, W, Vj, Lj, RjA, RjB, TjA, TjB));

            }
            Result = Result / 4. / pi * pow((-1),ks+1) ;
        }
        else if ((p==0 || p==1) && q==2 && (t==0 || t==1)) {
            ks = 1 - p;
            Result = delta(ks, t) / gama2 * I21(gama2, 2, l, dA, dB, V, MA, MB, Vbar, V3)
                     -1. / gama2 * I43_J23(ks, t, gama2, 2, l, dA, dB, V, W, MA, MB, Vbar, V3, Lbar, L3, R3A, R3B, T3A, T3B);
            for(int j=0;j<2;j++) {

                gj = g[j];
                gamaj = gama[j];
                Vj = sqrt(pow(Vbar,2) / pow(gamaj,2)  + pow(V[2],2));
                Lj = sqrt(pow(Lbar,2) + pow(l[2],2) / pow(gamaj,2));
                RjA = sqrt(pow(RbarA,2)  + pow(dA[2],2)  / pow(gamaj,2));
                RjB = sqrt(pow(RbarB,2)  + pow(dB[2],2) / pow(gamaj,2));
                TjA = TbarA + dA[2] * l[2] / pow(gamaj,2);
                TjB = TbarB + dB[2] * l[2] / pow(gamaj,2);
                Result = Result - gj * gamaj * J01(t, gamaj, ks, l, V, W, Vj, Lj, RjA, RjB, TjA, TjB)
                         +delta(ks, t) * gj / gamaj * I21(gamaj, 2, l, dA, dB, V, MA, MB, Vbar, Vj)
                         -gj / gamaj * I43_J23(ks, t, gamaj, 2, l, dA, dB, V, W, MA, MB, Vbar, Vj, Lbar, Lj, RjA, RjB, TjA, TjB);

            }
            Result = Result / 4. / pi * pow((-1),ks+1);
        }
        else if ((p==0 || p==1) && q==2 && t==2) {
            ks = 1 - p;
            Result = 1. / gama2 * J01(ks, gama2, 2, l, V, W, V3, L3, R3A, R3B, T3A, T3B);
            for(int j=0;j<2;j++) {

                gj = g[j];
                gamaj = gama[j];
                Vj = sqrt(pow(Vbar,2) / pow(gamaj,2)  + pow(V[2],2));
                Lj = sqrt(pow(Lbar,2) + pow(l[2],2) / pow(gamaj,2));
                RjA = sqrt(pow(RbarA,2)  + pow(dA[2],2)  / pow(gamaj,2));
                RjB = sqrt(pow(RbarB,2)  + pow(dB[2],2) / pow(gamaj,2));
                TjA = TbarA + dA[2] * l[2] / pow(gamaj,2);
                TjB = TbarB + dB[2] * l[2] / pow(gamaj,2);
                Result = Result - gj / gamaj * (J01(2, gamaj, ks, l, V, W, Vj, Lj, RjA, RjB, TjA, TjB)
                                                -J01(ks, gamaj, 2, l, V, W, Vj, Lj, RjA, RjB, TjA, TjB));

            }
            Result = Result / 4. / pi * pow((-1),ks+1);
        }
        else if (p==2 && q==2 && (t==0 || t==1)) {
            Result = 0;
            for(int j=0;j<2;j++) {
                mmj = m[j];
                gj = g[j];
                gamaj = gama[j];
                Vj = sqrt(pow(Vbar,2) / pow(gamaj,2)  + pow(V[2],2));
                Lj = sqrt(pow(Lbar,2) + pow(l[2],2) / pow(gamaj,2));
                RjA = sqrt(pow(RbarA,2)  + pow(dA[2],2)  / pow(gamaj,2));
                RjB = sqrt(pow(RbarB,2)  + pow(dB[2],2) / pow(gamaj,2));
                TjA = TbarA + dA[2] * l[2] / pow(gamaj,2);
                TjB = TbarB + dB[2] * l[2] / pow(gamaj,2);
                Result = Result - pow((-1.),t+1) * mmj * gj / gamaj * (J01(1 - t, gamaj, 2, l, V, W, Vj, Lj, RjA, RjB, TjA, TjB)
                                                                       -J01(2, gamaj, 1 - t, l, V, W, Vj, Lj, RjA, RjB, TjA, TjB));

            }
            Result = Result / 4. / pi;
        }
        else if (p==2 && q==2 && t==2) {
            Result = 0;
            for(int j=0;j<2;j++) {
                mmj = m[j];
                gj = g[j];
                gamaj = gama[j];
                Vj = sqrt(pow(Vbar,2) / pow(gamaj,2)  + pow(V[2],2));
                Lj = sqrt(pow(Lbar,2) + pow(l[2],2) / pow(gamaj,2));
                RjA = sqrt(pow(RbarA,2)  + pow(dA[2],2)  / pow(gamaj,2));
                RjB = sqrt(pow(RbarB,2)  + pow(dB[2],2) / pow(gamaj,2));
                TjA = TbarA + dA[2] * l[2] / pow(gamaj,2);
                TjB = TbarB + dB[2] * l[2] / pow(gamaj,2);
                Result = Result + mmj * gj / gamaj * (J01(0, gamaj, 1, l, V, W, Vj, Lj, RjA, RjB, TjA, TjB)
                                                      -J01(1, gamaj, 0, l, V, W, Vj, Lj, RjA, RjB, TjA, TjB));

            }
            Result = Result / 4. / pi;

        }
        double RRR=Result.real();
        return RRR;

    }

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

    double SC(int p,int q,int t, double xA[3],double xB[3],double x[3]){
        double l[3];
        double X33[3] = {x[0],x[1],-x[2]};
        std::complex<double> d33A[3], d33B[3], V33[3], W33[3], M33A[3], M33B[3];
        std::complex<double> gama3=gama[2];
        std::complex<double> result, Lbar, Vbar33, Rbar33A, Rbar33B, Tbar33A, Tbar33B, V3, L3, R3A, R3B, T3A, T3B;
        for (int i=0;i<3;i++){
            l[i] = xB[i] - xA[i];
            d33A[i] = X33[i] - xA[i];
            d33B[i] = X33[i] - xB[i];
        }
        cross(d33A,l,V33);
        cross(l,V33,W33);
        cross(d33A,V33,M33A);
        cross(d33B,V33,M33B);
        Lbar = sqrt(pow(l[0], 2) + pow(l[1], 2));
        Vbar33 = sqrt(pow(V33[0], 2) + pow(V33[1], 2));
        Rbar33A = sqrt(pow(d33A[0], 2) + pow(d33A[1], 2));
        Rbar33B = sqrt(pow(d33B[0], 2) + pow(d33B[1], 2));
        Tbar33A = d33A[0]*l[0] + d33A[1]*l[1];
        Tbar33B = d33B[0]*l[0] + d33B[1]*l[1];
        V3 = sqrt(pow(V33[2], 2) + pow(Vbar33/gama3, 2));
        L3 = sqrt(pow(Lbar, 2) + pow(l[2]/gama3, 2));
        R3A = sqrt(pow(Rbar33A, 2) + pow(d33A[2]/gama3, 2));
        R3B = sqrt(pow(Rbar33B, 2) + pow(d33B[2]/gama3, 2));
        T3A = Tbar33A+d33A[2]*l[2]/pow(gama3, 2);
        T3B = Tbar33B+d33B[2]*l[2]/pow(gama3, 2);
        std::complex<double> FIJ, GIJ, gamaI, gamaJ, dIJA[3], dIJB[3], VIJ[3], WIJ[3], MIJA[3], MIJB[3], VbarIJ, RbarIJA, RbarIJB, TbarIJA, TbarIJB, VI, LI, RIA, RIB, TIA, TIB;
        int it, ks;
        if ((p == 0 || p == 1) && (q == 0 || q == 1) && (t == 0 || t == 1)){
            it = p;
            ks = 1 - q;
            result = e(it, ks, 2)*pow((-1), t+1)/gama3*(J01(2, gama3, 1-t, l, V33, W33, V3, L3, R3A, R3B, T3A, T3B) - J01(1-t, gama3, 2, l, V33, W33, V3, L3, R3A, R3B, T3A, T3B))
                     + delta(it, ks)/gama3*J01(t, gama3, 2, l, V33, W33, V3, L3, R3A, R3B, T3A, T3B)
                     - delta(it, t)*(2./pow(gama3, 2)*gama3*I43_I21(ks, gama3, l, d33A, d33B, V33, M33A, M33B, Vbar33, V3)
                                     + 1./gama3*I21(gama3, ks, l, d33A, d33B, V33, M33A, M33B, Vbar33, V3))
                     - 2./pow(gama3,2)*gama3*(delta(ks, t)*I43_I21(it, gama3, l, d33A, d33B, V33, M33A, M33B, Vbar33, V3)
                                              + delta(it, ks)*I43_I21(t, gama3, l, d33A, d33B, V33, M33A, M33B, Vbar33, V3))
                     + 2./pow(gama3,2)*gama3*I65_I43_J23(it, ks, t, gama3, l, d33A, d33B, V33, M33A, M33B, Vbar33, V3, Lbar, L3, R3A, R3B, T3A, T3B)
                     + 1./gama3*I43_J23(it, t, gama3, ks, l, d33A, d33B, V33, W33, M33A, M33B, Vbar33, V3, Lbar, L3, R3A, R3B, T3A, T3B)
                     + 4./pow(gama3,2)*L64_L42(it, ks, t, l, d33A, d33B, V33, W33, Lbar, Rbar33A, Rbar33B, Tbar33A, Tbar33B);
            for (int i=0; i<2; i++){
                for (int j=0; j<2; j++){
                    FIJ = F[i][j];
                    GIJ = G[i][j];
                    gamaI = gama[i];
                    gamaJ = gama[j];
                    std::complex<double>XIJ[3] = {x[0], x[1], -gamaI/gamaJ*x[2]};
                    for (int k=0;k<3;k++){

                        dIJA[k] = XIJ[k] - xA[k];
                        dIJB[k] = XIJ[k] - xB[k];
                    }
                    cross(dIJA, l,VIJ);
                    cross(l, VIJ,WIJ);
                    cross(dIJA, VIJ,MIJA);
                    cross(dIJB, VIJ,MIJB);
                    VbarIJ = sqrt(pow(VIJ[0], 2) + pow(VIJ[1], 2));
                    RbarIJA = sqrt(pow(dIJA[0], 2) + pow(dIJA[1], 2));
                    RbarIJB = sqrt(pow(dIJB[0], 2) + pow(dIJB[1], 2));
                    TbarIJA = dIJA[0]*l[0] + dIJA[1]*l[1];
                    TbarIJB = dIJB[0]*l[0] + dIJB[1]*l[1];

                    VI = sqrt(pow(VbarIJ, 2)/pow(gamaI, 2) + pow(VIJ[2], 2));
                    LI = sqrt(pow(Lbar, 2) + pow(l[2], 2)/pow(gamaI, 2));
                    RIA = sqrt(pow(RbarIJA, 2) + pow(dIJA[2], 2)/pow(gamaI, 2));
                    RIB = sqrt(pow(RbarIJB, 2) + pow(dIJB[2], 2)/pow(gamaI, 2));
                    TIA = TbarIJA + dIJA[2]*l[2]/pow(gamaI, 2);
                    TIB = TbarIJB + dIJB[2]*l[2]/pow(gamaI, 2);
                    result = result - delta(it, t) *(2./pow(gama3, 2)*FIJ*gamaI*I43_I21(ks, gamaI, l, dIJA, dIJB, VIJ, MIJA, MIJB, VbarIJ, VI)
                                                     + GIJ/gamaI*I21(gamaI, ks, l ,dIJA, dIJB, VIJ, MIJA, MIJB, VbarIJ, VI))
                             - 2./pow(gama3, 2)*FIJ*gamaI*(delta(ks, t)*I43_I21(it, gamaI, l, dIJA, dIJB, VIJ, MIJA, MIJB, VbarIJ, VI)
                                                           + delta(it, ks)*I43_I21(t, gamaI, l, dIJA, dIJB, VIJ, MIJA, MIJB, VbarIJ, VI))
                             + 2./pow(gama3, 2)*FIJ*gamaI*I65_I43_J23(it, ks, t, gamaI, l, dIJA, dIJB, VIJ, MIJA, MIJB, VbarIJ, VI, Lbar, LI, RIA, RIB, TIA, TIB)
                             + GIJ/gamaI*I43_J23(it, t, gamaI, ks, l, dIJA, dIJB, VIJ, WIJ, MIJA, MIJB, VbarIJ, VI, Lbar, LI, RIA, RIB, TIA, TIB)
                             + 4./pow(gama3, 2)*FIJ*L64_L42(it, ks, t, l, dIJA, dIJB, VIJ, WIJ, Lbar,RbarIJA, RbarIJB, TbarIJA, TbarIJB);
                };
            }
            result = result/4./pi*pow((-1),ks+1);
        }
        else if ((p == 0 || p == 1) && (q == 0 || q == 1) && t == 2){
            it = p;
            ks = 1 - q;
            result = -delta(it, ks)/pow(gama3, 3)*J01(2, gama3, 2, l, V33, W33, V3, L3, R3A, R3B, T3A, T3B)
                     +1./gama3*J01(ks, gama3, it, l, V33, W33, V3, L3, R3A, R3B, T3A, T3B)
                     +2./pow(gama3,2)/gama3*(I43_J23(it, ks, gama3, 2, l, d33A, d33B, V33, W33, M33A, M33B, Vbar33, V3, Lbar, L3, R3A, R3B, T3A, T3B)
                                             - delta(it, ks)*I21(gama3, 2, l, d33A, d33B, V33, M33A, M33B, Vbar33, V3))
                     -2./pow(gama3,2)*FF*L42_L20(it, ks, l, d33A, d33B, V33, Lbar, Rbar33A, Rbar33B, Tbar33A, Tbar33B);
            for (int i = 0; i<2; i++){
                for (int j = 0; j<2; j++){
                    FIJ = F[i][j];
                    GIJ = G[i][j];
                    gamaI = gama[i];
                    gamaJ = gama[j];
                    std::complex<double> XIJ[3] = {x[0], x[1], -gamaI/gamaJ*x[2]};

                    for (int k=0;k<3;k++){
                        dIJA[k] = XIJ[k] - xA[k];
                        dIJB[k] = XIJ[k] - xB[k];
                    }
                    cross(dIJA, l,VIJ);
                    cross(l, VIJ,WIJ);
                    cross(dIJA, VIJ,MIJA);
                    cross(dIJB, VIJ,MIJB);
                    VbarIJ = sqrt(pow(VIJ[0], 2) + pow(VIJ[1], 2));
                    RbarIJA = sqrt(pow(dIJA[0], 2) + pow(dIJA[1], 2));
                    RbarIJB = sqrt(pow(dIJB[0], 2) + pow(dIJB[1], 2));
                    TbarIJA = dIJA[0]*l[0] + dIJA[1]*l[1];
                    TbarIJB = dIJB[0]*l[0] + dIJB[1]*l[1];

                    VI = sqrt(pow(VbarIJ, 2)/pow(gamaI, 2) + pow(VIJ[2], 2));
                    LI = sqrt(pow(Lbar, 2) + pow(l[2], 2)/pow(gamaI, 2));
                    RIA = sqrt(pow(RbarIJA, 2) + pow(dIJA[2], 2)/pow(gamaI, 2));
                    RIB = sqrt(pow(RbarIJB, 2) + pow(dIJB[2], 2)/pow(gamaI, 2));
                    TIA = TbarIJA + dIJA[2]*l[2]/pow(gamaI, 2);
                    TIB = TbarIJB + dIJB[2]*l[2]/pow(gamaI, 2);
                    result = result + GIJ/gamaJ*J01(it, gamaI, ks, l, VIJ, WIJ, VI, LI, RIA, RIB, TIA, TIB)
                             + 2./pow(gama3, 2)*FIJ/gamaJ*(I43_J23(it, ks, gamaI, 2, l, dIJA, dIJB, VIJ, WIJ, MIJA, MIJB, VbarIJ, VI, Lbar, LI, RIA, RIB, TIA, TIB)
                                                           - delta(it, ks)*I21(gamaI, 2, l, dIJA, dIJB, VIJ, MIJA, MIJB, VbarIJ, VI));
                }
            }
            result = result/4./pi*pow((-1),ks+1);
        }
        else if (p == 2 && (q == 0 || q == 1) && (t == 0 || t == 1)){
            ks = 1- q;
            result = 2./pow(gama3, 2)*FF*L42_L20(ks, t, l, d33A, d33B, V33, Lbar, Rbar33A, Rbar33B, Tbar33A, Tbar33B);
            for (int i = 0; i<2; i++){
                for (int j = 0; j<2; j++){
                    complex<double> mmJ = m[j];
                    FIJ = F[i][j];
                    GIJ = G[i][j];
                    gamaI = gama[i];
                    gamaJ = gama[j];
                    std::complex<double> XIJ[3] = {x[0], x[1], -gamaI/gamaJ*x[2]};

                    for (int k=0;k<3;k++){
                        dIJA[k] = XIJ[k] - xA[k];
                        dIJB[k] = XIJ[k] - xB[k];
                    }
                    cross(dIJA, l,VIJ);
                    cross(l, VIJ,WIJ);
                    cross(dIJA, VIJ,MIJA);
                    cross(dIJB, VIJ,MIJB);

                    VbarIJ = sqrt(pow(VIJ[0], 2) + pow(VIJ[1], 2));
                    RbarIJA = sqrt(pow(dIJA[0], 2) + pow(dIJA[1], 2));
                    RbarIJB = sqrt(pow(dIJB[0], 2) + pow(dIJB[1], 2));
                    TbarIJA = dIJA[0]*l[0] + dIJA[1]*l[1];
                    TbarIJB = dIJB[0]*l[0] + dIJB[1]*l[1];

                    VI = sqrt(pow(VbarIJ, 2)/pow(gamaI, 2) + pow(VIJ[2], 2));
                    LI = sqrt(pow(Lbar, 2) + pow(l[2], 2)/pow(gamaI, 2));
                    RIA = sqrt(pow(RbarIJA, 2) + pow(dIJA[2], 2)/pow(gamaI, 2));
                    RIB = sqrt(pow(RbarIJB, 2) + pow(dIJB[2], 2)/pow(gamaI, 2));
                    TIA = TbarIJA + dIJA[2]*l[2]/pow(gamaI, 2);
                    TIB = TbarIJB + dIJB[2]*l[2]/pow(gamaI, 2);
                    result = result + mmJ*GIJ/gamaJ*J01(t, gamaI, ks, l, VIJ, WIJ, VI, LI, RIA, RIB, TIA, TIB)
                             + 2./pow(gama3, 2)*mmJ*FIJ/gamaJ*(I43_J23(ks, t, gamaI, 2, l, dIJA, dIJB, VIJ, WIJ, MIJA, MIJB, VbarIJ, VI, Lbar, LI, RIA, RIB, TIA, TIB)
                                                               - delta(ks, t)*I21(gamaI, 2, l, dIJA, dIJB, VIJ, MIJA, MIJB, VbarIJ, VI));
                }
            }
            result = result/4./pi*pow((-1), ks+1);
        }
        else if (p == 2 && (q == 0 || q == 1) && t == 2){
            ks = 1 - q;
            result = 0;
            for (int i = 0; i<2; i++) {
                for (int j = 0; j < 2; j++) {
                    complex<double> mmJ = m[j];
                    FIJ = F[i][j];
                    GIJ = G[i][j];
                    gamaI = gama[i];
                    gamaJ = gama[j];
                    std::complex<double> XIJ[3] = {x[0], x[1], -gamaI/gamaJ*x[2]};
                    for (int k=0;k<3;k++){
                        dIJA[k] = XIJ[k] - xA[k];
                        dIJB[k] = XIJ[k] - xB[k];
                    }
                    cross(dIJA, l,VIJ);
                    cross(l, VIJ,WIJ);
                    cross(dIJA, VIJ,MIJA);
                    cross(dIJB, VIJ,MIJB);
                    VbarIJ = sqrt(pow(VIJ[0], 2) + pow(VIJ[1], 2));
                    RbarIJA = sqrt(pow(dIJA[0], 2) + pow(dIJA[1], 2));
                    RbarIJB = sqrt(pow(dIJB[0], 2) + pow(dIJB[1], 2));
                    TbarIJA = dIJA[0] * l[0] + dIJA[1] * l[1];
                    TbarIJB = dIJB[0] * l[0] + dIJB[1] * l[1];

                    VI = sqrt(pow(VbarIJ, 2) / pow(gamaI, 2) + pow(VIJ[2], 2));
                    LI = sqrt(pow(Lbar, 2) + pow(l[2], 2) / pow(gamaI, 2));
                    RIA = sqrt(pow(RbarIJA, 2) + pow(dIJA[2], 2) / pow(gamaI, 2));
                    RIB = sqrt(pow(RbarIJB, 2) + pow(dIJB[2], 2) / pow(gamaI, 2));
                    TIA = TbarIJA + dIJA[2] * l[2] / pow(gamaI, 2);
                    TIB = TbarIJB + dIJB[2] * l[2] / pow(gamaI, 2);
                    result = result -
                             mmJ / pow(gamaJ, 2) * (GIJ / gamaI * J01(2, gamaI, ks, l, VIJ, WIJ, VI, LI, RIA, RIB, TIA, TIB)
                                                    - 2. / pow(gama3, 2) * FIJ * gamaI *
                                                      J01(ks, gamaI, 2, l, VIJ, WIJ, VI, LI, RIA, RIB, TIA, TIB));
                }
            }
            result = result/4./pi*pow((-1), ks+1);
        }
        else if ( (p == 0 || p == 1) && q == 2 && (t == 0 || t == 1)){
            ks = 1 - p;
            result = delta(ks, t)/gama3*I21(gama3, 2, l, d33A, d33B, V33, M33A, M33B, Vbar33, V3)
                     - 1./gama3*I43_J23(ks, t, gama3, 2, l, d33A, d33B, V33, W33, M33A, M33B, Vbar33, V3, Lbar, L3, R3A, R3B, T3A, T3B);
            for (int i = 0; i<2; i++) {
                for (int j = 0; j < 2; j++) {
                    GIJ = G[i][j];
                    gamaI = gama[i];
                    gamaJ = gama[j];
                    std::complex<double> XIJ[3] = {x[0], x[1], -gamaI/gamaJ*x[2]};
                    for (int k=0;k<3;k++){
                        dIJA[k] = XIJ[k] - xA[k];
                        dIJB[k] = XIJ[k] - xB[k];
                    }
                    cross(dIJA, l,VIJ);
                    cross(l, VIJ,WIJ);
                    cross(dIJA, VIJ,MIJA);
                    cross(dIJB, VIJ,MIJB);
                    VbarIJ = sqrt(pow(VIJ[0], 2) + pow(VIJ[1], 2));
                    RbarIJA = sqrt(pow(dIJA[0], 2) + pow(dIJA[1], 2));
                    RbarIJB = sqrt(pow(dIJB[0], 2) + pow(dIJB[1], 2));
                    TbarIJA = dIJA[0] * l[0] + dIJA[1] * l[1];
                    TbarIJB = dIJB[0] * l[0] + dIJB[1] * l[1];

                    VI = sqrt(pow(VbarIJ, 2) / pow(gamaI, 2) + pow(VIJ[2], 2));
                    LI = sqrt(pow(Lbar, 2) + pow(l[2], 2) / pow(gamaI, 2));
                    RIA = sqrt(pow(RbarIJA, 2) + pow(dIJA[2], 2) / pow(gamaI, 2));
                    RIB = sqrt(pow(RbarIJB, 2) + pow(dIJB[2], 2) / pow(gamaI, 2));
                    TIA = TbarIJA + dIJA[2] * l[2] / pow(gamaI, 2);
                    TIB = TbarIJB + dIJB[2] * l[2] / pow(gamaI, 2);
                    result = result - GIJ*gamaI*J01(t, gamaI, ks, l, VIJ, WIJ, VI, LI, RIA, RIB, TIA, TIB)
                             + delta(ks, t)*GIJ/gamaI*I21(gamaI, 2, l, dIJA, dIJB, VIJ, MIJA, MIJB, VbarIJ, VI)
                             - GIJ/gamaI*I43_J23(ks, t, gamaI, 2, l, dIJA, dIJB, VIJ, WIJ, MIJA, MIJB, VbarIJ, VI, Lbar, LI, RIA, RIB, TIA, TIB);
                }
            }
            result = result/4./pi*pow((-1), ks+1);
        }
        else if ((p == 0 || p ==1) && q == 2 && t == 2) {
            ks = 1 - p;
            result = -1. / gama3 * J01(ks, gama3, 2, l, V33, W33, V3, L3, R3A, R3B, T3A, T3B);
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    GIJ = G[i][j];
                    gamaI = gama[i];
                    gamaJ = gama[j];
                    std::complex<double> XIJ[3] = {x[0], x[1], -gamaI/gamaJ*x[2]};
                    for (int k=0;k<3;k++){
                        dIJA[k] = XIJ[k] - xA[k];
                        dIJB[k] = XIJ[k] - xB[k];
                    }
                    cross(dIJA, l, VIJ);
                    cross(l, VIJ,WIJ);
                    VbarIJ = sqrt(pow(VIJ[0], 2) + pow(VIJ[1], 2));
                    RbarIJA = sqrt(pow(dIJA[0], 2) + pow(dIJA[1], 2));
                    RbarIJB = sqrt(pow(dIJB[0], 2) + pow(dIJB[1], 2));
                    TbarIJA = dIJA[0] * l[0] + dIJA[1] * l[1];
                    TbarIJB = dIJB[0] * l[0] + dIJB[1] * l[1];

                    VI = sqrt(pow(VbarIJ, 2) / pow(gamaI, 2) + pow(VIJ[2], 2));
                    LI = sqrt(pow(Lbar, 2) + pow(l[2], 2) / pow(gamaI, 2));
                    RIA = sqrt(pow(RbarIJA, 2) + pow(dIJA[2], 2) / pow(gamaI, 2));
                    RIB = sqrt(pow(RbarIJB, 2) + pow(dIJB[2], 2) / pow(gamaI, 2));
                    TIA = TbarIJA + dIJA[2] * l[2] / pow(gamaI, 2);
                    TIB = TbarIJB + dIJB[2] * l[2] / pow(gamaI, 2);
                    result = result + GIJ / gamaJ * (J01(2, gamaI, ks, l, VIJ, WIJ, VI, LI, RIA, RIB, TIA, TIB)
                                                     - J01(ks, gamaI, 2, l, VIJ, WIJ, VI, LI, RIA, RIB, TIA, TIB));
                }
            }
            result = result / 4. / pi * pow((-1), ks+1);
        }
        else if ( p == 2 && q == 2 && (t == 0 || t == 1)){
            result = 0;
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    complex<double> mmJ = m[j];
                    FIJ = F[i][j];
                    GIJ = G[i][j];
                    gamaI = gama[i];
                    gamaJ = gama[j];
                    std::complex<double> XIJ[3] = {x[0], x[1], -gamaI/gamaJ*x[2]};
                    for (int k=0;k<3;k++){
                        dIJA[k] = XIJ[k] - xA[k];
                        dIJB[k] = XIJ[k] - xB[k];
                    }
                    cross(dIJA, l,VIJ);
                    cross(l, VIJ,WIJ);
                    VbarIJ = sqrt(pow(VIJ[0], 2) + pow(VIJ[1], 2));
                    RbarIJA = sqrt(pow(dIJA[0], 2) + pow(dIJA[1], 2));
                    RbarIJB = sqrt(pow(dIJB[0], 2) + pow(dIJB[1], 2));
                    TbarIJA = dIJA[0] * l[0] + dIJA[1] * l[1];
                    TbarIJB = dIJB[0] * l[0] + dIJB[1] * l[1];

                    VI = sqrt(pow(VbarIJ, 2) / pow(gamaI, 2) + pow(VIJ[2], 2));
                    LI = sqrt(pow(Lbar, 2) + pow(l[2], 2) / pow(gamaI, 2));
                    RIA = sqrt(pow(RbarIJA, 2) + pow(dIJA[2], 2) / pow(gamaI, 2));
                    RIB = sqrt(pow(RbarIJB, 2) + pow(dIJB[2], 2) / pow(gamaI, 2));
                    TIA = TbarIJA + dIJA[2] * l[2] / pow(gamaI, 2);
                    TIB = TbarIJB + dIJB[2] * l[2] / pow(gamaI, 2);
                    result = result + pow((-1), t+1)*mmJ*GIJ/gamaJ*(J01(1-t, gamaI, 2, l, VIJ, WIJ, VI, LI, RIA, RIB, TIA, TIB)
                                                                    - J01(2, gamaI, 1-t, l, VIJ, WIJ, VI, LI, RIA, RIB, TIA, TIB));

                }
            }
            result = result/4./pi;
        }
        else if (p == 2 && p == 2 && t == 2){
            result = 0;
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    complex<double> mmJ = m[j];
                    GIJ = G[i][j];
                    gamaI = gama[i];
                    gamaJ = gama[j];
                    std::complex<double> XIJ[3] = {x[0], x[1], -gamaI/gamaJ*x[2]};
                    for (int k=0;k<3;k++){
                        dIJA[k] = XIJ[k] - xA[k];
                        dIJB[k] = XIJ[k] - xB[k];
                    }
                    cross(dIJA, l,VIJ);
                    cross(l, VIJ,WIJ);
                    VbarIJ = sqrt(pow(VIJ[0], 2) + pow(VIJ[1], 2));
                    RbarIJA = sqrt(pow(dIJA[0], 2) + pow(dIJA[1], 2));
                    RbarIJB = sqrt(pow(dIJB[0], 2) + pow(dIJB[1], 2));
                    TbarIJA = dIJA[0] * l[0] + dIJA[1] * l[1];
                    TbarIJB = dIJB[0] * l[0] + dIJB[1] * l[1];

                    VI = sqrt(pow(VbarIJ, 2) / pow(gamaI, 2) + pow(VIJ[2], 2));
                    LI = sqrt(pow(Lbar, 2) + pow(l[2], 2) / pow(gamaI, 2));
                    RIA = sqrt(pow(RbarIJA, 2) + pow(dIJA[2], 2) / pow(gamaI, 2));
                    RIB = sqrt(pow(RbarIJB, 2) + pow(dIJB[2], 2) / pow(gamaI, 2));
                    TIA = TbarIJA + dIJA[2] * l[2] / pow(gamaI, 2);
                    TIB = TbarIJB + dIJB[2] * l[2] / pow(gamaI, 2);
                    result = result + mmJ*GIJ/pow(gamaJ, 2)*gamaI*(J01(0, gamaI, 1, l, VIJ, WIJ, VI, LI, RIA, RIB, TIA, TIB)
                                                                   - J01(1, gamaI, 0, l, VIJ, WIJ, VI, LI, RIA, RIB, TIA, TIB));
                }
            }
            result = result/4./pi;
        }
        return result.real();

    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    double sL(int P,int Q,int T,double xA[3],double xB[3],double x[3], bool FS){

        double Result;
        if (FS){
            if(Q==0){
                Result=-cos(phi)*(SI(P,0,T,xA,xB,x)+SC(P,0,T,xA,xB,x))-sin(phi)*(SI(P,1,T,xA,xB,x)+SC(P,1,T,xA,xB,x));
            }
            else if (Q==1){
                Result=sin(phi)*cos(de)*(SI(P,0,T,xA,xB,x) + SC(P,0,T,xA,xB,x))-cos(phi)*cos(de)*(SI(P,1,T,xA,xB,x) + SC(P,1,T,xA,xB,x))-sin(de)*(SI(P,2,T,xA,xB,x)+SC(P,2,T,xA,xB,x));
            }
            else if (Q==2){
                Result=-sin(phi)*sin(de)*(SI(P,0,T,xA,xB,x) + SC(P,0,T,xA,xB,x))+cos(phi)*sin(de)*(SI(P,1,T,xA,xB,x) + SC(P,1,T,xA,xB,x))-cos(de)*(SI(P,2,T,xA,xB,x) + SC(P,2,T,xA,xB,x));

            }
        }
        else {
            if(Q==0){
                Result=-cos(phi)*(SI(P,0,T,xA,xB,x))-sin(phi)*(SI(P,1,T,xA,xB,x));
            }
            else if (Q==1){
                Result=sin(phi)*cos(de)*(SI(P,0,T,xA,xB,x))-cos(phi)*cos(de)*(SI(P,1,T,xA,xB,x))-sin(de)*(SI(P,2,T,xA,xB,x));
            }
            else if (Q==2){
                Result=-sin(phi)*sin(de)*(SI(P,0,T,xA,xB,x))+cos(phi)*sin(de)*(SI(P,1,T,xA,xB,x))-cos(de)*(SI(P,2,T,xA,xB,x));
            }
        }
        return Result;
    }
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    double strain(int P,int Q,int T,double xA[3],double xB[3],double xC[3],
                  double xD[3],double x[3], bool FS){
        double Result;
        Result=sL(P,Q,T,xA,xB,x,FS)+sL(P,Q,T,xB,xC,x,FS)+sL(P,Q,T,xC,xD,x,FS)+sL(P,Q,T,xD,xA,x,FS);
        return Result;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void Stress(int Q,double xA[3],double xB[3],double xC[3],
                double xD[3],double x[3], double res[6], bool FS){
        double e11=-strain(0,Q,0,xA,xB,xC,xD,x,FS);
        double e22=-strain(1,Q,1,xA,xB,xC,xD,x,FS);
        double e33=-strain(2,Q,2,xA,xB,xC,xD,x,FS);
        double e23=-0.5*(strain(1,Q,2,xA,xB,xC,xD,x,FS)+strain(2,Q,1,xA,xB,xC,xD,x,FS));
        double e31=-0.5*(strain(0,Q,2,xA,xB,xC,xD,x,FS)+strain(2,Q,0,xA,xB,xC,xD,x,FS));
        double e12=-0.5*(strain(1,Q,0,xA,xB,xC,xD,x,FS)+strain(0,Q,1,xA,xB,xC,xD,x,FS));
        double deformation[6];
        deformation[0]=e11;
        deformation[1]=e22;
        deformation[2]=e33;
        deformation[3]=2*e23;
        deformation[4]=2*e31;
        deformation[5]=2*e12;

        double Stiff[6][6];
        Stiff[0][0]=c11;
        Stiff[0][1]=c12;
        Stiff[0][2]=c13;
        Stiff[1][0]=c12;
        Stiff[1][1]=c11;
        Stiff[1][2]=c13;
        Stiff[2][0]=c13;
        Stiff[2][1]=c13;
        Stiff[2][2]=c33;
        Stiff[3][3]=c44;
        Stiff[4][4]=c44;
        Stiff[5][5]=c66;


        for (int i = 0; i<6; i++){
            res[i] = 0;
            for (int j = 0; j<6; j++){
                res[i] += Stiff[i][j]*deformation[j];
            }

        }


    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void Normal_Shear_Stress(double xA[3],double xB[3],double xC[3],
                             double xD[3],double x[3], double Result[3][3],bool FS){
        //Result(m,n) represent the m direction stress due to a n direction dislocation
        //we consider n=[0,-sin(de),cos(de)]
        ;
        double res[6];
        Stress(0, xA, xB, xC, xD, x,res,FS);

        Result[0][0]=res[4];
        Result[1][0]=res[3];
        Result[2][0]=res[2];

        Stress(1, xA, xB, xC, xD, x,res, FS);
        Result[0][1]=res[4];
        Result[1][1]=res[3];
        Result[2][1]=res[2];

        Stress(2, xA, xB, xC, xD, x,res,FS);
        Result[0][2]=res[4];
        Result[1][2]=res[3];
        Result[2][2]=res[2];

    }

}



