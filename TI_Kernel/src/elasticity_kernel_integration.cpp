//
// Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
// Geo-Energy Laboratory, 2016-2019.  All rights reserved.
// See the LICENSE.TXT file for more details.
//
// Contributors:
// Weihan Zhang
// Fatima-Ezzahra Moukhtari


#include <iostream>
#include <il/math.h>
#include <src/elasticity_kernel_integration.h>
#include <il/linearAlgebra.h>
#include <src/inputE.h>
#include <il/Array2D.h>

namespace hfp3d {


    double Error=1e-10;



    //delta and phi defined in (Pan,2014), delta is the angle inclined with the isotropic plan. phi is a rotation angle
    // in the isotropic plan which can be considered as 0.

    //phi=0, and if delta=0, dislocation loop in the isotropic plan, delta=pi/2, dislocation loop perpendicular to
    //the isotropic plan
    double de=il::pi/2;
    double phi=il::pi/180*0;

    il::Array<double>  Ce{hfp3d::Cmatrix("stiffness_matrix.json")};
    double c11=Ce[0];
    double c12=Ce[1];
    double c13=Ce[2];
    double c33=Ce[3];
    double c44=Ce[4];
    double c66=0.5*(c11-c12);

    double a=c44*(c13+c44);
    double b=pow((c13+c44),2)+pow(c44,2)-c11*c33;
    double c=c44*(c13+c44);


    std::complex<double> m0=(-b+sqrt(std::complex<double>(pow(b,2)-4*a*c)))/2./a;
    std::complex<double> m1=1./m0;
    std::complex<double> gama0 = sqrt(std::complex<double>((c44+m0*(c13+c44))/c11)) ;
    std::complex<double> gama1 = sqrt(std::complex<double>((c44+m1*(c13+c44))/c11)) ;
    double gama2 = sqrt(c44/c66) ;
    std::complex<double> theta=c11*(gama0+gama1)/(c13+c44);
    std::complex<double> f0=m1/(m0-m1);
    std::complex<double> f1=m0/(m1-m0);
    double f2=1;
    std::complex<double> g0=(m1+1.)/(m0-m1);
    std::complex<double> g1=(m0+1.)/(m1-m0);
    double g2=1;


    /////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
    std::complex<double> J01(int p, std::complex<double> gamaj,int k,il::StaticArray<double,3> l,il::StaticArray<double,3> V,il::StaticArray<double,3> W,
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
    std::complex<double> I43_I21(int p,std::complex<double> gamaj,il::StaticArray<double,3> l,il::StaticArray<double,3> dA,
                                 il::StaticArray<double,3> dB,il::StaticArray<double,3> V,il::StaticArray<double,3> MA,
                                 il::StaticArray<double,3> MB,std::complex<double> Vbar,std::complex<double> Vj){
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
    std::complex<double> I43_J23(int p, int q,  std::complex<double> gamaj, int k,il::StaticArray<double,3> l,il::StaticArray<double,3> dA,
                                 il::StaticArray<double,3> dB,il::StaticArray<double,3> V,il::StaticArray<double,3> W,il::StaticArray<double,3> MA,
                                 il::StaticArray<double,3> MB, std::complex<double> Vbar, std::complex<double> Vj, std::complex<double> Lbar, std::complex<double> Lj,
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
    std::complex<double> I65_I43_J23(int P,int Q, int T,  std::complex<double> gamaj,il::StaticArray<double,3> l,il::StaticArray<double,3> dA,
                                     il::StaticArray<double,3> dB,il::StaticArray<double,3> V,il::StaticArray<double,3> MA,
                                     il::StaticArray<double,3> MB, std::complex<double> Vbar, std::complex<double> Vj, std::complex<double> Lbar, std::complex<double> Lj, std::complex<double> RjA,
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
                     +l[2]*V[1-q]/pow(Vbar,3)/Vj*(4*pow((-1),q+1)*pow(V[1-p],2)+pow((-1),p+1)*pow(V[2],2)*(pow(V[1-q],2)-3*pow(V[q],2))*pow(gamaj,2)/pow(Vbar,2))
                      *((dB[2]*Vj*sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2))/(pow((dB[2]*V[2]),2)+pow(MB[2],2)))-(dA[2]*Vj*sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))/(pow((dA[2]*V[2]),2)+pow(MA[2],2))))
                     +l[2]*V[q]/Vbar*(3*pow((-1),p+q)*V[2]*(pow(V[q],2)-3*pow(V[1-q],2))*pow(gamaj,2)/pow(Vbar,4))
                      *((MB[2]*sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2))/(pow((dB[2]*V[2]),2)+pow(MB[2],2)))-(MA[2]*sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))/(pow((dA[2]*V[2]),2)+pow(MA[2],2))))
                     -pow((-1),p+1)*2*l[2]*pow(V[2],2)*V[1-q]*(pow(V[1-q],2)-3*pow(V[q],2))/pow(Vbar,3)/pow(Vj,3)*((pow((dB[2]*Vj),3)*sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2))/pow((pow((dB[2]*V[2]),2)+pow(MB[2],2)),2))
                                                                                                                   -(pow((dA[2]*Vj),3)*sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))/pow((pow((dA[2]*V[2]),2)+pow(MA[2],2)),2)))
                     -pow((-1),p+q)*2*l[2]*V[2]*V[q]*(pow(V[q],2)-3*pow(V[1-q],2))*pow(gamaj,2)/pow(Vbar,5)*((MB[2]*pow((sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2))),3)/pow((pow((dB[2]*V[2]),2)+pow(MB[2],2)),2))
                                                                                                             -(MA[2]*pow((sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))),3)/pow((pow((dA[2]*V[2]),2)+pow(MA[2],2)),2)))
                     -pow((-1),p+1)*l[2]*pow(V[2],2)/pow(Vbar,3)/Vj*(V[1-q]*(pow(V[1-q],2)-3*pow(V[q],2))*pow(gamaj,2)/pow(Vbar,2)-pow(l[2],2)/pow(Lbar,4)*(pow((-1),q+1)*V[1-q]*(pow(l[1],2)-pow(l[0],2))+2*l[0]*l[1]*V[q]))
                      *((dB[2]*Vj/sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2)))-(dA[2]*Vj/sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))))
                     -pow((-1),p+q)*l[2]*pow(V[2],2)/pow(Vbar,3)/pow(Vj,2)*(V[2]*V[q]*(pow(V[q],2)-3*pow(V[1-q],2))*pow(gamaj,2)/pow(Vbar,2)
                                                                            +l[2]/pow(Lbar,4)*(pow(l[q],3)*(2*pow(V[1-q],2)-pow(V[q],2))+3*l[1-q]*(pow(Lbar,2)*V[0]*V[1]-l[0]*l[1]*pow(V[q],2))))
                      *((MB[2]/sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2)))-(MA[2]/sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2)))) ;

        }
        else if ((abs(Lbar)>Error) && (abs(V[2])<=Error) && (abs(Vbar)>Error) && (abs(MA[2])>Error) && (abs(MB[2])<=Error)){
            Result = l[2]*pow(l[p],2)*l[q]/pow(Lbar,2)/pow(Lj,2)*(1./RjB-1./RjA)
                     +l[2]/pow(Lbar,4)/pow(Lj,2)/pow(Vj,2)*pow((-1.),p+1)*l[p]*((2.*l[q]*l[1-p]+pow((-1.),p+q)*l[1-q]*l[p])*pow(Lbar,2)*V[2]
                                                                                +l[2]/pow(gamaj,2)*((delta(p,q)+1.)*l[q]*l[1-p]*l[2]*V[2]-pow((-1.),p+q)*pow(l[p],3)*V[1-q]-pow(l[1-p],2)*l[q]*V[1-p]))*(TjB/RjB-TjA/RjA)
                     +l[2]*V[1-q]/pow(Vbar,3)/Vj*(4*pow((-1),q+1)*pow(V[1-p],2)+pow((-1),p+1)*pow(V[2],2)*(pow(V[1-q],2)-3*pow(V[q],2))*pow(gamaj,2)/pow(Vbar,2))
                      *((dB[2]*Vj*sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2))/(pow((dB[2]*V[2]),2)+pow(MB[2],2)))-(dA[2]*Vj*sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))/(pow((dA[2]*V[2]),2)+pow(MA[2],2))))
                     +l[2]*V[q]/Vbar*(3*pow((-1),p+q)*V[2]*(pow(V[q],2)-3*pow(V[1-q],2))*pow(gamaj,2)/pow(Vbar,4))
                      *((MB[2]*sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2))/(pow((dB[2]*V[2]),2)+pow(MB[2],2)))-(MA[2]*sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))/(pow((dA[2]*V[2]),2)+pow(MA[2],2))))
                     -pow((-1),p+1)*2*l[2]*pow(V[2],2)*V[1-q]*(pow(V[1-q],2)-3*pow(V[q],2))/pow(Vbar,3)/pow(Vj,3)*((pow((dB[2]*Vj),3)*sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2))/pow((pow((dB[2]*V[2]),2)+pow(MB[2],2)),2))
                                                                                                                   -(pow((dA[2]*Vj),3)*sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))/pow((pow((dA[2]*V[2]),2)+pow(MA[2],2)),2)))
                     -pow((-1),p+q)*2*l[2]*V[2]*V[q]*(pow(V[q],2)-3*pow(V[1-q],2))*pow(gamaj,2)/pow(Vbar,5)*((MB[2]*pow((sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2))),3)/pow((pow((dB[2]*V[2]),2)+pow(MB[2],2)),2))
                                                                                                             -(MA[2]*pow((sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))),3)/pow((pow((dA[2]*V[2]),2)+pow(MA[2],2)),2)))
                     -pow((-1),p+1)*l[2]*pow(V[2],2)/pow(Vbar,3)/Vj*(V[1-q]*(pow(V[1-q],2)-3*pow(V[q],2))*pow(gamaj,2)/pow(Vbar,2)-pow(l[2],2)/pow(Lbar,4)*(pow((-1),q+1)*V[1-q]*(pow(l[1],2)-pow(l[0],2))+2*l[0]*l[1]*V[q]))
                      *((dB[2]*Vj/sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2)))-(dA[2]*Vj/sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))))
                     -pow((-1),p+q)*l[2]*pow(V[2],2)/pow(Vbar,3)/pow(Vj,2)*(V[2]*V[q]*(pow(V[q],2)-3*pow(V[1-q],2))*pow(gamaj,2)/pow(Vbar,2)
                                                                            +l[2]/pow(Lbar,4)*(pow(l[q],3)*(2*pow(V[1-q],2)-pow(V[q],2))+3*l[1-q]*(pow(Lbar,2)*V[0]*V[1]-l[0]*l[1]*pow(V[q],2))))
                      *((MB[2]/sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2)))-(MA[2]/sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2)))) ;

        }
        else if ((abs(Lbar)>Error) && (abs(V[2])<=Error) && (abs(Vbar)>Error) && (abs(MA[2])>Error) && (abs(MB[2])>Error)){


            Result = l[2]*pow(l[p],2)*l[q]/pow(Lbar,2)/pow(Lj,2)*(1./RjB-1./RjA)
                     +l[2]/pow(Lbar,4)/pow(Lj,2)/pow(Vj,2)*pow((-1),p+1)*l[p]*((2*l[q]*l[1-p]+pow((-1),p+q)*l[1-q]*l[p])*pow(Lbar,2)*V[2]
                                                                               +l[2]/pow(gamaj,2)*((delta(p,q)+1.)*l[q]*l[1-p]*l[2]*V[2]-pow((-1.),p+q)*pow(l[p],3)*V[1-q]-pow(l[1-p],2)*l[q]*V[1-p]))*(TjB/RjB-TjA/RjA)
                     +l[2]*V[1-q]/pow(Vbar,3)/Vj*(4*pow((-1),q+1)*pow(V[1-p],2)+pow((-1),p+1)*pow(V[2],2)*(pow(V[1-q],2)-3*pow(V[q],2))*pow(gamaj,2)/pow(Vbar,2))
                      *((dB[2]*Vj*sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2))/(pow((dB[2]*V[2]),2)+pow(MB[2],2)))-(dA[2]*Vj*sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))/(pow((dA[2]*V[2]),2)+pow(MA[2],2))))
                     +l[2]*V[q]/Vbar*(3*pow((-1),p+q)*V[2]*(pow(V[q],2)-3*pow(V[1-q],2))*pow(gamaj,2)/pow(Vbar,4))
                      *((MB[2]*sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2))/(pow((dB[2]*V[2]),2)+pow(MB[2],2)))-(MA[2]*sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))/(pow((dA[2]*V[2]),2)+pow(MA[2],2))))
                     -pow((-1),p+1)*2*l[2]*pow(V[2],2)*V[1-q]*(pow(V[1-q],2)-3*pow(V[q],2))/pow(Vbar,3)/pow(Vj,3)*((pow((dB[2]*Vj),3)*sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2))/pow((pow((dB[2]*V[2]),2)+pow(MB[2],2)),2))
                                                                                                                   -(pow((dA[2]*Vj),3)*sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))/pow((pow((dA[2]*V[2]),2)+pow(MA[2],2)),2)))
                     -pow((-1),p+q)*2*l[2]*V[2]*V[q]*(pow(V[q],2)-3*pow(V[1-q],2))*pow(gamaj,2)/pow(Vbar,5)*((MB[2]*pow((sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2))),3)/pow((pow((dB[2]*V[2]),2)+pow(MB[2],2)),2))
                                                                                                             -(MA[2]*pow((sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))),3)/pow((pow((dA[2]*V[2]),2)+pow(MA[2],2)),2)))
                     -pow((-1),p+1)*l[2]*pow(V[2],2)/pow(Vbar,3)/Vj*(V[1-q]*(pow(V[1-q],2)-3*pow(V[q],2))*pow(gamaj,2)/pow(Vbar,2)-pow(l[2],2)/pow(Lbar,4)*(pow((-1),q+1)*V[1-q]*(pow(l[1],2)-pow(l[0],2))+2*l[0]*l[1]*V[q]))
                      *((dB[2]*Vj/sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2)))-(dA[2]*Vj/sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))))
                     -pow((-1),p+q)*l[2]*pow(V[2],2)/pow(Vbar,3)/pow(Vj,2)*(V[2]*V[q]*(pow(V[q],2)-3*pow(V[1-q],2))*pow(gamaj,2)/pow(Vbar,2)
                                                                            +l[2]/pow(Lbar,4)*(pow(l[q],3)*(2*pow(V[1-q],2)-pow(V[q],2))+3*l[1-q]*(pow(Lbar,2)*V[0]*V[1]-l[0]*l[1]*pow(V[q],2))))
                      *((MB[2]/sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2)))-(MA[2]/sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2)))) ;
        }
        else{
            Result = l[2]*pow(l[p],2)*l[q]/pow(Lbar,2)/pow(Lj,2)*(1./RjB-1./RjA)
                     +l[2]/pow(Lbar,4)/pow(Lj,2)/pow(Vj,2)*pow((-1),p+1)*l[p]*((2*l[q]*l[1-p]+pow((-1),p+q)*l[1-q]*l[p])*pow(Lbar,2)*V[2]
                                                                               +l[2]/pow(gamaj,2)*((delta(p,q)+1.)*l[q]*l[1-p]*l[2]*V[2]-pow((-1.),p+q)*pow(l[p],3)*V[1-q]-pow(l[1-p],2)*l[q]*V[1-p]))*(TjB/RjB-TjA/RjA)
                     -(2.*delta(p,q)+1.)*l[2]*V[q]/gamaj/pow(V[2],2)*(atan(MB[2]*Vbar/gamaj/V[2]/sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2)))
                                                                      -atan(MA[2]*Vbar/gamaj/V[2]/sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))))
                     +l[2]*V[1-q]/pow(Vbar,3)/Vj*(4*pow((-1),q+1)*pow(V[1-p],2)+pow((-1),p+1)*pow(V[2],2)*(pow(V[1-q],2)-3*pow(V[q],2))*pow(gamaj,2)/pow(Vbar,2))
                      *((dB[2]*Vj*sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2))/(pow((dB[2]*V[2]),2)+pow(MB[2],2)))-(dA[2]*Vj*sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))/(pow((dA[2]*V[2]),2)+pow(MA[2],2))))
                     +l[2]*V[q]/Vbar*(-(2.*delta(p,q)+1.)/V[2]+3.*pow((-1),p+q)*V[2]*(pow(V[q],2)-3.*pow(V[1-q],2))*pow(gamaj,2)/pow(Vbar,4))
                      *((MB[2]*sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2))/(pow((dB[2]*V[2]),2)+pow(MB[2],2)))-(MA[2]*sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))/(pow((dA[2]*V[2]),2)+pow(MA[2],2))))
                     -pow((-1),p+1)*2*l[2]*pow(V[2],2)*V[1-q]*(pow(V[1-q],2)-3*pow(V[q],2))/pow(Vbar,3)/pow(Vj,3)*((pow((dB[2]*Vj),3)*sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2))/pow((pow((dB[2]*V[2]),2)+pow(MB[2],2)),2))
                                                                                                                   -(pow((dA[2]*Vj),3)*sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))/pow((pow((dA[2]*V[2]),2)+pow(MA[2],2)),2)))
                     -pow((-1),p+q)*2*l[2]*V[2]*V[q]*(pow(V[q],2)-3*pow(V[1-q],2))*pow(gamaj,2)/pow(Vbar,5)*((MB[2]*pow((sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2))),3)/pow((pow((dB[2]*V[2]),2)+pow(MB[2],2)),2))
                                                                                                             -(MA[2]*pow((sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))),3)/pow((pow((dA[2]*V[2]),2)+pow(MA[2],2)),2)))
                     -pow((-1),p+1)*l[2]*pow(V[2],2)/pow(Vbar,3)/Vj*(V[1-q]*(pow(V[1-q],2)-3*pow(V[q],2))*pow(gamaj,2)/pow(Vbar,2)-pow(l[2],2)/pow(Lbar,4)*(pow((-1),q+1)*V[1-q]*(pow(l[1],2)-pow(l[0],2))+2*l[0]*l[1]*V[q]))
                      *((dB[2]*Vj/sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2)))-(dA[2]*Vj/sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2))))
                     -pow((-1),p+q)*l[2]*pow(V[2],2)/pow(Vbar,3)/pow(Vj,2)*(V[2]*V[q]*(pow(V[q],2)-3*pow(V[1-q],2))*pow(gamaj,2)/pow(Vbar,2)
                                                                            +l[2]/pow(Lbar,4)*(pow(l[q],3)*(2*pow(V[1-q],2)-pow(V[q],2))+3*l[1-q]*(pow(Lbar,2)*V[0]*V[1]-l[0]*l[1]*pow(V[q],2))))
                      *((MB[2]/sqrt(pow((dB[2]*Vj),2)+pow(MB[2],2)))-(MA[2]/sqrt(pow((dA[2]*Vj),2)+pow(MA[2],2)))) ;
        }
        return Result;

    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::complex<double> I21( std::complex<double> gamaj,int k,il::StaticArray<double,3> l,il::StaticArray<double,3> dA,
                              il::StaticArray<double,3> dB,il::StaticArray<double,3> V,il::StaticArray<double,3> MA,
                              il::StaticArray<double,3> MB, std::complex<double> Vbar, std::complex<double> Vj){
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
    int e(int p,int q,int s){
        int Result;
        if((p==0&&q==1&s==2)||(p==1&&q==2&&s==0)||(p==2&&q==0&&s==1)){
            Result=1;
        }
        else if((p==2&&q==1&&s==0)||(p==1&&q==0&&s==2)||(p==0&&q==2&&s==1)){
            Result=-1;
        }
        else{
            Result=0;
        }
        return Result;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    double SI(int p,int q,int t, il::StaticArray<double,3> xA,il::StaticArray<double,3> xB,il::StaticArray<double,3> x){
        std::complex<double> Result;
        il::StaticArray<double,3> l,dA,dB,V,W,MA,MB ;
        il::StaticArray<std::complex<double>,2> mm;
        mm[0]=m0;
        mm[1]=m1;
        double Vbar,Lbar,RbarA,RbarB,TbarA,TbarB,V3,L3,R3A,R3B,T3A,T3B;
        il::StaticArray<std::complex<double>,3> f,g,gama;
        f[0]=f0;
        f[1]=f1;
        f[2]=f2;
        g[0]=g0;
        g[1]=g1;
        g[2]=g2;
        gama[0]=gama0;
        gama[1]=gama1;
        gama[2]=gama2;
        l[0]=xB[0]-xA[0];
        l[1]=xB[1]-xA[1];
        l[2]=xB[2]-xA[2];
        dA[0] = x[0]-xA[0] ;
        dA[1] = x[1]-xA[1] ;
        dA[2] = x[2]-xA[2] ;
        dB[0] = x[0]-xB[0] ;
        dB[1] = x[1]-xB[1] ;
        dB[2] = x[2]-xB[2] ;

        V = cross(dA,l) ;
        W = cross(l,V) ;
        MA = cross(dA,V) ;
        MB = cross(dB,V) ;
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
            Result = e(it, ks, 2) * pow((-1),t+1) / gama2 * (J01(2, gama2, 1 - t, l, V, W, V3, L3, R3A, R3B, T3A, T3B)
                                                             -J01(1 - t, gama2, 2, l, V, W, V3, L3, R3A, R3B, T3A, T3B))
                     +delta(it, ks) / gama2 * J01(t, gama2, 2, l, V, W, V3, L3, R3A, R3B, T3A, T3B)
                     -delta(it, t) * (2 / pow(gama2,2) * gama2 * I43_I21(ks, gama2, l, dA, dB, V, MA, MB, Vbar, V3)
                                      +1 / gama2 * I21(gama2, ks, l, dA, dB, V, MA, MB, Vbar, V3))
                     -2 / pow(gama2,2) * gama2 * (delta(ks, t) * I43_I21(it, gama2, l, dA, dB, V, MA, MB, Vbar, V3)
                                                  +delta(it, ks) * I43_I21(t, gama2, l, dA, dB, V, MA, MB, Vbar, V3))
                     +2 / pow(gama2,2)* gama2 * I65_I43_J23(it, ks, t, gama2, l, dA, dB, V, MA, MB, Vbar, V3, Lbar, L3, R3A, R3B, T3A, T3B)
                     +1 / gama2 * I43_J23(it, t, gama2, ks, l, dA, dB, V, W, MA, MB, Vbar, V3, Lbar, L3, R3A, R3B, T3A, T3B);



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
                         delta(it, t) * (2 / pow(gama2,2) * fj * gamaj * I43_I21(ks, gamaj, l, dA, dB, V, MA, MB, Vbar, Vj)
                                         +gj / gamaj * I21(gamaj, ks, l, dA, dB, V, MA, MB, Vbar, Vj))
                         -2 / pow(gama2,2) * fj * gamaj * (delta(ks, t) * I43_I21(it, gamaj, l, dA, dB, V, MA, MB, Vbar, Vj)
                                                           +delta(it, ks) * I43_I21(t, gamaj, l, dA, dB, V, MA, MB, Vbar, Vj))
                         +2 / pow(gama2,2) * fj * gamaj *
                          I65_I43_J23(it, ks, t, gamaj, l, dA, dB, V, MA, MB, Vbar, Vj, Lbar, Lj, RjA, RjB, TjA,
                                      TjB)
                         +gj / gamaj *
                          I43_J23(it, t, gamaj, ks, l, dA, dB, V, W, MA, MB, Vbar, Vj, Lbar, Lj, RjA, RjB, TjA, TjB);

            }



            Result = Result / 4. / il::pi * pow((-1),ks+1);
        }

        else if ((p==0 || p==1) && (q==0 || q==1) && t==2) {
            it = p;
            ks = 1 - q;
            Result = delta(it, ks) / pow(gama2,3) * J01(2, gama2, 2, l, V, W, V3, L3, R3A, R3B, T3A, T3B)
                     -1 / gama2 * J01(ks, gama2, it, l, V, W, V3, L3, R3A, R3B, T3A, T3B)
                     -2 / pow(gama2,2)/ gama2 * (I43_J23(it, ks, gama2, 2, l, dA, dB, V, W, MA, MB, Vbar, V3, Lbar, L3, R3A, R3B, T3A, T3B)
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
                         -2 / pow(gama2,2) * fj / gamaj *
                          (I43_J23(it, ks, gamaj, 2, l, dA, dB, V, W, MA, MB, Vbar, Vj, Lbar, Lj, RjA, RjB, TjA,
                                   TjB)
                           -delta(it, ks) * I21(gamaj, 2, l, dA, dB, V, MA, MB, Vbar, Vj));

            }
            Result = Result / 4. / il::pi * pow((-1),ks+1);
        }
        else if (p==2 && (q==0 || q==1) && (t==0 || t==1)) {
            ks = 1- q;
            Result = 0;
            for(int j=0;j<2;j++) {

                mmj = mm[j];
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
                         -2 / pow(gama2,2) * mmj * fj / gamaj *
                          (I43_J23(ks, t, gamaj, 2, l, dA, dB, V, W, MA, MB, Vbar, Vj, Lbar, Lj, RjA, RjB, TjA,
                                   TjB)
                           -delta(ks, t) * I21(gamaj, 2, l, dA, dB, V, MA, MB, Vbar, Vj));

            }
            Result = Result / 4. / il::pi * pow((-1),ks+1);
        }
        else if (p==2 && (q==0 || q==1) && t==2) {
            ks = 1 - q;
            Result = 0;
            for(int j=0;j<2;j++) {

                mmj = mm[j];
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
                                                         -2 / pow(gama2,2) * fj * gamaj * J01(ks, gamaj, 2, l, V, W, Vj, Lj, RjA, RjB, TjA, TjB));

            }
            Result = Result / 4. / il::pi * pow((-1),ks+1) ;
        }
        else if ((p==0 || p==1) && q==2 && (t==0 || t==1)) {
            ks = 1 - p;
            Result = delta(ks, t) / gama2 * I21(gama2, 2, l, dA, dB, V, MA, MB, Vbar, V3)
                     -1 / gama2 * I43_J23(ks, t, gama2, 2, l, dA, dB, V, W, MA, MB, Vbar, V3, Lbar, L3, R3A, R3B, T3A, T3B);
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
            Result = Result / 4. / il::pi * pow((-1),ks+1);
        }
        else if ((p==0 || p==1) && q==2 && t==2) {
            ks = 1 - p;
            Result = 1 / gama2 * J01(ks, gama2, 2, l, V, W, V3, L3, R3A, R3B, T3A, T3B);
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
            Result = Result / 4. / il::pi * pow((-1),ks+1);
        }
        else if (p==2 && q==2 && (t==0 || t==1)) {
            Result = 0;
            for(int j=0;j<2;j++) {
                mmj = mm[j];
                gj = g[j];
                gamaj = gama[j];
                Vj = sqrt(pow(Vbar,2) / pow(gamaj,2)  + pow(V[2],2));
                Lj = sqrt(pow(Lbar,2) + pow(l[2],2) / pow(gamaj,2));
                RjA = sqrt(pow(RbarA,2)  + pow(dA[2],2)  / pow(gamaj,2));
                RjB = sqrt(pow(RbarB,2)  + pow(dB[2],2) / pow(gamaj,2));
                TjA = TbarA + dA[2] * l[2] / pow(gamaj,2);
                TjB = TbarB + dB[2] * l[2] / pow(gamaj,2);
                Result = Result - pow((-1),t+1) * mmj * gj / gamaj * (J01(1 - t, gamaj, 2, l, V, W, Vj, Lj, RjA, RjB, TjA, TjB)
                                                                      -J01(2, gamaj, 1 - t, l, V, W, Vj, Lj, RjA, RjB, TjA, TjB));

            }
            Result = Result / 4. / il::pi;
        }
        else if (p==2 && q==2 && t==2) {
            Result = 0;
            for(int j=0;j<2;j++) {
                mmj = mm[j];
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
            Result = Result / 4. / il::pi;

        }
        double RRR=Result.real();
        return RRR;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    double sL(int P,int Q,int T,il::StaticArray<double,3> xA,il::StaticArray<double,3> xB,il::StaticArray<double,3> x){
        double Result;
        if(Q==0){
            Result=-cos(phi)*SI(P,0,T,xA,xB,x)-sin(phi)*SI(P,1,T,xA,xB,x);
        }
        else if (Q==1){
            Result=sin(phi)*cos(de)*SI(P,0,T,xA,xB,x)-cos(phi)*cos(de)*SI(P,1,T,xA,xB,x)-sin(de)*SI(P,2,T,xA,xB,x);
        }
        else if (Q==2){
            Result=-sin(phi)*sin(de)*SI(P,0,T,xA,xB,x)+cos(phi)*sin(de)*SI(P,1,T,xA,xB,x)-cos(de)*SI(P,2,T,xA,xB,x);
        }
        return Result;
    }
//    double sL(int P,int Q,int T,il::StaticArray<double,3> xA,il::StaticArray<double,3> xB,il::StaticArray<double,3> x){
//        double Result;
//        if(Q==0){
//            Result=-SI(P,0,T,xA,xB,x);
//        }
//        else if (Q==1){
//            Result=-cos(de)*SI(P,1,T,xA,xB,x)-sin(de)*SI(P,2,T,xA,xB,x);
//        }
//        else if (Q==2){
//            Result=sin(de)*SI(P,1,T,xA,xB,x)-cos(de)*SI(P,2,T,xA,xB,x);
//        }
//        return Result;
//    }
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    double strain(int P,int Q,int T, il::StaticArray<double,3> xA,il::StaticArray<double,3> xB,il::StaticArray<double,3> xC,
                  il::StaticArray<double,3> xD,il::StaticArray<double,3> x){
        double Result;
        Result=sL(P,Q,T,xA,xB,x)+sL(P,Q,T,xB,xC,x)+sL(P,Q,T,xC,xD,x)+sL(P,Q,T,xD,xA,x);
        return Result;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    il::StaticArray<double,6> Stress(int Q,il::StaticArray<double,3> xA,il::StaticArray<double,3> xB,il::StaticArray<double,3> xC,
                                     il::StaticArray<double,3> xD,il::StaticArray<double,3> x){
        double e11=-strain(0,Q,0,xA,xB,xC,xD,x);
        double e22=-strain(1,Q,1,xA,xB,xC,xD,x);
        double e33=-strain(2,Q,2,xA,xB,xC,xD,x);
        double e23=-0.5*(strain(1,Q,2,xA,xB,xC,xD,x)+strain(2,Q,1,xA,xB,xC,xD,x));
        double e31=-0.5*(strain(0,Q,2,xA,xB,xC,xD,x)+strain(2,Q,0,xA,xB,xC,xD,x));
        double e12=-0.5*(strain(1,Q,0,xA,xB,xC,xD,x)+strain(0,Q,1,xA,xB,xC,xD,x));
        il::StaticArray<double,6> deformation;
        deformation[0]=e11;
        deformation[1]=e22;
        deformation[2]=e33;
        deformation[3]=2*e23;
        deformation[4]=2*e31;
        deformation[5]=2*e12;
        il::StaticArray2D<double,6,6> Stiff={0};
        Stiff(0,0)=c11;
        Stiff(0,1)=c12;
        Stiff(0,2)=c13;
        Stiff(1,0)=c12;
        Stiff(1,1)=c11;
        Stiff(1,2)=c13;
        Stiff(2,0)=c13;
        Stiff(2,1)=c13;
        Stiff(2,2)=c33;
        Stiff(3,3)=c44;
        Stiff(4,4)=c44;
        Stiff(5,5)=c66;

        il::StaticArray<double,6> Stress;
        Stress=il::dot(Stiff,deformation);

        return Stress;

    };

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    il::StaticArray2D<double,3,3> Normal_Shear_Stress(il::StaticArray<double,3> xA,il::StaticArray<double,3> xB,il::StaticArray<double,3> xC,
                                                      il::StaticArray<double,3> xD,il::StaticArray<double,3> x){
        //Result(m,n) represent the m direction stress due to a n direction dislocation
        //we consider n=[0,-sin(de),cos(de)]
        il::StaticArray2D<double,3,3> Result;


        Result(0,0)=Stress(0, xA, xB, xC, xD, x)[4];
        Result(1,0)=Stress(0, xA, xB, xC, xD, x)[3];
        Result(2,0)=Stress(0, xA, xB, xC, xD, x)[2];

        Result(0,1)=Stress(1, xA, xB, xC, xD, x)[4];
        Result(1,1)=Stress(1, xA, xB, xC, xD, x)[3];
        Result(2,1)=Stress(1, xA, xB, xC, xD, x)[2];

        Result(0,2)=Stress(2, xA, xB, xC, xD, x)[4];
        Result(1,2)=Stress(2, xA, xB, xC, xD, x)[3];
        Result(2,2)=Stress(2, xA, xB, xC, xD, x)[2];




        return Result;

    };


}


