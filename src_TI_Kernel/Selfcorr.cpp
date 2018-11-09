//
// Created by Student on 6/8/2017.
//
#include "Selfcorr.h"

namespace hfp3d {
    const double pi = 3.141592653589793238463;
    double Self_corr[][](double global[][]) {

        int n;
        std::ifstream fin3("C:\\python\\PyFrac - ellipse\\TipSize.txt");
        fin3 >> n;
        fin3.close();

        int Element[n];
        std::ifstream fin("C:\\python\\PyFrac - ellipse\\EltTip.txt");

        for (int i = 0; i < n; i++) {
            fin >> Element[i];
        }

        fin.close();


        double FillF[n];
        std::ifstream fin2("C:\\python\\PyFrac - ellipse\\FillFraction.txt");

        for (int i = 0; i < n; i++) {
            fin2 >> FillF[i];
        }

        fin2.close();

        for (int i = 0; i < n; i++) {
            double r = FillF[i] - 0.25;
            if (r < 0.1) {
                r = 0.1;
            }
            double ac = (1 - r) / r;
            global[Element[i]][Element[i]] = global[Element[i]][Element[i]] * (1. + ac * pi / 4);

        }

        return global;

    }

////////////////////////////////////////////////////////////////////////////////////////

    double K_local[][](double global[][]) {
        int n;
        std::ifstream fin3("C:\\python\\PyFrac - ellipse\\CrackSize.txt");
        fin3 >> n;
        fin3.close();

        int Element[n];
        std::ifstream fin("C:\\python\\PyFrac - ellipse\\EltCrack.txt");

        for (int i = 0; i < n; i++) {
            fin >> Element[i];
        }

        fin.close();

        double K[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                K(i, j) = global[Element[i]][Element[j]];
            }
        }
        return K;

    }

////////////////////////////////////////////////////////////////////////////////////////////////////
    double dd_corr[](double dd[]) {
        int n;
        //n for tipsize
        std::ifstream fin3("C:\\python\\PyFrac - ellipse\\TipSize.txt");
        fin3 >> n;
        fin3.close();
        double FillF[n];
        std::ifstream fin2("C:\\python\\PyFrac - ellipse\\FillFraction.txt");

        for (int i = 0; i < n; i++) {
            fin2 >> FillF[i];
        }

            fin2.close();

            int nn = dd.size();
            //nn for crackelement
            int CrackElement[nn];
            std::ifstream fin("C:\\python\\PyFrac - ellipse\\EltCrack.txt");

            for (int i = 0; i < n; i++) {
                fin >> CrackElement[i];
            }

            fin.close();

            int TipElement[n];
            std::ifstream fin4("C:\\python\\PyFrac - ellipsec\\EltTip.txt");

            for (int i = 0; i < n; i++) {
                fin4 >> TipElement[i];
            }

            fin4.close();

            int ref;

            ref = 0;
        for(int i=0;i<n;i++){

            while(CrackElement[ref]<  TipElement[i]){
                ref++;
            }
            if(FillF[i]<0.4){
                dd[ref]=0;
            }

        }
            return dd;

        }


    }
