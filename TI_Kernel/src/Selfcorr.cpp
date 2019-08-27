//
// Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
// Geo-Energy Laboratory, 2016-2019.  All rights reserved.
// See the LICENSE.TXT file for more details.
//
// Contributors:
// Weihan Zhang

#include <src/Selfcorr.h>

namespace hfp3d {
    il::Array2D<double> Self_corr(il::Array2D<double> global) {

        int n;
        std::ifstream fin3("TipSize.txt");
        fin3 >> n;
        fin3.close();

        int Element[n];
        std::ifstream fin("EltTip.txt");

        for (int i = 0; i < n; i++) {
            fin >> Element[i];
        }

        fin.close();


        double FillF[n];
        std::ifstream fin2("/Users/moukhtar/Desktop/archive-Weihan/backup/PyFrac-ellipse/FillFraction.txt");

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
            global(Element[i], Element[i]) = global(Element[i], Element[i]) * (1. + ac * il::pi / 4);

        }

        return global;

    }

////////////////////////////////////////////////////////////////////////////////////////

    il::Array2D<double> K_local(il::Array2D<double> global) {
        int n;
        std::ifstream fin3("/Users/moukhtar/Desktop/archive-Weihan/backup/PyFrac-ellipse/CrackSize.txt");
        fin3 >> n;
        fin3.close();

        int Element[n];
        std::ifstream fin("/Users/moukhtar/Desktop/archive-Weihan/backup/PyFrac-ellipse/EltCrack.txt");

        for (int i = 0; i < n; i++) {
            fin >> Element[i];
        }

        fin.close();

        il::Array2D<double> K(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                K(i, j) = global(Element[i], Element[j]);
            }
        }
        return K;

    }

////////////////////////////////////////////////////////////////////////////////////////////////////
    il::Array<double> dd_corr(il::Array<double> dd) {
        int n;
        //n for tipsize
        std::ifstream fin3("/Users/moukhtar/Desktop/archive-Weihan/backup/PyFrac-ellipse/TipSize.txt");
        fin3 >> n;
        fin3.close();
        double FillF[n];
        std::ifstream fin2("/Users/moukhtar/Desktop/archive-Weihan/backup/PyFrac-ellipse/FillFraction.txt");

        for (int i = 0; i < n; i++) {
            fin2 >> FillF[i];
        }

            fin2.close();

            int nn = dd.size();
            //nn for crackelement
            int CrackElement[nn];
            std::ifstream fin("/Users/moukhtar/Desktop/archive-Weihan/backup/PyFrac-ellipse/EltCrack.txt");

            for (int i = 0; i < n; i++) {
                fin >> CrackElement[i];
            }

            fin.close();

            int TipElement[n];
            std::ifstream fin4("/Users/moukhtar/Desktop/archive-Weihan/backup/PyFrac-ellipse/EltTip.txt");

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
