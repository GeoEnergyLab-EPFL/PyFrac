//
// Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
// Geo-Energy Laboratory, 2016-2019.  All rights reserved.
// See the LICENSE.TXT file for more details.
//
// Contributors:
// Weihan Zhang
// Fatima-Ezzahra Moukhtari
// Brice Lecampion
// Dmitry Nikolskiy


#ifndef HFPX3D_VC_OUTPUT_H
#define HFPX3D_VC_OUTPUT_H

#include <il/Array2D.h>
#include <il/Array.h>
#include <iostream>

namespace hfp3d {

//store a 2D matrix
void output( il::Array2D<double> global, const std::string trg_dir){
    std::string f_path1 = trg_dir+".txt";
    const char *format1 = "%.12g";
    int max=global.size(1);
    FILE *of1 = std::fopen(f_path1.c_str(), "w");
    for (int j = 0; j < max; ++j) {
        for (int k = 0; k < max; ++k) {
            double out = global(j, k);
            std::fprintf(of1, format1, out);
            std::fprintf(of1, " ");

        }
        std::fprintf(of1, "\n");
    }
    std::fclose(of1);
}



    // store an 1D array
    void output_array( il::Array<double> global, const std::string trg_dir){
        std::string f_path1 = trg_dir+".txt";
        const char *format1 = "%.6g";
        int max=global.size();
        FILE *of1 = std::fopen(f_path1.c_str(), "w");
        for (int j = 0; j < max; ++j) {

                double out = global[j];
                std::fprintf(of1, format1, out);
                std::fprintf(of1, " ");



        }
        std::fclose(of1);
    }
}

#endif
