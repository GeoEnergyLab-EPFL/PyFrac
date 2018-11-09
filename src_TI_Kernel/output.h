

#ifndef HFPX3D_VC_OUTPUT_H
#define HFPX3D_VC_OUTPUT_H

#include<iostream>
#include <vector>
using namespace std;
namespace hfp3d {

//store a 2D matrix
void output(vector<vector<double>> global, const std::string trg_dir){
    std::string f_path1 = trg_dir+".txt";
    const char *format1 = "%.2g";
    int max=global.size();
    FILE *of1 = std::fopen(f_path1.c_str(), "w");
    for (int j = 0; j < max; ++j) {
        for (int k = 0; k < max; ++k) {
            double out = global[j][k];
            std::fprintf(of1, format1, out);
            std::fprintf(of1, " ");

        }
        std::fprintf(of1, "\n");
    }
    std::fclose(of1);
}



    // store an 1D array
    void output_array(vector<double> global, const std::string trg_dir){
        std::string f_path1 = trg_dir+".txt";
        const char *format1 = "%.4g";
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




#endif //HFPX3D_VC_OUTPUT_H
