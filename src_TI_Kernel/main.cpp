// main.cpp will be used for testing the code parts under development

#include <cstdio>
#include <fstream>
#include "inputE.h"
#include "AssemblyDDM.h"
#include "json.hpp"
#include <vector>
using namespace std;


int main() {

//Read the elastic parameters
    ifstream i("stiffness_matrix.json");
    json j;
    i >> j;


    //il::Array<double>  Ce{Cmatrix("data.json")};
    //Read the mesh parameters
    double L1= j["Mesh"]["L1"].get<double>();
    double L3= j["Mesh"]["L3"].get<double>();
    int n1= j["Mesh"]["n1"].get<int>();
    int n3= j["Mesh"]["n3"].get<int>();
    bool FS = j["Free_Surf_Param"]["FS_Flag"].get<bool>();
    double Frac_Depth = j["Free_Surf_Param"]["Frac_Depth"].get<double>();




//create mesh (L1,L3,n1,n3)
    hfp3d::Mesh mesh1=hfp3d::create_Mesh(L1,  L3, n1,  n3, FS ,Frac_Depth);
    ofstream outputFile;
    outputFile.open("StrainResult.bin",ios::binary);
    vector<vector<double>> global;
    global.resize(mesh1.nelts());
    for (int i =0;i<global.size(); i++){
        global[i].resize(mesh1.nelts());
    }
        cout << "Transverse Isotropic case;" << "\n";

        global = hfp3d::simplified_opening_assembly(mesh1);

        for (int i=0;i<global.size();i++){
            for (int j=0;j<global[i].size();j++){
                outputFile.write((char *) (&(global[i][j])), sizeof(global[i][j]));

        }
    }

   return 0;
}
