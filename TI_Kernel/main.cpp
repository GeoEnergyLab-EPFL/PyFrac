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


#include <cstdio>
#include <fstream>
#include <src/inputE.h>
#include <src/elasticity_kernel_isotropy.h>
#include <src/AssemblyDDM.h>
#include <src/output.h>

using namespace std;


int main() {

//Read the elastic parameters
    ifstream i("stiffness_matrix.json");
    json j;
    i >> j;


//    Read the mesh parameters
    double Lx= j["Mesh"]["L1"].get<double>();
    double Ly= j["Mesh"]["L3"].get<double>();
    int nx= j["Mesh"]["n1"].get<int>();
    int ny= j["Mesh"]["n3"].get<int>();


//create mesh (Lx,Ly,nx,ny)
    hfp3d::Mesh mesh1=hfp3d::create_Mesh(Lx,  Ly, nx,  ny);
    ofstream outputFile;
    outputFile.open("StrainResult.bin",ios::binary);
    il::Array<double>  Ce{hfp3d::Cmatrix("stiffness_matrix.json")};
    il::Array2D<double> global;
    if( ((Ce[0] == Ce[3])&&(Ce[2]==Ce[1])&&(Ce[4]==0.5*(Ce[0]-Ce[1]))) ) {
        // if condition is true then the isotropic case
           cout << "Isotropic case;" << "\n";
        //Plain strain Modulus
        double Ep=(Ce[0]-Ce[1])*(Ce[0]+Ce[1])/Ce[0];

        //calculate the elasticity matrix
        global = hfp3d::CIMatrix(mesh1, Ep);


        //calculate the elasticity matrix
        for (int i=0;i<(nx*ny);i++){
            for (int j=0;j<(nx*ny);j++) {
                outputFile.write((char *) (&(global(i,j))), sizeof(global(i,j)));
            }
        }
    } else {
        // if condition is false then print the following
        cout << "Transverse Isotropic case;" << "\n";

        //calculate the elasticity matrix
        global = hfp3d::perpendicular_opening_assembly(mesh1);
        for (int i=0;i<(nx*ny);i++){
            for (int j=0;j<(nx*ny);j++) {
                outputFile.write((char *) (&(global(i,j))), sizeof(global(i,j)));
            }
        }
    }

   return 0;
}
