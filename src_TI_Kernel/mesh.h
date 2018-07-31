//
// Created by Student on 5/29/2017.
//

#ifndef HFPX3D_VC_MESH_H
#define HFPX3D_VC_MESH_H

#include<iostream>
#include <complex>
#include <vector>
using namespace std;
namespace hfp3d{
    class Mesh {

    public:

        //the mesh occupies the domaine [-Real_L1,Real_L1] times [-Real_L3,Real_L3]
        // the centers of the mesh occupz the domaine [-L1,L1] times [-L3,L3]
        double L1,L3,d1,d3,Real_L1,Real_L3,Frac_depth;
        bool Free_surf_flag;
        // n1 and n3 are separately the number of elements in x direction and y direction
        int n1,n3;
        vector<vector<double>> node;

        void set_values(vector<vector<double>> xy,int nn1,int nn3,double LL1, double LL3,double dd1, double dd3, bool FS, double Depth);

        //k begins from 0
        void get_node(int k, double res[2]);



        // for example, return (3,5) the fourth in x direction, the sixth in y direction
        void coor(int k, int Result[2]);

        // return number of elements of the mesh
        int nelts();
    };

    // Create a mesh in the 2D domaine [-Real_L1,Real_L1] times [-Real_L3,Real_L3] with n1 times n3 elements
    //the nomination of the nodes is from left to right and from bottom to top
    Mesh create_Mesh(double L1, double L3,int n1, int n3, bool FS, double Depth);



}


#endif //HFPX3D_VC_MESH_H
