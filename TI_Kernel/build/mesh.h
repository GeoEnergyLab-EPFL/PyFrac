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

#ifndef HFPX3D_VC_MESH_H
#define HFPX3D_VC_MESH_H

#include <iostream>
#include <complex>
#include <il/math.h>
#include <il/StaticArray.h>
#include <il/StaticArray3D.h>
#include <il/StaticArray4D.h>
#include <il/StaticArray2D.h>
#include <il/Array.h>
#include <il/Array2D.h>

namespace hfp3d{
    class Mesh {

    private:
        il::Array<il::StaticArray<double,2>> node_;

    public:

        //the mesh occupies the domaine [-Real_Lx,Real_Lx] times [-Real_Ly,Real_Ly]
        // the centers of the mesh occupz the domaine [-Lx,Lx] times [-Ly,Ly]
        double Lx,Ly,dx,dy,Real_Lx,Real_Ly;

        // nx and ny are separately the number of elements in x direction and y direction
        int nx,ny;

        void set_values(il::Array<il::StaticArray<double,2>> xy,int nnx,int nny,double LLx, double LLy,double ddx, double ddy);


        //k begins from 0
        il::StaticArray<double,2> node(il::int_t k);



        // for example, return (3,5) the fourth in x direction, the sixth in y direction
        il::StaticArray<int,2> coor(il::int_t k);

        // return number of elements of the mesh
     int nelts();
    };

    // Create a mesh in the 2D domaine [-Real_Lx,Real_Lx] times [-Real_Ly,Real_Ly] with nx times ny elements
    //the nomination of the nodes is from left to right and from bottom to top
    Mesh create_Mesh(double Lx, double Ly,int nx, int ny);



}


#endif
