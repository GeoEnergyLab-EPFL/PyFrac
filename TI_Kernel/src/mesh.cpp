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

#include<iostream>
#include <complex>
#include <il/math.h>
#include <il/StaticArray.h>
#include <il/StaticArray3D.h>
#include <il/StaticArray4D.h>
#include <il/StaticArray2D.h>
#include<il/linearAlgebra.h>
#include <il/Array.h>
#include <il/Array2D.h>
#include <src/mesh.h>

namespace hfp3d{
    void Mesh::set_values(il::Array<il::StaticArray<double,2>> xy,int nnx,int nny,double LLx,double LLy,double ddx, double ddy) {


        node_ = xy;           // list of coordinates of points in the mesh
        nx=nnx;
        ny=nny;
        Lx=LLx;
        Ly=LLy;
        dx=ddx;
        dy=ddy;
        Real_Lx=Lx+0.5*dx;
        Real_Ly=Ly+0.5*dy;

    }

    il::StaticArray<double,2> Mesh::node(il::int_t k) { return node_[k]; }

    il::StaticArray<int,2> Mesh::coor(il::int_t k){
        int x=k/nx;
        int y=k-x*nx;
        il::StaticArray<int,2> Result;
        Result[0]=y;
        Result[1]=x;
        return Result;
    };


    int Mesh::nelts(){return node_.size();}

    Mesh create_Mesh(double Lx, double Ly,int nx, int ny){
        il::Array<il::StaticArray<double,2>> node{nx*ny};
        double dx=2*Lx/(nx-1);
        double dy=2*Ly/(ny-1);
        //dx and dy evaluate the size of each element
        for(int i=0;i<ny;i++){
           for(int j=0; j<nx;j++){

               node[i*nx+j][0]=-Lx+j*dx;
               node[i*nx+j][1]=-Ly+i*dy;
           }
        }
        Mesh mesh;
       mesh.set_values(node,nx,ny,Lx,Ly,dx,dy);

        return mesh;
    }

}