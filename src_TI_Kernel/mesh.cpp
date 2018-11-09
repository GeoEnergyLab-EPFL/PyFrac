//
// Created by Student on 5/29/2017.
//

#include<iostream>
#include <complex>
#include <math.h>
#include "mesh.h"
#include <cmath>
#include <vector>
using namespace std;
namespace hfp3d{
    const double pi = 3.141592653589793238463;
    void Mesh::set_values(vector<vector<double>> xy,int nn1,int nn3,double LL1,double LL3,double dd1, double dd3,bool FS, double Depth) {
        node = xy;         // list of coordinates of points in the mesh
        n1=nn1;
        n3=nn3;
        L1=LL1;
        L3=LL3;
        d1=dd1;
        d3=dd3;
        Real_L1=L1+0.5*d1;
        Real_L3=L3+0.5*d3;
        Free_surf_flag = FS;
        Frac_depth=Depth;

    }

    void Mesh::get_node(int k, double res[2]) {
        for (int i = 0; i < 2; i++) {
            res[i] = node[k][i];
        }
    }
    void Mesh::coor(int k, int Result[2]){
        int y=k/n1;
        int x=k-y*n1;
        Result[0]=x;
        Result[1]=y;
    };


    int Mesh::nelts(){return node.size();}

    Mesh create_Mesh(double L1, double L3,int n1, int n3,bool FS,double Depth){
        vector<vector<double>> node;
        node.resize(n1*n3);
        for (int i = 0; i<node.size();i++){
            node[i].resize(2);
        }
        double d1=2*L1/(n1-1);
        double d3=2*L3/(n3-1);
        for(int i=0;i<n3;i++){
           for(int j=0; j<n1;j++){
               node[i*n1+j][0]=-L1+j*d1;
               node[i*n1+j][1]=-L3+i*d3;
           }
        }
        Mesh mesh;
       mesh.set_values(node,n1,n3,L1,L3,d1,d3,FS,Depth);

        return mesh;
    }

}