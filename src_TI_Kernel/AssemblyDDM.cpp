//
// Created by Student on 5/29/2017.
//

#include "AssemblyDDM.h"
#include <math.h>
#include <vector>
#include <iostream>
using namespace std;
namespace hfp3d {


    //de is the angle inclined between the crack plan and the isotropic plan,
    // which is defined in elasticity_kernel_integration.cpp
    extern double de;


    void basic_assembly(Mesh mesh, vector<vector<double>> res, double h, bool FS) {
        res.resize((int)3 * mesh.nelts());
        for (int i = 0; i < res.size(); i++) {
            res[i].resize(3 * mesh.nelts());
        }

        double d1 = mesh.d1;
        double d3 = mesh.d3;


        //nelts is the number of elements in the mesh
        int nelts = mesh.nelts();


        double xA[3], xB[3], xC[3], xD[3];
        xA[0] = -d1 / 2;
        xA[1] = d3 / 2;
        xA[2] = -h;
        xB[0] = d1 / 2;
        xB[1] = d3 / 2;
        xB[2] = -h;
        xC[0] = d1 / 2;
        xC[1] = -d3 / 2;
        xC[2] = -h;
        xD[0] = -d1 / 2;
        xD[1] = -d3 / 2;
        xD[2] = -h;

        double local_point[3];

        double relativeP[3];

        double local_matrix[3][3];

        for (int j = 0; j < nelts; j++) {
            double tmp_res1[2];
            mesh.get_node(j, tmp_res1);
            local_point[0] = tmp_res1[0];
            local_point[1] = tmp_res1[1];
            local_point[2] = 0;


            for (int k = 0; k < nelts; k++) {
                double tmp_res2[2];
                mesh.get_node(k, tmp_res2);
                relativeP[0] = tmp_res2[0] - local_point[0];
                relativeP[1] = tmp_res2[1] - local_point[1];
                relativeP[2] = 0;
                double local_matrix[3][3];
                Normal_Shear_Stress(xA, xB, xC, xD, relativeP, local_matrix,mesh.Free_surf_flag);
                //influence de j a k
                res[3 * k][3 * j] = local_matrix[0][0];
                res[3 * k][3 * j + 1] = local_matrix[0][1];
                res[3 * k][3 * j + 2] = local_matrix[0][2];
                res[3 * k + 1][3 * j] = local_matrix[1][0];
                res[3 * k + 1][3 * j + 1] = local_matrix[1][1];
                res[3 * k + 1][3 * j + 2] = local_matrix[1][2];
                res[3 * k + 2][3 * j] = local_matrix[2][0];
                res[3 * k + 2][3 * j + 1] = local_matrix[2][1];
                res[3 * k + 2][3 * j + 2] = local_matrix[2][2];


            }

        }

    }

    void opening_assembly(Mesh mesh, vector<vector<double>> res, double h, bool FS) {

        res.resize(mesh.nelts());
        for (int i = 0; i < res.size(); i++) {
            res[i].resize(mesh.nelts());
        }


        double d1 = mesh.d1;
        double d3 = mesh.d3;

        int nelts = mesh.nelts();
        double xA[3], xB[3], xC[3], xD[3];
        xA[0] = -d1 / 2;
        xA[1] = d3 / 2;
        xA[2] = -h;
        xB[0] = d1 / 2;
        xB[1] = d3 / 2;
        xB[2] = -h;
        xC[0] = d1 / 2;
        xC[1] = -d3 / 2;
        xC[2] = -h;
        xD[0] = -d1 / 2;
        xD[1] = -d3 / 2;
        xD[2] = -h;

        double local_point[3];

        double relativeP[3];

        double local_matrix[3][3];

        for (int j = 0; j < nelts; j++) {
            double tmp_res1[2];
            mesh.get_node(j, tmp_res1);
            local_point[0] = tmp_res1[0];
            local_point[1] = tmp_res1[1];
            local_point[2] = 0;


            for (int k = 0; k < nelts; k++) {
                double tmp_res2[2];
                mesh.get_node(k, tmp_res2);
                relativeP[0] = tmp_res2[0] - local_point[0];
                relativeP[1] = tmp_res2[1] - local_point[1];
                relativeP[2] = 0;
                double local_matrix[3][3];
                Normal_Shear_Stress(xA, xB, xC, xD, relativeP, local_matrix,mesh.Free_surf_flag);
                //influence de j a k

                res[k][j] = local_matrix[2][2];


            }

        }
    }


    vector<vector<double>> simplified_opening_assembly(Mesh mesh) {
        vector<vector<double>> res;
        res.resize(mesh.nelts());
        for (int i = 0; i < res.size(); i++) {
            res[i].resize(mesh.nelts());
        }

        double d1 = mesh.d1;
        double d3 = mesh.d3;

        int nelts = mesh.nelts();
        double xA[3], xB[3], xC[3], xD[3];
        xA[0] = -d1 / 2;
        xA[1] = d3 / 2;
        xA[2] = -mesh.Frac_depth;
        xB[0] = d1 / 2;
        xB[1] = d3 / 2;
        xB[2] = -mesh.Frac_depth;
        xC[0] = d1 / 2;
        xC[1] = -d3 / 2;
        xC[2] = -mesh.Frac_depth;
        xD[0] = -d1 / 2;
        xD[1] = -d3 / 2;
        xD[2] = -mesh.Frac_depth;

        double local_point[3];

        double relativeP[3];
        int n = nelts*nelts;
        for (int i = 0; i<nelts; i++) {

            double tmp_resi[2];
            mesh.get_node(i,tmp_resi);
            local_point[0] = tmp_resi[0];
            local_point[1] = tmp_resi[1];
            local_point[2] = -mesh.Frac_depth;


            for (int j = 0; j < nelts; j++) {
                double tmp_resj[2];
                mesh.get_node(j,tmp_resj);
                relativeP[0] = tmp_resj[0] - local_point[0];
                relativeP[1] = tmp_resj[1] - local_point[1];
                relativeP[2] = -mesh.Frac_depth;

                double StressRes[6] = {0,0,0,0,0,0};
                Stress(2,xA,xB,xC,xD,relativeP,StressRes,mesh.Free_surf_flag);
                res[i][j] = -StressRes[2];
                  // cout<<res[i][j]<<endl;

            };
        }
//        vector<double> reference;
//        reference.resize(n);
//
//        double tmp_res1[2];
//        mesh.get_node(0, tmp_res1);
//        local_point[0] = tmp_res1[0];
//        local_point[1] = tmp_res1[1];
//        local_point[2] = -mesh.Frac_depth;
//
//        for (int k = 0; k < nelts; k++) {
//            double tmp_res2[2];
//            mesh.get_node(k, tmp_res2);
//
//            relativeP[0] = tmp_res2[0] - local_point[0];
//            relativeP[1] = tmp_res2[1] - local_point[1];
//            relativeP[2] = -mesh.Frac_depth;
//
//
//            //influence de j a k
//            double tmp_res[6]={0,0,0,0,0,0};
//            Stress(2, xA, xB, xC, xD, relativeP, tmp_res, mesh.Free_surf_flag);
//            int Result[2];
//            mesh.coor(k, Result);
//            reference[Result[0] * nelts + Result[1]] = tmp_res[2];
//        }
//
//        for (int j = 0; j < nelts; j++) {
//            for (int k = 0; k < nelts; k++) {
//                int tmp_resk[2];
//                int tmp_resj[2];
//                mesh.coor(k,tmp_resk);
//                mesh.coor(j,tmp_resj);
//                int nn = abs(tmp_resk[0] - tmp_resj[0]) * nelts + abs(tmp_resk[1] - tmp_resj[1]);
////                int nn = j*nelts + k;
//                res[k][j] = -reference[nn];
//                res[j][k] = res[k][j];
//            }
//        }
//        for (int j = 0; j < nelts / 2; j++) {
//            double tmp_res2[2];
//            mesh.get_node(j, tmp_res2);
//            local_point[0] = tmp_res2[0];
//            local_point[1] = tmp_res2[1];
//            local_point[2] = 0;
//
//
//            for (int k = j; k < nelts - j; k++) {
//                double tmp_res3[2];
//                mesh.get_node(k, tmp_res3);
//                relativeP[0] = tmp_res3[0] - local_point[0];
//                relativeP[1] = tmp_res3[1] - local_point[1];
//                relativeP[2] = 0;
//                int tmp_resk[2];
//                int tmp_resj[2];
//                mesh.coor(k,tmp_resk);
//                mesh.coor(j,tmp_resj);
//                int nn = abs(tmp_resk[0] - tmp_resj[0]) * nelts + abs(tmp_resk[1] - tmp_resj[1]);
//
//                res[k][j] = -reference[nn];
//                res[j][k] = res[k][j];
//
//            }
//
//        }
//
//
        // By the symmetry
//
//        for (int j = 0; j < nelts; j++) {
//
//            for (int k = 0; k < nelts - j; k++) {
//                res[nelts - k - 1][nelts - j - 1] = res[k][j];
//            }
//        }
////
//        res[(nelts - 1) / 2][(nelts - 1) / 2] = res[0][0];
        return res;
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////
    void shearing_assembly(Mesh mesh, vector<vector<double>> res, double h, bool FS) {

        res.resize(mesh.nelts());
        for (int i = 0; i < res.size(); i++) {
            res[i].resize(mesh.nelts());
        }

        double d1 = mesh.d1;
        double d3 = mesh.d3;

        int nelts = mesh.nelts();

        double xA[3], xB[3], xC[3], xD[3];
        xA[0] = -d1 / 2;
        xA[1] = d3 / 2;
        xA[2] = -h;
        xB[0] = d1 / 2;
        xB[1] = d3 / 2;
        xB[2] = -h;
        xC[0] = d1 / 2;
        xC[1] = -d3 / 2;
        xC[2] = -h;
        xD[0] = -d1 / 2;
        xD[1] = -d3 / 2;
        xD[2] = -h;

        double local_point[3];

        double relativeP[3];

        vector<double> reference;
        reference.resize(nelts*nelts);

        double tmp_res1[2];
        mesh.get_node(0, tmp_res1);
        local_point[0] = tmp_res1[0];
        local_point[1] = tmp_res1[1];
        local_point[2] = 0;


        for (int k = 0; k < nelts; k++) {
            double tmp_res2[2];
            mesh.get_node(k, tmp_res2);
            relativeP[0] = tmp_res2[0] - local_point[0];
            relativeP[1] = tmp_res2[1] - local_point[1];
            relativeP[2] = 0;

            //local_matrix=Normal_Shear_Stress( xA, xB, xC, xD, relativeP);
            //influence de j a k


            //reference[mesh.coor(k)[0]*nelts+mesh.coor(k)[1]]=local_matrix(0,0);
            double tmp_res[6];
            Stress(2, xA, xB, xC, xD, relativeP, tmp_res, mesh.Free_surf_flag);
            int tmp_res3[2];
            mesh.coor(k, tmp_res3);
            reference[tmp_res3[0] * nelts + tmp_res3[1]] = tmp_res[4];

        }
        for (int j = 0; j < nelts / 2; j++) {
            double tmp_res2[2];
            mesh.get_node(j, tmp_res2);
            local_point[0] = tmp_res2[0];
            local_point[1] = tmp_res2[1];
            local_point[2] = 0;


            for (int k = j; k < nelts - j; k++) {
                double tmp_res3[2];
                mesh.get_node(k, tmp_res3);
                relativeP[0] = tmp_res3[0] - local_point[0];
                relativeP[1] = tmp_res3[1] - local_point[1];
                relativeP[2] = 0;
                int tmp_resk[2];
                int tmp_resj[2];
                mesh.coor(k, tmp_resk);
                mesh.coor(j, tmp_resj);
                int nn = abs(tmp_resk[0] - tmp_resj[0]) * nelts + abs(tmp_resk[1] - tmp_resj[1]);

                res[k][j] = reference[nn];
                res[j][k] = res[k][j];
                //  kmat(nelts-k-1,nelts-j-1)=kmat(k,j);

            }

        }


        // By the symetry

        for (int j = 0; j < nelts; j++) {


            for (int k = 0; k < nelts - j; k++) {


                res[nelts - k - 1][nelts - j - 1] = res[k][j];

            }

        }

        res[(nelts - 1) / 2][(nelts - 1) / 2] = res[0][0];
    }


    ////////////////////////////////////////////////////////////////////////////////////

    void perpendicular_opening_assembly(Mesh mesh, vector<vector<double>> res) {

        res.resize(mesh.nelts());
        for (int i = 0; i < res.size(); i++) {
            res[i].resize(mesh.nelts());
        }

        double d1 = mesh.d1;
        double d3 = mesh.d3;

        int nelts = mesh.nelts();

        double xA[3], xB[3], xC[3], xD[3];
        xA[0] = -d1 / 2;
        xA[1] = 0;
        xA[2] = d3 / 2;
        xB[0] = d1 / 2;
        xB[1] = 0;
        xB[2] = d3 / 2;
        xC[0] = d1 / 2;
        xC[1] = 0;
        xD[0] = -d1 / 2;
        xD[1] = 0;
        xD[2] = -d3 / 2;

        double local_point[3];

        double relativeP[3];


        vector<double> reference;
        reference.resize(nelts*nelts);

        double tmp_res1[2];
        mesh.get_node(0, tmp_res1);
        local_point[0] = tmp_res1[0];
        local_point[1] = 0;
        local_point[2] = tmp_res1[1];


        for (int k = 0; k < nelts; k++) {
            double tmp_res2[2];
            mesh.get_node(k, tmp_res2);
            relativeP[0] = tmp_res2[0] - local_point[0];
            relativeP[1] = 0;
            relativeP[2] = tmp_res2[1] - local_point[2];

            //influence de j a k
            double tmp_res[6];
            Stress(2, xA, xB, xC, xD, relativeP, tmp_res, mesh.Free_surf_flag);
            int tmp_res3[2];
            mesh.coor(k, tmp_res3);
            reference[tmp_res3[0] * nelts + tmp_res3[1]] = tmp_res[1];
        }

        for (int j = 0; j < nelts / 2; j++) {
            double tmp_res2[2];
            mesh.get_node(j, tmp_res2);
            local_point[0] = tmp_res2[0];
            local_point[1] = 0;
            local_point[2] = tmp_res2[1];


            for (int k = j; k < nelts - j; k++) {
                double tmp_res3[2];
                mesh.get_node(k, tmp_res3);
                relativeP[0] = tmp_res3[0] - local_point[0];
                relativeP[1] = 0;
                relativeP[2] = tmp_res3[1] - local_point[2];
                int tmp_resk[2];
                int tmp_resj[2];
                int nn = abs(tmp_resk[0] - tmp_resj[0]) * nelts + abs(tmp_resk[1] - tmp_resj[1]);

                res[k][j] = reference[nn];
                res[j][k] = res[k][j];
                //  kmat(nelts-k-1,nelts-j-1)=kmat(k,j);

            }

        }


        // By the symetry

        for (int j = 0; j < nelts; j++) {


            for (int k = 0; k < nelts - j; k++) {


                res[nelts - k - 1][nelts - j - 1] = res[k][j];

            }

        }

        res[(nelts - 1) / 2][(nelts - 1) / 2] = res[0][0];
    }



    //////////////////////////////////////////////////////////////////////////

    void simplified_opening_assembly_2(Mesh mesh, vector<vector<double>> res, double h,  bool FS) {

        res.resize(mesh.nelts());
        for (int i = 0; i < res.size(); i++) {
            res[i].resize(mesh.nelts());
        }

        //d1 and d3 evaluate the size of each element
        double d1 = mesh.d1;
        double d3 = mesh.d3;

        //nelts is the number of elements of the mesh
        int nelts = mesh.nelts();


        //kmat is the global matrix
        vector< vector<double>> kmat;
        kmat.resize(nelts);
        for (int i = 0; i<kmat.size(); i++){
            kmat[i].resize(nelts);
        }
        double xA[3], xB[3], xC[3], xD[3];
        xA[0] = -d1 / 2;
        xA[1] = d3 / 2;
        xA[2] = -h;
        xB[0] = d1 / 2;
        xB[1] = d3 / 2;
        xB[2] = -h;
        xC[0] = d1 / 2;
        xC[1] = -d3 / 2;
        xC[2] = -h;
        xD[0] = -d1 / 2;
        xD[1] = -d3 / 2;
        xD[2] = -h;

        // the center of the first element in the mesh
        double local_point[3];

        double relativeP[3];

        vector< vector<double>> reference;
        reference.resize(mesh.n1);
        for (int i = 0; i<reference.size(); i++){
            reference[i].resize(mesh.n3);
        };
        double tmp_res1[2];
        mesh.get_node(0, tmp_res1);
        local_point[0] = tmp_res1[0];
        local_point[1] = tmp_res1[1];
        local_point[2] = 0;

        for (int k = 0; k < nelts; k++) {
            double tmp_res2[2];
            mesh.get_node(k, tmp_res2);
            relativeP[0] = tmp_res2[0] - local_point[0];
            relativeP[1] = tmp_res2[1] - local_point[1];
            relativeP[2] = 0;

            //influence from j to k
            double tmp_res[6];
            Stress(2, xA, xB, xC, xD, relativeP, tmp_res, mesh.Free_surf_flag);
            int tmp_res3[2];
            mesh.coor(k, tmp_res3);
            reference[tmp_res3[0]][tmp_res3[1]] = tmp_res[2];
        }



        //Full the global matrix by the coefficient in the 2D array reference
        for (int j = 0; j < nelts / 2; j++) {
            int tmp_res2[2];
            mesh.coor(j, tmp_res2);
            for (int k = j; k < nelts - j; k++) {
                int tmp_res3[2];
                mesh.coor(k, tmp_res3);
                int refx = abs(tmp_res3[0] - tmp_res2[0]);
                int refy = abs(tmp_res3[1] - tmp_res2[1]);
                kmat[k][j] = reference[refx][refy];
                kmat[j][k] = kmat[k][j];
                //  kmat(nelts-k-1,nelts-j-1)=kmat(k,j);
            }
        }



        // Full all the position of the global matrix by using the symetry
        for (int j = 0; j < nelts; j++) {
            for (int k = 0; k < nelts - j; k++) {
                kmat[nelts - k - 1][nelts- j - 1] = kmat[k][j];
            }
        }
        kmat[(nelts - 1) / 2][(nelts - 1) / 2] = kmat[0][0];
    }

/////////////////////////////////////////////////////////////////////////////////////////////////
    void perpendicular_opening_assembly_2(Mesh mesh, vector<vector<double>> res, double h, bool FS) {

        res.resize(mesh.nelts());
        for (int i = 0; i < res.size(); i++) {
            res[i].resize(mesh.nelts());
        }

        double d1 = mesh.d1;
        double d3 = mesh.d3;

        int nelts = mesh.nelts();

        double xA[3], xB[3], xC[3], xD[3];
        xA[0] = -d1 / 2;
        xA[1] = -h;
        xA[2] = d3 / 2;
        xB[0] = d1 / 2;
        xB[1] = -h;
        xB[2] = d3 / 2;
        xC[0] = d1 / 2;
        xC[1] = -h;
        xC[2] = -d3 / 2;
        xD[0] = -d1 / 2;
        xD[1] = -h;
        xD[2] = -d3 / 2;

        double local_point[3];

        double relativeP[3];

        vector< vector<double>> reference;
        reference.resize(mesh.n1);
        for (int i = 0; i<reference.size(); i++){
            reference[i].resize(mesh.n3);
        };
        double tmp_res1[2];
        mesh.get_node(0, tmp_res1);

        local_point[0] = tmp_res1[0];
        local_point[1] = 0;
        local_point[2] = tmp_res1[1];


        for (int k = 0; k < nelts; k++) {
            double tmp_res2[2];
            mesh.get_node(k, tmp_res2);
            relativeP[0] = tmp_res2[0] - local_point[0];
            relativeP[1] = 0;
            relativeP[2] = tmp_res2[1] - local_point[2];

            //influence de j a k
            double tmp_res[6];
            Stress(2, xA, xB, xC, xD, relativeP, tmp_res, mesh.Free_surf_flag);
            int tmp_res3[2];
            mesh.coor(k, tmp_res3);
            reference[tmp_res3[0]][tmp_res3[1]] = tmp_res[1];
        }

        for (int j = 0; j < nelts / 2; j++) {
            int tmp_resj[2];
            mesh.coor(j, tmp_resj);

            for (int k = j; k < nelts - j; k++) {

                int tmp_resk[2];
                mesh.coor(k, tmp_resk);

                int refx = abs(tmp_resk[0] - tmp_resj[0]);
                int refy = abs(tmp_resk[1] - tmp_resj[1]);
                res[k][j] = reference[refx][refy];
                res[j][k] = res[k][j];
                //  kmat(nelts-k-1,nelts-j-1)=kmat(k,j);
            }
        }

        // By the symetry

        for (int j = 0; j < nelts; j++) {
            for (int k = 0; k < nelts - j; k++) {
                res[nelts - k - 1][nelts - j - 1] = res[k][j];
            }
        }
        res[(nelts - 1) / 2][(nelts - 1) / 2] = res[0][0];
    }

///////////////////////////////////////////////////////////////////////////////////////////////
    void make_vector(Mesh mesh, vector<vector<double>> res, double h, bool FS) {
        res.resize(mesh.n1);
        for (int i = 0; i < res.size(); i++) {
            res[i].resize(mesh.n3);
        }

        double d1 = mesh.d1;
        double d3 = mesh.d3;

        int nelts = mesh.nelts();


        double xA[3], xB[3], xC[3], xD[3];
        xA[0] = -d1 / 2;
        xA[1] = d3 / 2 * cos(de);
        xA[2] = -h +d3 / 2 * sin(de);
        xB[0] = d1 / 2;
        xB[1] = d3 / 2 * cos(de);
        xB[2] = -h + d3 / 2 * sin(de);
        xC[0] = d1 / 2;
        xC[1] = -d3 / 2 * cos(de);
        xC[2] = -h -d3 / 2 * sin(de);
        xD[0] = -d1 / 2;
        xD[1] = -d3 / 2 * cos(de);
        xD[2] = -h -d3 / 2 * sin(de);

        double local_point[3];

        double relativeP[3];
        double tmp_res1[2];
        mesh.get_node(0, tmp_res1);

        local_point[0] = tmp_res1[0];
        local_point[1] = tmp_res1[1] * cos(de);
        local_point[2] = tmp_res1[1] * sin(de);

        for (int k = 0; k < nelts; k++) {
            double tmp_res2[2];
            mesh.get_node(k, tmp_res2);
            relativeP[0] = tmp_res2[0] - local_point[0];
            relativeP[1] = tmp_res2[1] * cos(de) - local_point[1];
            relativeP[2] = tmp_res2[1] * sin(de) - local_point[2];

            //influence de j a k
            double tmp_res[6];
            Stress(2, xA, xB, xC, xD, relativeP, tmp_res, mesh.Free_surf_flag);
            int tmp_resk[2];
            mesh.coor(k, tmp_resk);
            res[tmp_resk[0]][tmp_resk[1]] = tmp_res[2] * cos(de)
                                            + tmp_res[1] * sin(de);
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    void make_vector_opening(Mesh mesh, vector<vector<double>> res, double h, bool FS) {
        res.resize(mesh.n1);
        for (int i = 0; i < res.size(); i++) {
            res[i].resize(mesh.n3);
        }
        double d1 = mesh.d1;
        double d3 = mesh.d3;

        int nelts = mesh.nelts();


        double xA[3], xB[3], xC[3], xD[3];
        xA[0] = -d1 / 2;
        xA[1] = d3 / 2;
        xA[2] = -h;
        xB[0] = d1 / 2;
        xB[1] = d3 / 2;
        xB[2] = -h;
        xC[0] = d1 / 2;
        xC[1] = -d3 / 2;
        xC[2] = -h;
        xD[0] = -d1 / 2;
        xD[1] = -d3 / 2;
        xD[2] = -h;

        double local_point[3];

        double relativeP[3];
        double tmp_res1[2];
        mesh.get_node(0, tmp_res1);

        local_point[0] = tmp_res1[0];
        local_point[1] = tmp_res1[1];
        local_point[2] = 0;

        for (int k = 0; k < nelts; k++) {
            double tmp_res2[2];
            mesh.get_node(k, tmp_res2);
            relativeP[0] = tmp_res2[0] - local_point[0];
            relativeP[1] = tmp_res2[1] - local_point[1];
            relativeP[2] = 0;

            //influence de j a k
            double tmp_res[6];
            Stress(2, xA, xB, xC, xD, relativeP, tmp_res, mesh.Free_surf_flag);
            int tmp_resk[2];
            mesh.coor(k, tmp_resk);
            res[tmp_resk[0]][tmp_resk[1]] = tmp_res[2];
        }
    }

}
    ////////////////////////////////////////////////////////////////////////////////////