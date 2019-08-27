//
// Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
// Geo-Energy Laboratory, 2016-2019.  All rights reserved.
// See the LICENSE.TXT file for more details.
//
// Contributors:
// Weihan Zhang
// Fatima-Ezzahra Moukhtari
// Brice Lecampion

#include <src/AssemblyDDM.h>


namespace hfp3d {


    //de is the angle inclined between the crack plan and the isotropic plan,
    // which is defined in elasticity_kernel_integration.cpp
   extern double de;



    il::Array2D<double> basic_assembly(Mesh mesh){

        double dx=mesh.dx;
        double dy=mesh.dy;


        //nelts is the number of elements in the mesh
        int nelts=mesh.nelts();

          il::Array2D<double> kmat{3*nelts,3*nelts};
        il::StaticArray<double,3> xA,xB,xC,xD;
        xA[0]=-dx/2;xA[1]=dy/2;xA[2]=0;
        xB[0]=dx/2; xB[1]=dy/2;xB[2]=0;
        xC[0]=dx/2; xC[1]=-dy/2;xC[2]=0;
        xD[0]=-dx/2; xD[1]=-dy/2;xD[2]=0;

        il::StaticArray<double,3> local_point;

        il::StaticArray<double,3> relativeP;

        il::StaticArray2D<double,3,3> local_matrix;

        for(int j=0;j<nelts;j++){

            local_point[0]=mesh.node(j)[0];
            local_point[1]=mesh.node(j)[1];
            local_point[2]=0;


            for(int k=0;k<nelts;k++){

                relativeP[0]=mesh.node(k)[0]-local_point[0];
                relativeP[1]=mesh.node(k)[1]-local_point[1];
                relativeP[2]=0;

               local_matrix=Normal_Shear_Stress( xA, xB, xC, xD, relativeP);
                //influence de j a k
                kmat(3*k,3*j)=local_matrix(0,0);
                kmat(3*k,3*j+1)=local_matrix(0,1);
                kmat(3*k,3*j+2)=local_matrix(0,2);
                kmat(3*k+1,3*j)=local_matrix(1,0);
                kmat(3*k+1,3*j+1)=local_matrix(1,1);
                kmat(3*k+1,3*j+2)=local_matrix(1,2);
                kmat(3*k+2,3*j)=local_matrix(2,0);
                kmat(3*k+2,3*j+1)=local_matrix(2,1);
                kmat(3*k+2,3*j+2)=local_matrix(2,2);


            }

        }


     return kmat;
}

    il::Array2D<double> opening_assembly(Mesh mesh){
        double dx=mesh.dx;
        double dy=mesh.dy;

        int nelts=mesh.nelts();
        il::Array2D<double> kmat{nelts,nelts};
        il::StaticArray<double,3> xA,xB,xC,xD;
        xA[0]=-dx/2;xA[1]=dy/2;xA[2]=0;
        xB[0]=dx/2; xB[1]=dy/2;xB[2]=0;
        xC[0]=dx/2; xC[1]=-dy/2;xC[2]=0;
        xD[0]=-dx/2; xD[1]=-dy/2;xD[2]=0;

        il::StaticArray<double,3> local_point;

        il::StaticArray<double,3> relativeP;

        il::StaticArray2D<double,3,3> local_matrix;

        for(int j=0;j<nelts;j++){

            local_point[0]=mesh.node(j)[0];
            local_point[1]=mesh.node(j)[1];
            local_point[2]=0;


            for(int k=0;k<nelts;k++){

                relativeP[0]=mesh.node(k)[0]-local_point[0];
                relativeP[1]=mesh.node(k)[1]-local_point[1];
                relativeP[2]=0;

                local_matrix=Normal_Shear_Stress( xA, xB, xC, xD, relativeP);
                //influence de j a k

                kmat(k,j)=local_matrix(2,2);


            }

        }


        return kmat;
    }




    il::Array2D<double> simplified_opening_assembly(Mesh mesh){
        double dx=mesh.dx;
        double dy=mesh.dy;

        int nelts=mesh.nelts();

        il::Array2D<double> kmat{nelts,nelts};
        il::StaticArray<double,3> xA,xB,xC,xD;
        xA[0]=-dx/2;xA[1]=dy/2;xA[2]=0;
        xB[0]=dx/2; xB[1]=dy/2;xB[2]=0;
        xC[0]=dx/2; xC[1]=-dy/2;xC[2]=0;
        xD[0]=-dx/2; xD[1]=-dy/2;xD[2]=0;

        il::StaticArray<double,3> local_point;

        il::StaticArray<double,3> relativeP;



        il::Array<double> reference{nelts*nelts,0};


            local_point[0]=mesh.node(0)[0];
            local_point[1]=mesh.node(0)[1];
            local_point[2]=0;


            for(int k=0;k<nelts;k++){

                relativeP[0]=mesh.node(k)[0]-local_point[0];
                relativeP[1]=mesh.node(k)[1]-local_point[1];
                relativeP[2]=0;


                //influence de j a k


               reference[mesh.coor(k)[0]*nelts+mesh.coor(k)[1]]=Stress(2, xA, xB, xC, xD, relativeP)[2];


            }


        for(int j=0;j<nelts/2;j++){

            local_point[0]=mesh.node(j)[0];
            local_point[1]=mesh.node(j)[1];
            local_point[2]=0;


            for(int k=j;k<nelts-j;k++){

                relativeP[0]=mesh.node(k)[0]-local_point[0];
                relativeP[1]=mesh.node(k)[1]-local_point[1];
                relativeP[2]=0;

               int nn=abs(mesh.coor(k)[0]-mesh.coor(j)[0])*nelts+abs(mesh.coor(k)[1]-mesh.coor(j)[1]);

                kmat(k,j)=-reference[nn];
                kmat(j,k)=kmat(k,j);
              //  kmat(nelts-k-1,nelts-j-1)=kmat(k,j);

            }

        }


        // By the symetry

        for(int j=0;j<nelts;j++){

            for(int k=0;k<nelts-j;k++){
                 kmat(nelts-k-1,nelts-j-1)=kmat(k,j);
            }
        }

       kmat((nelts-1)/2,(nelts-1)/2)=kmat(0,0);
        return kmat;
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////
    il::Array2D<double> shearing_assembly(Mesh mesh){
        double dx=mesh.dx;
        double dy=mesh.dy;

        int nelts=mesh.nelts();

        il::Array2D<double> kmat{nelts,nelts};
        il::StaticArray<double,3> xA,xB,xC,xD;
        xA[0]=-dx/2;xA[1]=dy/2;xA[2]=0;
        xB[0]=dx/2; xB[1]=dy/2;xB[2]=0;
        xC[0]=dx/2; xC[1]=-dy/2;xC[2]=0;
        xD[0]=-dx/2; xD[1]=-dy/2;xD[2]=0;

        il::StaticArray<double,3> local_point;

        il::StaticArray<double,3> relativeP;



        il::Array<double> reference{nelts*nelts,0};


        local_point[0]=mesh.node(0)[0];
        local_point[1]=mesh.node(0)[1];
        local_point[2]=0;


        for(int k=0;k<nelts;k++){

            relativeP[0]=mesh.node(k)[0]-local_point[0];
            relativeP[1]=mesh.node(k)[1]-local_point[1];
            relativeP[2]=0;

            //local_matrix=Normal_Shear_Stress( xA, xB, xC, xD, relativeP);
            //influence de j a k


            //reference[mesh.coor(k)[0]*nelts+mesh.coor(k)[1]]=local_matrix(0,0);
            reference[mesh.coor(k)[0]*nelts+mesh.coor(k)[1]] =Stress(0, xA, xB, xC, xD, relativeP)[4];

        }
        for(int j=0;j<nelts/2;j++){

            local_point[0]=mesh.node(j)[0];
            local_point[1]=mesh.node(j)[1];
            local_point[2]=0;


            for(int k=j;k<nelts-j;k++){

                relativeP[0]=mesh.node(k)[0]-local_point[0];
                relativeP[1]=mesh.node(k)[1]-local_point[1];
                relativeP[2]=0;

                int nn=abs(mesh.coor(k)[0]-mesh.coor(j)[0])*nelts+abs(mesh.coor(k)[1]-mesh.coor(j)[1]);

                kmat(k,j)=reference[nn];
                kmat(j,k)=kmat(k,j);
                //  kmat(nelts-k-1,nelts-j-1)=kmat(k,j);

            }

        }


        // By the symetry

        for(int j=0;j<nelts;j++){




            for(int k=0;k<nelts-j;k++){




                kmat(nelts-k-1,nelts-j-1)=kmat(k,j);

            }

        }

        kmat((nelts-1)/2,(nelts-1)/2)=kmat(0,0);
        return kmat;
    }


    ////////////////////////////////////////////////////////////////////////////////////

    il::Array2D<double> perpendicular_opening_assembly(Mesh mesh){
        double dx=mesh.dx;
        double dy=mesh.dy;

        int nelts=mesh.nelts();

        il::Array2D<double> kmat{nelts,nelts};
        il::StaticArray<double,3> xA,xB,xC,xD;
        xA[0]=-dx/2;xA[1]=0;xA[2]=dy/2;
        xB[0]=dx/2; xB[1]=0;xB[2]=dy/2;
        xC[0]=dx/2; xC[1]=0;xC[2]=-dy/2;
        xD[0]=-dx/2; xD[1]=0;xD[2]=-dy/2;

        il::StaticArray<double,3> local_point;

        il::StaticArray<double,3> relativeP;


        il::Array<long double> reference{nelts*nelts,0};


        local_point[0]=mesh.node(0)[0];
        local_point[1]=0;
        local_point[2]=mesh.node(0)[1];


        for(int k=0;k<nelts;k++){

            relativeP[0]=mesh.node(k)[0]-local_point[0];
            relativeP[1]=0;
            relativeP[2]=mesh.node(k)[1]-local_point[2];

            //influence de j a k

            reference[mesh.coor(k)[0]*nelts+mesh.coor(k)[1]]= Stress(2, xA, xB, xC,
                                                                     xD,relativeP)[1];
        }

        for(int j=0;j<nelts/2;j++){

            local_point[0]=mesh.node(j)[0];
            local_point[1]=0;
            local_point[2]=mesh.node(j)[1];


            for(int k=j;k<nelts-j;k++){

                relativeP[0]=mesh.node(k)[0]-local_point[0];
                relativeP[1]=0;
                relativeP[2]=mesh.node(k)[1]-local_point[2];

                int nn=abs(mesh.coor(k)[0]-mesh.coor(j)[0])*nelts+abs(mesh.coor(k)[1]-mesh.coor(j)[1]);

                kmat(k,j)=-reference[nn];
                kmat(j,k)=kmat(k,j);
                //  kmat(nelts-k-1,nelts-j-1)=kmat(k,j);

            }

        }


        // By the symetry

        for(int j=0;j<nelts;j++){




            for(int k=0;k<nelts-j;k++){




                kmat(nelts-k-1,nelts-j-1)=kmat(k,j);

            }

        }

        kmat((nelts-1)/2,(nelts-1)/2)=kmat(0,0);
        return kmat;
    }



    //////////////////////////////////////////////////////////////////////////

    il::Array2D<double> simplified_opening_assembly_2(Mesh mesh){
        //dx and dy evaluate the size of each element
        double dx=mesh.dx;
        double dy=mesh.dy;

        //nelts is the number of elements of the mesh
        int nelts=mesh.nelts();


       //kmat is the global matrix
        il::Array2D<double> kmat{nelts,nelts};

        // four corner of the rectangular dislocation loop
        il::StaticArray<double,3> xA,xB,xC,xD;
        xA[0]=-dx/2;xA[1]=dy/2;xA[2]=0;
        xB[0]=dx/2; xB[1]=dy/2;xB[2]=0;
        xC[0]=dx/2; xC[1]=-dy/2;xC[2]=0;
        xD[0]=-dx/2; xD[1]=-dy/2;xD[2]=0;

        // the center of the first element in the mesh
        il::StaticArray<double,3> local_point;

        //the distance vector from the kth element to the first element in the mesh
        il::StaticArray<double,3> relativeP;

        //the 2D array to store the influence coefficient relating the first element and every other element in the mesh
        il::Array2D<double> reference{mesh.nx,mesh.ny};

        local_point[0]=mesh.node(0)[0];
        local_point[1]=mesh.node(0)[1];
        local_point[2]=0;

        for(int k=0;k<nelts;k++){

            relativeP[0]=mesh.node(k)[0]-local_point[0];
            relativeP[1]=mesh.node(k)[1]-local_point[1];
            relativeP[2]=0;

            //influence from j to k
            reference(mesh.coor(k)[0],mesh.coor(k)[1])=Stress(2, xA, xB, xC, xD, relativeP)[2];
        }



        //Full the global matrix by the coefficient in the 2D array reference
        for(int j=0;j<nelts/2;j++){

            for(int k=j;k<nelts-j;k++){

                int refx=abs(mesh.coor(k)[0]-mesh.coor(j)[0]);
                int refy=abs(mesh.coor(k)[1]-mesh.coor(j)[1]);
                kmat(k,j)=reference(refx,refy);
                kmat(j,k)=kmat(k,j);
                //  kmat(nelts-k-1,nelts-j-1)=kmat(k,j);
            }
        }



        // Full all the position of the global matrix by using the symetry
        for(int j=0;j<nelts;j++){
            for(int k=0;k<nelts-j;k++){
                kmat(nelts-k-1,nelts-j-1)=kmat(k,j);
            }
        }
        kmat((nelts-1)/2,(nelts-1)/2)=kmat(0,0);
        return kmat;
    }

/////////////////////////////////////////////////////////////////////////////////////////////////
    il::Array2D<double> perpendicular_opening_assembly_2(Mesh mesh){
        double dx=mesh.dx;
        double dy=mesh.dy;

        int nelts=mesh.nelts();

        il::Array2D<double> kmat{nelts,nelts};
        il::StaticArray<double,3> xA,xB,xC,xD;
        xA[0]=-dx/2;xA[1]=0;xA[2]=dy/2;
        xB[0]=dx/2; xB[1]=0;xB[2]=dy/2;
        xC[0]=dx/2; xC[1]=0;xC[2]=-dy/2;
        xD[0]=-dx/2; xD[1]=0;xD[2]=-dy/2;

        il::StaticArray<double,3> local_point;

        il::StaticArray<double,3> relativeP;


        il::Array2D<double> reference{mesh.nx,mesh.ny};


        local_point[0]=mesh.node(0)[0];
        local_point[1]=0;
        local_point[2]=mesh.node(0)[1];


        for(int k=0;k<nelts;k++){

            relativeP[0]=mesh.node(k)[0]-local_point[0];
            relativeP[1]=0;
            relativeP[2]=mesh.node(k)[1]-local_point[2];

            //influence de j a k

            reference(mesh.coor(k)[0],mesh.coor(k)[1])= Stress(2, xA, xB, xC,
                                                                     xD,relativeP)[1];
        }

        for(int j=0;j<nelts/2;j++){


            for(int k=j;k<nelts-j;k++){


                int refx=abs(mesh.coor(k)[0]-mesh.coor(j)[0]);
                int refy=abs(mesh.coor(k)[1]-mesh.coor(j)[1]);
                kmat(k,j)=reference(refx,refy);
                kmat(j,k)=kmat(k,j);
                //  kmat(nelts-k-1,nelts-j-1)=kmat(k,j);
            }
        }

        // By the symetry

        for(int j=0;j<nelts;j++){
            for(int k=0;k<nelts-j;k++){
                kmat(nelts-k-1,nelts-j-1)=kmat(k,j);
            }
        }
        kmat((nelts-1)/2,(nelts-1)/2)=kmat(0,0);
        return kmat;
    }
///////////////////////////////////////////////////////////////////////////////////////////////
    il::Array2D<double> make_vector(Mesh mesh){
        double dx=mesh.dx;
        double dy=mesh.dy;

        int nelts=mesh.nelts();


        il::StaticArray<double,3> xA,xB,xC,xD;
        xA[0]=-dx/2;xA[1]=dy/2*cos(de);xA[2]=dy/2*sin(de);
        xB[0]=dx/2; xB[1]=dy/2*cos(de);xB[2]=dy/2*sin(de);
        xC[0]=dx/2; xC[1]=-dy/2*cos(de);xC[2]=-dy/2*sin(de);
        xD[0]=-dx/2; xD[1]=-dy/2*cos(de);xD[2]=-dy/2*sin(de);

        il::StaticArray<double,3> local_point;

        il::StaticArray<double,3> relativeP;

        il::Array2D<double> reference{mesh.nx,mesh.ny};

        local_point[0]=mesh.node(0)[0];
        local_point[1]=mesh.node(0)[1]*cos(de);
        local_point[2]=mesh.node(0)[1]*sin(de);

        for(int k=0;k<nelts;k++){

            relativeP[0]=mesh.node(k)[0]-local_point[0];
            relativeP[1]=mesh.node(k)[1]*cos(de)-local_point[1];
            relativeP[2]=mesh.node(k)[1]*sin(de)-local_point[2];

            //influence de j a k
            reference(mesh.coor(k)[0],mesh.coor(k)[1])=Stress(2, xA, xB, xC, xD, relativeP)[2]*cos(de)
                                                       +Stress(2, xA, xB, xC, xD, relativeP)[1]*sin(de);
        }
        return reference;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    il::Array2D<double> make_vector_opening(Mesh mesh){
        double dx=mesh.dx;
        double dy=mesh.dy;

        int nelts=mesh.nelts();


        il::StaticArray<double,3> xA,xB,xC,xD;
        xA[0]=-dx/2;xA[1]=dy/2;xA[2]=0;
        xB[0]=dx/2; xB[1]=dy/2;xB[2]=0;
        xC[0]=dx/2; xC[1]=-dy/2;xC[2]=0;
        xD[0]=-dx/2; xD[1]=-dy/2;xD[2]=0;

        il::StaticArray<double,3> local_point;

        il::StaticArray<double,3> relativeP;

        il::Array2D<double> reference{mesh.nx,mesh.ny};

        local_point[0]=mesh.node(0)[0];
        local_point[1]=mesh.node(0)[1];
        local_point[2]=0;

        for(int k=0;k<nelts;k++){

            relativeP[0]=mesh.node(k)[0]-local_point[0];
            relativeP[1]=mesh.node(k)[1]-local_point[1];
            relativeP[2]=0;

            //influence de j a k
            reference(mesh.coor(k)[0],mesh.coor(k)[1])=Stress(2, xA, xB, xC, xD, relativeP)[2];
        }
        return reference;
    }


    ///////////////////////////////////////////////////////////////////////////////////////////////
    il::Array2D<double> make_vector_perp(Mesh mesh){
        double dx=mesh.dx;
        double dy=mesh.dy;

        int nelts=mesh.nelts();


        il::StaticArray<double,3> xA,xB,xC,xD;
        xA[0]=-dx/2;xA[1]=0;xA[2]=dy/2;
        xB[0]=dx/2; xB[1]=0;xB[2]=dy/2;
        xC[0]=dx/2; xC[1]=0;xC[2]=-dy/2;
        xD[0]=-dx/2; xD[1]=0;xD[2]=-dy/2;

        il::StaticArray<double,3> local_point;

        il::StaticArray<double,3> relativeP;

        il::Array2D<double> reference{mesh.nx,mesh.ny};

        local_point[0]=mesh.node(0)[0];
        local_point[1]=0;
        local_point[2]=mesh.node(0)[1];

        for(int k=0;k<nelts;k++){

            relativeP[0]=mesh.node(k)[0]-local_point[0];
            relativeP[1]=0;
            relativeP[2]=mesh.node(k)[1]-local_point[2];

            //influence de j a k
            reference(mesh.coor(k)[0],mesh.coor(k)[1])= Stress(2, xA, xB, xC, xD, relativeP)[1];
        }
        return reference;
    }
/////////////////////////////////////////////////////////////////////////////////////////////////////
    il::Array2D<double> make_matrix(Mesh mesh){
        int nelts=mesh.nelts();
        il::Array2D<double> kmat{nelts,nelts};
        il::Array2D<double> reference=make_vector(mesh);
        for(int j=0;j<nelts/2;j++){

            for(int k=j;k<nelts-j;k++){

                int refx=abs(mesh.coor(k)[0]-mesh.coor(j)[0]);
                int refy=abs(mesh.coor(k)[1]-mesh.coor(j)[1]);
                kmat(k,j)=reference(refx,refy);
                kmat(j,k)=kmat(k,j);
                //  kmat(nelts-k-1,nelts-j-1)=kmat(k,j);
            }
        }
        // By the symetry

        for(int j=0;j<nelts;j++){
            for(int k=0;k<nelts-j;k++){
                kmat(nelts-k-1,nelts-j-1)=kmat(k,j);
            }
        }
        kmat((nelts-1)/2,(nelts-1)/2)=kmat(0,0);
        return kmat;
    }
 ////////////////////////////////////////////////////////////////////////////////////////////////////

    double find_coeff (il::Array2D<double> reference,Mesh mesh,int k,int j){
        int refx=abs(mesh.coor(k)[0]-mesh.coor(j)[0]);
        int refy=abs(mesh.coor(k)[1]-mesh.coor(j)[1]);
        return reference(refx,refy);
    }

}