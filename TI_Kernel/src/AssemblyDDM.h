//
// Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
// Geo-Energy Laboratory, 2016-2022.  All rights reserved.
// See the LICENSE.TXT file for more details.
//
// Contributors:
// Weihan Zhang
// Fatima-Ezzahra Moukhtari
// Brice Lecampion
// Dmitry Nikolskiy

#ifndef HFPX3D_VC_ASSEMBLYDDM_H
#define HFPX3D_VC_ASSEMBLYDDM_H

#include<iostream>
#include <il/math.h>
#include<il/linearAlgebra.h>
#include <il/Array.h>
#include <il/Array2D.h>
#include "mesh.h"
#include "elasticity_kernel_integration.h"
//#include "inputE.h"


namespace hfp3d{

    //this script gives several ways to assembler the global influnece matrix
    //Some are just noted as a documentation, if the crack is in the isotropic plan then one
    //can use directly the function simplified_opening_assembly_2
    //if the crack is perpendicular to the isotropic plane then one can use the function
    //perpendicular_opening_assembly_2
    //Since those assemblage functions are quite similar, the comments details are only written for the function
    //simplified_opening_assembly_2 in the cpp file.
    //other functions like make_vector are explained below




    // basic_assembly returns the global matrix for the pre-mesh parallel to the isotropic plan
    // which contains the influence component of all three
    //directions, the size of the matrix is (3 nelts) times (3 nelts)
    il::Array2D<double> basic_assembly(Mesh mesh);


    //opening_assembly returns the global matrix for the pre-mesh parallel to the isotropic plan
    // which contains only the influence component of the opening mode
    // the size of the matrix is nelts times nelts
    il::Array2D<double> opening_assembly(Mesh mesh);


    //simplified_opening_assembly returns the same result of opening_assembly but it uses the toeplitz property
    //which reduces the cost of calculation
    il::Array2D<double> simplified_opening_assembly(Mesh mesh);


    //shearing_assembly returns the global matrix for the pre-mesh parallel to the isotropic plan
    // which contains only the influence component for a shear mode (dislocation in x direction and shear stress is sigma31)
    il::Array2D<double> shearing_assembly(Mesh mesh);


//perpendicular_opening_assembly returns the global matrix for the pre-mesh perpendicular to the isotropic plan
    // which contains only the influence component for opening mode.
    il::Array2D<double> perpendicular_opening_assembly(Mesh mesh);



    //similar to simplified_opening_assembly, but the influence coefficients for the first element in the mesh are
    //stored in a 2D array instead of 1D array. it reduces the memory from O(nelts times nelts) to O(nelts)
    il::Array2D<double> simplified_opening_assembly_2(Mesh mesh);


    //similar to perpendicular_opening_assembly, but the influence coefficients for the first element in the mesh are
    //stored in a 2D array instead of 1D array. it reduces the memory from O(nelts times nelts) to O(nelts)
    il::Array2D<double> perpendicular_opening_assembly_2(Mesh mesh);



///////////////////////////////////////////////////////////////////////////////

    //make_vector returns a 2D array which contains the influence coefficients for the first element
    // in the mesh for the general case where the pre-mesh is inclined to the isotropic plane with a certain
    //angle de.
    il::Array2D<double> make_vector(Mesh mesh);


    //particular case of make_vector
    //de=0 which signifies that the pre-mesh is parallel to the isotropic plan, which would be faster
    // then the make_vector
    il::Array2D<double> make_vector_opening(Mesh mesh);


    //particular case of make_vector
    //de=pi/2 which signifies that the pre-mesh is perpendicular to the isotropic plan, which would be faster
    // then the make_vector
    il::Array2D<double> make_vector_perp(Mesh mesh);


    //generate the global matrix in a general case by using the function make_vector, it returns the same matrix
    // as the function simplified_opening_assembly
    il::Array2D<double> make_matrix(Mesh mesh);


    // return the influence coefficient for the opening mode related to kth and jth elements in the mesh
    //where reference is generated by make_vector (_perp or _opening)
    double find_coeff (il::Array2D<double> reference,Mesh mesh,int k,int j);

}




#endif //HFPX3D_VC_ASSEMBLYDDM_H
