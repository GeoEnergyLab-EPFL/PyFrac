# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Tue Dec 27 17:41:56 2016.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2017. All rights reserved.
See the LICENSE.TXT file for more details.
"""

import numpy as np
import pickle


def Kernel_ZZ(ax, ay, x, y, Ep):
    """
    Elasticity kernel (see e.g. Dontsov and Peirce, 2008)
    Arguments:
        ax (float): 
        ay (float): 
        x (float):
        y (float):
        Ep (float):     plain strain modulus
        
    Returns:
        float:          the influence weight of a cell on another (see e.g. Dontsov and Peirce, 2008)       
    """
    amx = ax - x
    apx = ax + x
    bmy = ay - y
    bpy = ay + y
    return (Ep / (8 * (np.pi))) * (
    np.sqrt(amx ** 2 + bmy ** 2) / (amx * bmy) + np.sqrt(apx ** 2 + bmy ** 2) / (apx * bmy) +
    np.sqrt(amx ** 2 + bpy ** 2) / (amx * bpy) + np.sqrt(apx ** 2 + bpy ** 2) / (apx * bpy))


#################################


def elasticity_matrix_all_mesh(Mesh, Ep):
    """
    Evaluate the elasticity matrix for the whole mesh
    Arguments:
        Mesh (object CartesianMesh):    a mesh object describing the domain 
        Ep (float):                     plain strain modulus
        
    Returns:
        ndarray-float:                  the elasticity martix
    """

    a = Mesh.hx / 2.
    b = Mesh.hy / 2.
    Ne = Mesh.NumberOfElts

    A = np.empty([Ne, Ne], dtype=float)

    for i in range(0, Ne):
        for j in range(0, Ne):
            x = Mesh.CenterCoor[i, 0] - Mesh.CenterCoor[j, 0]
            y = Mesh.CenterCoor[i, 1] - Mesh.CenterCoor[j, 1]
            amx = a - x
            apx = a + x
            bmy = b - y
            bpy = b + y
            # !!! Reconsider: Tried avoiding excessive function calls by embedding kernel here. No performance
            # improvement was observed.
            A[i, j] = (Ep / (8 * (np.pi))) * (
            np.sqrt(amx ** 2 + bmy ** 2) / (amx * bmy) + np.sqrt(apx ** 2 + bmy ** 2) / (apx * bmy) +
            np.sqrt(amx ** 2 + bpy ** 2) / (amx * bpy) + np.sqrt(apx ** 2 + bpy ** 2) / (apx * bpy))
            # A[i,j]=KernelZZ(a,b,x,y,Ep);

    return A


#################################

def load_elasticity_matrix(Mesh, EPrime):
    """
    The function loads the elasticity matrix(see e.g. Dontsov and Peirce 2008) from the saved file. If the loaded matrix
    is not compatible with respect to the current mesh or plain strain modulus, the compatible matrix is computed and
    saved in a file. If the file is not found, the elasticity matrix is computed and saved in a file with the name
    'CMatrix'.
    
    Arguments:
        Mesh (object CartesianMesh) : a mesh object describing the domain 
        EPrime (float)              : plain strain modulus
    
    Returns:
         (ndarray-float) : the elasticity matrix
    """
    print('Reading global elasticity matrix...')
    try:
        with open('CMatrix', 'rb') as input:
            (C, MeshLoaded, EPrimeLoaded) = pickle.load(input)
        # check if the loaded matrix is correct with respect to the current mesh and plain strain modulus
        if (Mesh.nx, Mesh.ny, Mesh.Lx, Mesh.Ly, EPrime) == (MeshLoaded.nx, MeshLoaded.ny, MeshLoaded.Lx,
                                                            MeshLoaded.Ly, EPrimeLoaded):
            return C
        else:
            print(
                'The loaded matrix is not correct with respect to the current mesh or the current plain strain modulus.'
                '\nMaking global matrix...')
            C = elasticity_matrix_all_mesh(Mesh, EPrime)
            Elast = (C, Mesh, EPrime)
            with open('CMatrix', 'wb') as output:
                pickle.dump(Elast, output, -1)
            return C
    except FileNotFoundError:
        # if 'CMatrix' file is not found
        print('file not found\nBuilding the global elasticity matrix...')
        C = elasticity_matrix_all_mesh(Mesh, EPrime)
        Elast = (C, Mesh, EPrime)
        with open('CMatrix', 'wb') as output:
            pickle.dump(Elast, output, -1)
        return C
