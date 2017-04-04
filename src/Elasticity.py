
# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Tue Dec 27 17:41:56 2016.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2017. All rights reserved.
See the LICENSE.TXT file for more details.
"""

import numpy as np
import pickle


# Elasticity kernel ....
#################################
def KernelZZ(ax,ay,x,y,Ep):
    amx=ax-x;
    apx=ax+x;
    bmy=ay-y;
    bpy =ay+y;
    return (Ep/(8*(np.pi)))*(np.sqrt(amx**2+bmy**2)/(amx*bmy)+np.sqrt(apx**2+bmy**2)/(apx*bmy) +
                             np.sqrt(amx**2+bpy**2)/(amx*bpy) +np.sqrt(apx**2+bpy**2)/(apx*bpy) )

#################################


def ElasticityMatrixAllMesh(Mesh,Ep):
    # Mesh
    # Ep : Plane Strain Modulus
    # Assemble the Elasticity matrix for the whole mesh

    a = Mesh.hx /2.;
    b= Mesh.hy /2. ;
    Ne=Mesh.NumberOfElts;
    
    A=np.empty([Ne, Ne],dtype=float, order='F');
    
    for i in range(0,Ne):
        for j in range (0,Ne):
            x=Mesh.CenterCoor[i,0]-Mesh.CenterCoor[j,0];
            y=Mesh.CenterCoor[i,1]-Mesh.CenterCoor[j,1];
            amx=a-x;
            apx=a+x;
            bmy=b-y;
            bpy=b+y;
            A[i,j]=(Ep/(8*(np.pi)))*(np.sqrt(amx**2+bmy**2)/(amx*bmy)+np.sqrt(apx**2+bmy**2)/(apx*bmy) +
                                     np.sqrt(amx**2+bpy**2)/(amx*bpy) +np.sqrt(apx**2+bpy**2)/(apx*bpy) )
#            A[i,j]=KernelZZ(a,b,x,y,Ep);
    
    return A
#################################

def LoadElastMatrix(Mesh,EPrime):

    print('Reading global elasticity matrix...')
    try:
        with open('CMatrix', 'rb') as input:
            (C,MeshLoaded,EPrimeLoaded) = pickle.load(input)
        if Mesh.nx==MeshLoaded.nx and  Mesh.ny==MeshLoaded.ny and Mesh.Lx==MeshLoaded.Lx and Mesh.Ly==MeshLoaded.Ly and EPrime==EPrimeLoaded:
            return C
        else:
            print('The loaded matrix is not correct w.r. to the current mesh \nMaking global matrix...')
            C = ElasticityMatrixAllMesh(Mesh,EPrime)
            Elast = (C,Mesh,EPrime)
            with open('CMatrix', 'wb') as output:
                pickle.dump(Elast, output, -1)
            return C
    except FileNotFoundError:
        print('file not found\n Building the global elasticity matrix...')
        C = ElasticityMatrixAllMesh(Mesh,EPrime)
        Elast = (C,Mesh,EPrime)
        with open('CMatrix', 'wb') as output:
            pickle.dump(Elast, output, -1)
        return C
