# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Thu Dec 22 17:18:37 2016.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2017. All rights reserved.
See the LICENSE.TXT file for more details.
"""

import numpy as np
from VolIntegral import Pdistance
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pickle

def RadiusLevelSet(xy,R) :
    """sign distance for a circle <0 inside circle , >0 outside, - zero at boundary"""
    if len(xy)>2:
        return np.linalg.norm(xy,2,1)-R     # norm(xy)=(x^2+y^2)^1/2    -R
    else:
        return np.linalg.norm(xy,2)-R

def Neighbors(elem,nx,ny) :
    """Neighbouring elements of an element. Boundary elements have themselves as neighbor"""

    j = elem//nx
    i = elem%nx

    if i==0 :
        left = elem
    else :
        left = j*nx+i-1

    if i== nx-1 :
        right = elem
    else :
        right = j*nx+i+1

    if j == 0 :
        bottom = elem
    else :
        bottom = (j-1)*nx+i

    if j == ny-1 :
        up = elem
    else :
        up = (j+1)*nx+i

    return (left,right,bottom,up)

def PrintDomain(Elem,Matrix,mesh):
    """
    3D plot of all elements given in the form of a list;
    Elem    -- list of elements
    Matrix  -- values to be ploted
    mesh    -- mesh object
    """

    temp = np.zeros((mesh.NumberOfElts,))
    temp[Elem] = Matrix
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(mesh.CenterCoor[:,0],mesh.CenterCoor[:,1],temp, cmap=cm.jet, linewidth=0.2)
    plt.show()
    plt.pause(0.0001)



##########################################

def PlotMeshFractureTrace(Mesh,EltTip,EltChannel,EltRibbon,I,J,Ranalytical,sigma,Identify) :
    """ Plot fracture trace and different regions of fracture"""
    
    import matplotlib
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    fig, ax = plt.subplots()
    #plt.scatter(CoorMid[:,0],CoorMid[:,1])
    ax.set_xlim([-Mesh.Lx, Mesh.Lx])
    ax.set_ylim([-Mesh.Ly, Mesh.Ly])

    patches = []

    for i in range(Mesh.NumberOfElts):
        polygon = Polygon(np.reshape(Mesh.VertexCoor[Mesh.Connectivity[i],:],(4,2)), True)
        patches.append(polygon)

    p = PatchCollection(patches,cmap=matplotlib.cm.jet, alpha=0.4)

    # colorsr = 100*np.random.rand(len(patches))
    EltHighSigma= np.where(sigma!=np.mean(sigma[Mesh.CenterElts]))
    
    colors =100.*np.full(len(patches),0.4)
    colors[EltHighSigma]=50
    colors[EltTip]=70.
    colors[EltChannel]=10.
    colors[EltRibbon]=90.
    colors[Identify]= 0.
    

    p.set_array(np.array(colors))
    ax.add_collection(p)

    #    # Plot the analytical solution

    if not np.isnan(Ranalytical):
        circle = plt.Circle((0, 0), radius=Ranalytical)
        circle.set_ec('r')
        circle.set_fill(False)
        ax.add_patch(circle)

    for i in range(len(Identify)):
        ax.text(Mesh.CenterCoor[Identify[i],0]-Mesh.hx/4,Mesh.CenterCoor[Identify[i],1]-Mesh.hy/4,repr(Identify[i]),fontsize=10)
        
    for e in range(0,len(I)) :
#        print(abs(max(I[e,:]-J[e,:])))
        if max(abs(I[e,:]-J[e,:]))<3*(Mesh.hx**2+Mesh.hy**2)**0.5: #sometimes get very large values, needs to be resolved
            plt.plot(np.array([I[e,0],J[e,0]]),np.array([I[e,1],J[e,1]]),'.-k')
    #        else:
    #            raise SystemExit('wrong front coordinate')
    plt.axis('equal')

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    return  plt.show()
    
######################################

def StressIntensityFactor(w,lvlSetData,EltTip,EltRibbon,stagnant,mesh,Eprime):
    """ See Donstov & Pierce Comput. Methods Appl. Mech. Engrn. 2017"""
    KIPrime = np.zeros((EltTip.size,),float)
    for i in range(0,len(EltTip)):
        if stagnant[i]:
            neighbors  = np.asarray(Neighbors(EltTip[i],mesh.nx,mesh.ny))
            enclosing  = np.append(neighbors,np.asarray([neighbors[2]-1,neighbors[2]+1,neighbors[3]-1,neighbors[3]+1])) # eight enclosing cells
            
            InRibbon = np.asarray([],int) #find neighbors in Ribbon cells
            for e in range(8):
                found = np.where(EltRibbon==enclosing[e])[0]
                if found.size>0:
                    InRibbon= np.append(InRibbon,EltRibbon[found[0]])

            if InRibbon.size==1:
                KIPrime[i] = w[InRibbon[0]]*Eprime/(-lvlSetData[InRibbon[0]])**0.5
            elif InRibbon.size>1: #evaluate using least squares method
                KIPrime[i] = Eprime*(w[InRibbon[0]]*(-lvlSetData[InRibbon[0]])**0.5 + w[InRibbon[1]]*(-lvlSetData[InRibbon[1]])**0.5)/(-lvlSetData[InRibbon[0]]-lvlSetData[InRibbon[1]])
            else: # ribbon cells not found in enclosure, evaluating with the closest ribbon cell
                RibbonCellsDist = ((mesh.CenterCoor[EltRibbon,0]-mesh.CenterCoor[EltTip[i],0])**2+(mesh.CenterCoor[EltRibbon,1]-mesh.CenterCoor[EltTip[i],1])**2)**0.5
                closest = EltRibbon[np.argmin(RibbonCellsDist)]
                KIPrime[i] = w[closest]*Eprime/(-lvlSetData[closest])**0.5

                
    return KIPrime
    
#######################################
def ReadFracture(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)