# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Tue Dec 27 19:01:22 2016.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2017. All rights reserved.
See the LICENSE.TXT file for more details.
"""

# imports
import numpy as np
#from importlib.machinery import SourceFileLoader

from src.Utility import Neighbors



#-----------------------------
def Eikonal_Res(Tij, *args): 
    """quadratic Eikonal equation residual to be used for numerical root finding solver"""

    (Tleft, Tright, Tbottom, Ttop, Fij, dx, dy) = args
    return np.nanmax([(Tij-Tleft)/dx, 0])**2 + np.nanmin([(Tright-Tij)/dx, 0])**2 + np.nanmax([(Tij-Tbottom)/dy, 0])**2 + \
           np.nanmin([(Ttop-Tij)/dy, 0])**2 - Fij**2

#################################    


def SolveFMM(T, EltRibbon, EltChannel,mesh):
    """solve Eikonal equation to get level set"""

    Alive   = EltChannel.astype(int)
    NarrowBand = EltRibbon.astype(int)
    FarAway = np.delete(range(mesh.NumberOfElts),np.intersect1d(range(mesh.NumberOfElts),EltChannel,None))
    FarAway = FarAway.astype(int)  
    maxdist = 4*(mesh.Lx**2+mesh.Ly**2)**0.5;
    
    while NarrowBand.size > 0 :
        
        SmallestT  = int(NarrowBand[T[NarrowBand.astype(int)].argmin()])
        neighbors  = np.asarray(Neighbors(SmallestT,mesh.nx,mesh.ny))
        
        for neighbor in neighbors:
            if not np.where(Alive==neighbor)[0].size > 0:
             
                if np.where(FarAway==neighbor)[0].size > 0:
                    NarrowBand  = np.append(NarrowBand, neighbor)
                    FarAway     = np.delete(FarAway,np.where(FarAway==neighbor))

                Stencil = np.asarray(Neighbors(neighbor,mesh.nx,mesh.ny)) 
            
                NeigxMin= min(T[Stencil[0]],T[Stencil[1]])
                NeigyMin= min(T[Stencil[2]],T[Stencil[3]])
                beta    = mesh.hx/mesh.hy
                delT    = NeigyMin - NeigxMin                    
                theta   = (mesh.hx**2*(1+beta**2) - beta**2*delT**2)**0.5
  
                if not np.isnan((NeigxMin + beta*NeigyMin + theta)/(1+beta**2)):
                    T[neighbor] = (NeigxMin + beta**2*NeigyMin + theta)/(1+beta**2)
                else:
                    if NeigxMin>maxdist: # used to check if very large value (level set value for uninitialized elements)
                        T[neighbor] = NeigyMin+mesh.hy
                    if NeigyMin>maxdist:
                        T[neighbor] = NeigxMin+mesh.hx    
                         #### numerical root finding    
#                            TSten   = np.empty(Stencil.shape)
#                            TSten[~np.isnan(Stencil)] = T[np.asarray(Stencil[~np.isnan(Stencil)],dtype=np.int)]
#                            TSten[np.isnan(Stencil)]  = 1000           
#                            Eikargs         = (TSten[0], TSten[1], TSten[2], TSten[3], 1, mesh.hx, mesh.hy)     #arguments for the eikinal equation function
#                            guess           = np.max(TSten)             # initial starting guess for the numerical solver
#                            T[neighbor]     = fsolve(Eikonal, guess, args=Eikargs)          #numerical solver
##                            print('Elem '+repr(neighbor)+' T= '+repr(T[neighbor]))
                            
        Alive      = np.append(Alive,SmallestT)
        NarrowBand = np.delete(NarrowBand,np.where(NarrowBand==SmallestT))
#    notEvaltd = np.where(T>maxdist)[0]
#    if notEvaltd.size>0:
#        for i in range(0,len(notEvaltd)):
#            T[notEvaltd[i]]=np.mean(T[np.asarray(Neighbors(notEvaltd[i],mesh.nx,mesh.ny))])


def TrackFront(dist, EltChannel, mesh):    # rename : it should be ReconstructFront
    """Track the fracture front, l and alpha from the Distances calculated with FMM"""
    
    EltRest = np.delete(range(mesh.NumberOfElts),np.intersect1d(range(mesh.NumberOfElts),EltChannel,None))
    ElmntTip= np.asarray([],int)
    l       = np.asarray([])
    alpha   = np.asarray([])     
    
    for i in range(0,len(EltRest)):
         neighbors  = np.asarray(Neighbors(EltRest[i],mesh.nx,mesh.ny))

         if not np.isnan(neighbors).any():
             minx = min(dist[neighbors[0]],dist[neighbors[1]])
             miny = min(dist[neighbors[2]],dist[neighbors[3]])
             Pdis = -(minx+miny)/2
             
             if Pdis >= 0:
                ElmntTip= np.append(ElmntTip,EltRest[i])
                l       = np.append(l,Pdis)
                
                # calculate angle imposed by the perpendicular on front (see Peirce & Detournay 2008)
                delDist = miny - minx
                beta    = mesh.hx/mesh.hy
                theta   = (mesh.hx**2*(1+beta**2) - beta**2*delDist**2)**0.5
                a1       = np.arccos((theta + beta**2*delDist)/(mesh.hx*(1+beta**2)))
                sinalpha= beta*(theta-delDist)/(mesh.hx*(1+beta**2))
                a2       = np.arcsin(sinalpha)
                # angle with tan
#                if minx<0 and miny<0:
#                a3       = np.arccos(abs(minx-miny)/(mesh.hx**2+mesh.hy**2)**0.5)-np.arctan(mesh.hy/mesh.hx)
           
                # checks to remove numerical noise in angle calculation
                if a2>=0 and a2<=np.pi/2:
                    alpha   = np.append(alpha, a2)
                elif a1>=0 and a1<=np.pi/2:
                    alpha   = np.append(alpha, a1)
                elif a2<0 and a2>-1e-6:
                    alpha   = np.append(alpha, 0)
                elif a2>np.pi/2 and a2<np.pi/2+1e-6:
                    alpha   = np.append(alpha, np.pi/2)
                elif a1<0 and a1>-1e-6:
                    alpha   = np.append(alpha, 0)
                elif a1>np.pi/2 and a1<np.pi/2+1e-6:
                    alpha   = np.append(alpha, np.pi/2)
                else:
                    alpha   = np.append(alpha, np.nan)
    
         
    CellStatusNew             = np.zeros((mesh.NumberOfElts),int)  
    CellStatusNew[EltChannel] = 1
    CellStatusNew[ElmntTip]   = 2
       

    return (ElmntTip, l, alpha, CellStatusNew)                
    

################################# 
# what is the goal of this one below....
def UpdateLists(EltsChannel, EltsTipNew, FillFrac, Dist, mesh):
    """Update the Element lists"""

    eltsTip     = EltsTipNew[np.where(FillFrac <= 0.999999)]
    inTip       = np.zeros((mesh.NumberOfElts,),bool)
    inTip[eltsTip] = True
#    Mod         = np.asarray([])
    i           = 0    
    while i<len(eltsTip): # to remove a special case encountered in sharp edges and rectangular cells
        neighbors  = np.asarray(Neighbors(eltsTip[i],mesh.nx,mesh.ny))
        if inTip[neighbors[0]] and inTip[neighbors[3]] and inTip[neighbors[3]-1]:
            conjoined = np.asarray([neighbors[0], neighbors[3], neighbors[3]-1,eltsTip[i]])
            mindist = np.argmin(mesh.distCenter[conjoined])
            inTip[conjoined[mindist]] = False            
            eltsTip = np.delete(eltsTip,np.where(eltsTip==conjoined[mindist]))
            i-=1
        i += 1
            
    eltnewTModT = np.copy(EltsTipNew) # new channel elements
    for i in range(0,len(eltsTip)):
        eltnewTModT = np.delete(eltnewTModT,np.where(eltnewTModT==eltsTip[i]))
        
    eltsChannel = np.append(EltsChannel, eltnewTModT)
    eltsCrack   = np.append(eltsChannel, eltsTip)
    eltsRibbon  = np.array([],int)    
    direction   = np.zeros((len(eltsTip),),int)
    
    for i in range(0,len(eltsTip)):
        neighbors  = np.asarray(Neighbors(eltsTip[i],mesh.nx,mesh.ny))
    
        if Dist[neighbors[0]]<= Dist[neighbors[1]]:
            eltsRibbon = np.append(eltsRibbon, neighbors[0])
            drctx = -1
        else:
            eltsRibbon = np.append(eltsRibbon, neighbors[1])
            drctx = 1
            
        if Dist[neighbors[2]]<= Dist[neighbors[3]]:
            eltsRibbon = np.append(eltsRibbon, neighbors[2])
            drcty = -1
        else:
            eltsRibbon = np.append(eltsRibbon, neighbors[3])
            drcty = 1
      
        if drctx<0 and drcty<0:
            direction[i] = 0
        if drctx>0 and drcty<0:
            direction[i] = 1
        if drctx<0 and drcty>0:
            direction[i] = 3
        if drctx>0 and drcty>0:
            direction[i] = 2

    eltsRibbon = np.unique(eltsRibbon)
    for i in range(0,len(eltsTip)):
        eltsRibbon = np.delete(eltsRibbon,np.where(eltsRibbon==eltsTip[i]))
    
    CellStatusNew             = np.zeros((mesh.NumberOfElts),int)  
    CellStatusNew[eltsChannel] = 1
    CellStatusNew[eltsTip]     = 2
    CellStatusNew[eltsRibbon]  = 3
    
    return (eltsChannel, eltsTip, eltsCrack, eltsRibbon, direction, CellStatusNew)
    
##################################                