# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 17:49:21 2016

@author: Haseeb
"""
#import sys
#sys.path.append('./Classes')
#sys.path.append('./FractureMechanics')
#sys.path.append('./LevelSet')
#sys.path.append('./Utility')
#sys.path.append('./Solver')


#from importlib.machinery import SourceFileLoader
#
#foo = SourceFileLoader("CartesianMesh", "./Classes/CartesianMesh.py").load_module()
from CartesianMesh import *

#foo = SourceFileLoader("Fracture", "./Classes/Fracture.py").load_module()
from Fracture import *

#foo = SourceFileLoader("ElasticityKernel", "./FractureMechanics/ElasticityKernel.py").load_module()
from ElasticityKernel import *

#foo = SourceFileLoader("LevelSet", "./LevelSet/LevelSet.py").load_module()
from LevelSet import *

#foo = SourceFileLoader("VolIntegral", "./FractureMechanics/VolIntegral.py").load_module()
from VolIntegral import *

import numpy as np


K_Ic    = 1.4*1e6
Kprime  = 4 * (2/math.pi)**0.5 * K_Ic
Eprime  = 32*1e9
Q0      = 0.03
Cprime  = 0.00025
muPrime = 12*1e-3
sigma0  = 0

initRad     = 10
t0          = 15
tol_FrntPos = 0.001
tol_Picard  = 1e-5
Tend        = 300
Mesh        = CartesianMesh(30,30,40,40);
prntcount   = 0
timeout     = 0.15
first       = 1
#del Fr

Fr      = Fracture(Mesh,Eprime,Kprime,sigma0,Cprime,muPrime,Q0)
Fr.InitializeRadialFracture(t0,'time','Mt')
#Fr.PrintVariable('complete','width')

#pa          = max(Fr.p)
#sigmaNot    = 2e6*np.ones((Mesh.ny,Mesh.nx),float)
#sigmaNot[0:int(1*Mesh.ny/4),:] = 3.5e6
#sigmaNot[int(3*Mesh.ny/4)+1:Mesh.ny,:] = 3.5e6
#sigmaNot    = np.resize(sigmaNot,(Mesh.NumberOfElts,))
#Fr.sigma0   = sigmaNot

print('making global matrix')
#C       = ElasticityMatrixAllMesh(Mesh,Eprime);
print('Global matrix done')


#(EltsTipNew, Fr.l, Fr.alpha, Fr.CellStatus)= TrackFront(Fr.sgndDist, Fr.EltChannel, Fr.mesh)
#Fr.w[Fr.EltTip]   = VolumeIntegral(Fr.alpha, Fr.l, Fr.mesh.hx, Fr.mesh.hy, 2, Fr.Kprime[Fr.EltTip], Eprime, Fr.muPrime[Fr.EltTip], Fr.Cprime[Fr.EltTip], Fr.v)/Fr.mesh.hx/Fr.mesh.hy
#Fr.PrintVariable('complete','width')


while Fr.time<Tend:
    print('\n*********************\nAdvancing time step, time = ' + repr(Fr.time))
    
    dt = 0.5*Mesh.hx/np.mean(Fr.v)
    if dt<0:
        raise SystemExit('negative dt')

    Fr.Propagate(dt,C,tol_FrntPos,tol_Picard)
    

#R       = (3/2**0.5/math.pi * Q0*Eprime*T/Kprime)**0.4     # Toughness dominated
    R_Msol  = 0.6976*Fr.Eprime**(1/9)*sum(Fr.Q)**(1/3)*Fr.time**(4/9)/np.mean(Fr.muPrime)**(1/9)     #Viscocity dominated
    R_Mtsol = (2*sum(Fr.Q)/np.mean(Fr.Cprime))**0.5*Fr.time**0.25/np.pi
    if Fr.time//timeout>prntcount or first==1:
        Fr.PlotFracture('complete','footPrint')
        plt.pause(0.5)
        if first!=1:
            prntcount+=1
        first =0
    print('injected = '+repr(Q0*Fr.time)+' leaked off '+repr(sum(Fr.Leakedoff))+' in Fracture '+repr(Fr.mesh.EltArea*sum(Fr.w)))
    print('diff = '+repr(1-(sum(Fr.Leakedoff)+Fr.mesh.EltArea*sum(Fr.w))/(Q0*Fr.time)))
#    plt.close("all") 