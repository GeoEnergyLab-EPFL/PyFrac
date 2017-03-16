# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri Dec 23 17:49:21 2016.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2017. All rights reserved.
See the LICENSE.TXT file for more details.
"""
import sys
if not '.\\Utility' in sys.path:
    sys.path.append('.\\Utility')
if not '.\\Classes' in sys.path:
    sys.path.append('.\\Classes')
if not '.\\FractureMechanics' in sys.path:
    sys.path.append('.\\FractureMechanics')
if not '.\\LevelSet' in sys.path:
    sys.path.append('.\\LevelSet')
if not '.\\Solver' in sys.path:
    sys.path.append('.\\Solver')  



from CartesianMesh import *
from Fracture import *
from ElasticityKernel import *
from LevelSet import *
from VolIntegral import *
import numpy as np


K_Ic    = 0.005e6
Kprime  = 4 * (2/math.pi)**0.5 * K_Ic
nu      = 0.4
Eprime  = 3.3e10/(1-nu**2)
Q0      = 0.027
Cprime  = 0*0.00025
muPrime = 12*1.1e-3
rho     = 1000
sigma0  = 0*1e6

h           = 10
initRad     = 10
t0          = 40
tol_FrntPos = 0.001
tol_Picard  = 1e-5
Tend        = 1000
Mesh        = CartesianMesh(30,30,40,40);
prntcount   = 0
timeout     = 0
timeinc     = 10
first       = 1
CFL         = 0.6

minw        = np.asarray([])

Fr      = Fracture(Mesh,Eprime,Kprime,sigma0,Cprime,muPrime,rho,Q0)
Fr.InitializeRadialFracture(initRad,'radius','M')
#Fr.PlotFracture('complete','footPrint')
#plt.pause(1)
#Fr.InitializePKN(t0,0,h)
#Fr.PlotFracture('complete','footPrint',l_cr,evol=1)
timeout = Fr.time

#upLayer = np.where(Mesh.CenterCoor[:,0]>h/2)
#dnLayer = np.where(Mesh.CenterCoor[:,0]<-h/2)
#
#Fr.sigma0[upLayer]=1
#Fr.sigma0[dnLayer]=1
#Fr.Kprime[upLayer]=3e7
#Fr.Kprime[dnLayer]=3e7

#src = np.where(abs(Fr.mesh.CenterCoor[:,0])<Fr.mesh.hx)[0]
#src = src[np.where(abs(Fr.mesh.CenterCoor[src,1])<h/2-Fr.mesh.hy)[0]]
#Fr.Q= np.zeros((Fr.mesh.NumberOfElts),float)
#Fr.Q[src] = Q0/len(src)

#src = np.where(abs(Fr.mesh.CenterCoor[:,1])<Fr.mesh.hy)[0]
#src = src[np.where(abs(Fr.mesh.CenterCoor[src,0])<h/2-Fr.mesh.hx)[0]]
#Fr.Q= np.zeros((Fr.mesh.NumberOfElts),float)
#Fr.Q[src] = Q0/len(src)
C = LoadElastMatrix(Mesh,Eprime)

while Fr.time<Tend:
    print('\n*********************\ntime = ' + repr(Fr.time))
    Fr.Propagate(C,tol_FrntPos,tol_Picard,'M',CFL=0.6)
    R_Msol  = 0.6976*Fr.Eprime**(1/9)*sum(Fr.Q)**(1/3)*Fr.time**(4/9)/np.mean(Fr.muPrime)**(1/9)     #Viscocity dominated
#    R_Mtsol = (2*sum(Fr.Q)/np.mean(Fr.Cprime))**0.5*Fr.time**0.25/np.pi
#    R_Ksol  = (3/2**0.5/np.pi * Q0*Eprime*Fr.time/Kprime)**0.4 
    
    if Fr.time>timeout:
        print(repr(Fr.time//timeout)+' cnt '+repr(prntcount))
        l_cr    = (Eprime*Q0**3*Fr.time**4/(4*np.pi**3*(h+2*Mesh.hx)**4*(muPrime/12)))**(1/5)
        plt.close("all")
        status=Fr.PlotFracture('complete','footPrint',R_Msol)
#        Fr.SaveFracture('..\\StressJumpData\\Vertical2\\file'+repr(prntcount))
        prntcount+=1
        timeout+=timeinc
        
    fract   = np.where(Fr.w>1e-10)[0]
    print('injected = '+repr(Q0*Fr.time)+' leaked off '+repr(sum(Fr.Leakedoff))+' in Fracture '+repr(Fr.mesh.EltArea*sum(Fr.w)))
    print('diff = '+repr(1-(sum(Fr.Leakedoff)+Fr.mesh.EltArea*sum(Fr.w))/(Q0*Fr.time)))
    
     