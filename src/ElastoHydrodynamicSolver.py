# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Wed Dec 28 14:43:38 2016.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2017. All rights reserved.
See the LICENSE.TXT file for more details.
"""

import numpy as np
# import numdifftools as nd
import scipy.sparse.linalg as spla
from scipy.linalg import cho_factor
from scipy.linalg import cho_solve
import matplotlib.pyplot as plt

from src.Utility import *



def FiniteDiff_operator_laminar(w,EltCrack,muPrime,Mesh,InCrack):
    
    FinDiffOprtr    = np.zeros((w.size,w.size),dtype=np.float64)
    dx  = Mesh.hx
    dy  = Mesh.hy
    
    wLftEdge = (w[EltCrack]+w[Mesh.NeiElements[EltCrack,0]])/2*InCrack[Mesh.NeiElements[EltCrack,0]]
    wRgtEdge = (w[EltCrack]+w[Mesh.NeiElements[EltCrack,1]])/2*InCrack[Mesh.NeiElements[EltCrack,1]]
    wBtmEdge = (w[EltCrack]+w[Mesh.NeiElements[EltCrack,2]])/2*InCrack[Mesh.NeiElements[EltCrack,2]]
    wTopEdge = (w[EltCrack]+w[Mesh.NeiElements[EltCrack,3]])/2*InCrack[Mesh.NeiElements[EltCrack,3]]
        
    FinDiffOprtr[EltCrack,EltCrack] = -(wLftEdge**3+wRgtEdge**3)/dx**2/muPrime[EltCrack] - (wBtmEdge**3+wTopEdge**3)/dy**2/muPrime[EltCrack]
    FinDiffOprtr[EltCrack,Mesh.NeiElements[EltCrack,0]] = wLftEdge**3/dx**2/muPrime[EltCrack]
    FinDiffOprtr[EltCrack,Mesh.NeiElements[EltCrack,1]] = wRgtEdge**3/dx**2/muPrime[EltCrack]
    FinDiffOprtr[EltCrack,Mesh.NeiElements[EltCrack,2]] = wBtmEdge**3/dy**2/muPrime[EltCrack]
    FinDiffOprtr[EltCrack,Mesh.NeiElements[EltCrack,3]] = wTopEdge**3/dy**2/muPrime[EltCrack]
    
           
    return FinDiffOprtr


######################################

def FiniteDiff_operator_turbulent(w,EltCrack,muPrime,Mesh,InCrack,rho,vkm1,C,sigma0):
    
    
    FinDiffOprtr    = np.zeros((w.size,w.size),dtype=np.float64)
    dx      = Mesh.hx
    dy      = Mesh.hy
    mu      = muPrime/12     #### loosy should pass mu instead ?
    
    wLftEdge = (w[EltCrack]+w[Mesh.NeiElements[EltCrack,0]])/2
    wRgtEdge = (w[EltCrack]+w[Mesh.NeiElements[EltCrack,1]])/2
    wBtmEdge = (w[EltCrack]+w[Mesh.NeiElements[EltCrack,2]])/2
    wTopEdge = (w[EltCrack]+w[Mesh.NeiElements[EltCrack,3]])/2
    
    (dpdxLft,dpdxRgt,dpdyBtm,dpdyTop) = pressure_gradient(w,C,sigma0,Mesh,EltCrack,InCrack)

    ReLftEdge = 4/3 * rho*wLftEdge*(vkm1[0,EltCrack]**2+vkm1[4,EltCrack]**2)**0.5/mu[EltCrack]
    ReRgtEdge = 4/3 * rho*wRgtEdge*(vkm1[1,EltCrack]**2+vkm1[5,EltCrack]**2)**0.5/mu[EltCrack]
    ReBtmEdge = 4/3 * rho*wBtmEdge*(vkm1[2,EltCrack]**2+vkm1[6,EltCrack]**2)**0.5/mu[EltCrack]
    ReTopEdge = 4/3 * rho*wTopEdge*(vkm1[3,EltCrack]**2+vkm1[7,EltCrack]**2)**0.5/mu[EltCrack]

    maxRe = max(ReLftEdge)

    rough     = 10000*np.ones((EltCrack.size,),np.float64)
    ffLftEdge = FF_YangJoseph(ReLftEdge,rough)
    ffRgtEdge = FF_YangJoseph(ReRgtEdge,rough)
    ffBtmEdge = FF_YangJoseph(ReBtmEdge,rough)
    ffTopEdge = FF_YangJoseph(ReTopEdge,rough)

    # velocity current iteration, arrangement row wise: left x, right x, bottom y, top y, left y, right y, bottom x, top x
    vk      = np.zeros((8,Mesh.NumberOfElts),dtype=np.float64) 
    vk[0,EltCrack] = -wLftEdge/(rho*ffLftEdge*(vkm1[0,EltCrack]**2+vkm1[4,EltCrack]**2)**0.5)*dpdxLft
    vk[1,EltCrack] = -wRgtEdge/(rho*ffRgtEdge*(vkm1[1,EltCrack]**2+vkm1[5,EltCrack]**2)**0.5)*dpdxRgt
    vk[2,EltCrack] = -wBtmEdge/(rho*ffBtmEdge*(vkm1[2,EltCrack]**2+vkm1[6,EltCrack]**2)**0.5)*dpdyBtm
    vk[3,EltCrack] = -wTopEdge/(rho*ffTopEdge*(vkm1[3,EltCrack]**2+vkm1[7,EltCrack]**2)**0.5)*dpdyTop
    vk[0,np.where(np.isnan(vk[0,:]))]=0 # for edges adjacent to cells outside fracture
    vk[1,np.where(np.isnan(vk[1,:]))]=0
    vk[2,np.where(np.isnan(vk[2,:]))]=0
    vk[3,np.where(np.isnan(vk[3,:]))]=0
    
    vk[4,EltCrack] = (vk[2,Mesh.NeiElements[EltCrack,0]]+vk[3,Mesh.NeiElements[EltCrack,0]]+vk[2,EltCrack]+vk[3,EltCrack])/4
    vk[5,EltCrack] = (vk[2,Mesh.NeiElements[EltCrack,1]]+vk[3,Mesh.NeiElements[EltCrack,1]]+vk[2,EltCrack]+vk[3,EltCrack])/4
    vk[6,EltCrack] = (vk[0,Mesh.NeiElements[EltCrack,2]]+vk[1,Mesh.NeiElements[EltCrack,2]]+vk[0,EltCrack]+vk[1,EltCrack])/4
    vk[7,EltCrack] = (vk[0,Mesh.NeiElements[EltCrack,3]]+vk[1,Mesh.NeiElements[EltCrack,3]]+vk[0,EltCrack]+vk[1,EltCrack])/4

    ReLftEdge = 4/3 * rho*wLftEdge*(vk[0,EltCrack]**2+vk[4,EltCrack]**2)**0.5/mu[EltCrack]
    ReRgtEdge = 4/3 * rho*wRgtEdge*(vk[1,EltCrack]**2+vk[5,EltCrack]**2)**0.5/mu[EltCrack]
    ReBtmEdge = 4/3 * rho*wBtmEdge*(vk[2,EltCrack]**2+vk[6,EltCrack]**2)**0.5/mu[EltCrack]
    ReTopEdge = 4/3 * rho*wTopEdge*(vk[3,EltCrack]**2+vk[7,EltCrack]**2)**0.5/mu[EltCrack]
    
    ffLftEdge = FF_YangJoseph(ReLftEdge,rough)
    ffRgtEdge = FF_YangJoseph(ReRgtEdge,rough)
    ffBtmEdge = FF_YangJoseph(ReBtmEdge,rough)
    ffTopEdge = FF_YangJoseph(ReTopEdge,rough)
    

    ffLftEdge[np.where(np.isinf(ffLftEdge))]=0 # for edges adjacent to cells outside fracture
    ffRgtEdge[np.where(np.isinf(ffRgtEdge))]=0
    ffBtmEdge[np.where(np.isinf(ffBtmEdge))]=0
    ffTopEdge[np.where(np.isinf(ffTopEdge))]=0
    
   
    cond  = np.zeros((4,EltCrack.size),dtype=np.float64)
    cond[0,:]=wLftEdge**2/(rho*ffLftEdge*(vk[0,EltCrack]**2+vk[4,EltCrack]**2)**0.5)
    cond[1,:]=wRgtEdge**2/(rho*ffRgtEdge*(vk[1,EltCrack]**2+vk[5,EltCrack]**2)**0.5)
    cond[2,:]=wBtmEdge**2/(rho*ffBtmEdge*(vk[2,EltCrack]**2+vk[6,EltCrack]**2)**0.5)
    cond[3,:]=wTopEdge**2/(rho*ffTopEdge*(vk[3,EltCrack]**2+vk[7,EltCrack]**2)**0.5)
    
    cond[0,np.where(np.isinf(cond[0,:]))]=0
    cond[1,np.where(np.isinf(cond[1,:]))]=0
    cond[2,np.where(np.isinf(cond[2,:]))]=0
    cond[3,np.where(np.isinf(cond[3,:]))]=0    

   
    FinDiffOprtr[EltCrack,EltCrack] = -(cond[0,:] + cond[1,:])/dx**2 - (cond[2,:] + cond[3,:])/dy**2
    FinDiffOprtr[EltCrack,Mesh.NeiElements[EltCrack,0]] = cond[0,:]/dx**2
    FinDiffOprtr[EltCrack,Mesh.NeiElements[EltCrack,1]] = cond[1,:]/dx**2
    FinDiffOprtr[EltCrack,Mesh.NeiElements[EltCrack,2]] = cond[2,:]/dy**2
    FinDiffOprtr[EltCrack,Mesh.NeiElements[EltCrack,3]] = cond[3,:]/dy**2
    
   
    return (FinDiffOprtr,vk)


######################################
# ARHHHHHHH DON T USE pointers !!!!!!
def MakeEquationSystemExtendedFP(solk,vkm1,*args):

    (EltChannel, EltsTipNew, wLastTS, wTip, EltCrack, Mesh, dt, Q, C, muPrime, rho, InCrack, LeakOff, sigma0, turb) = args
    
    Ccc = C[np.ix_(EltChannel,EltChannel)]
    Cct = C[np.ix_(EltChannel,EltsTipNew)]
    
    A   = np.zeros((EltChannel.size+EltsTipNew.size,EltChannel.size+EltsTipNew.size),dtype=np.float64)
    S   = np.zeros((EltChannel.size+EltsTipNew.size,),dtype=np.float64)

    delwK = solk[np.arange(EltChannel.size)]
    wcNplusOne              = np.copy(wLastTS)
    wcNplusOne[EltChannel]  = wcNplusOne[EltChannel]+delwK
    wcNplusOne[EltsTipNew]  = wTip

    if turb:
        (FinDiffOprtr,vk)    = FiniteDiff_operator_turbulent(wcNplusOne,EltCrack,muPrime,Mesh,InCrack,rho,vkm1,C,sigma0)
    else:
        FinDiffOprtr  = FiniteDiff_operator_laminar(wcNplusOne,EltCrack,muPrime,Mesh,InCrack)
        vk = vkm1
        
    condCC  = FinDiffOprtr[np.ix_(EltChannel,EltChannel)]
    condCT  = FinDiffOprtr[np.ix_(EltChannel,EltsTipNew)]
    condTC  = FinDiffOprtr[np.ix_(EltsTipNew,EltChannel)]
    condTT  = FinDiffOprtr[np.ix_(EltsTipNew,EltsTipNew)]    
    
    Channel = np.arange(EltChannel.size)
    Tip     = Channel.size+np.arange(EltsTipNew.size)
    
    A[np.ix_(Channel,Channel)]   = np.identity(Channel.size) - dt*np.dot(condCC,Ccc)
    A[np.ix_(Channel,Tip)]       = -dt*condCT
    A[np.ix_(Tip,Channel)]       = -dt*np.dot(condTC,Ccc)
    A[np.ix_(Tip,Tip)]           = -dt*condTT

   
    S[Channel]  = dt*np.dot(condCC,np.dot(Ccc,wLastTS[EltChannel])+np.dot(Cct,wTip)+sigma0[EltChannel]) + dt/Mesh.hx/Mesh.hy*Q[EltChannel] \
                  - LeakOff[EltChannel]/Mesh.hx/Mesh.hy
    S[Tip]      = -(wTip-wLastTS[EltsTipNew]) + dt*np.dot(condTC,np.dot(Ccc,wLastTS[EltChannel])+np.dot(Cct,wTip)+sigma0[EltChannel]) \
                  - LeakOff[EltsTipNew]/Mesh.hx/Mesh.hy
    
    return (A,S,vk)

######################################
#
def MakeEquationSystemSameFP(delwk,vkm1,*args):
    (w, EltCrack, Q, C, dt, muPrime, mesh, InCrack, LeakOff, sigma0, rho, turb) = args
    wnPlus1 = np.copy(w)
    wnPlus1[EltCrack] = wnPlus1[EltCrack]+delwk

    if turb:
        (con,vk)= FiniteDiff_operator_turbulent(wnPlus1,EltCrack,muPrime,mesh,InCrack, rho, vkm1, C, sigma0)
    else:
        con = FiniteDiff_operator_laminar(wnPlus1, EltCrack, muPrime, mesh, InCrack)
        vk  = vkm1
    con     = con[np.ix_(EltCrack,EltCrack)]
    CCrack  = C[np.ix_(EltCrack,EltCrack)]
    
    A = np.identity(EltCrack.size)-dt*np.dot(con,CCrack)
    S = dt*np.dot(con,np.dot(CCrack,w[EltCrack])+sigma0[EltCrack]) + dt/mesh.EltArea*Q[EltCrack] - LeakOff[EltCrack]/mesh.EltArea
    return (A,S,vk)

#######################################

def Elastohydrodynamic_ResidualFun_sameFP(solk, interItr,*args):

    (A, S, vk) = MakeEquationSystemSameFP(solk,interItr,*args)
    return (np.dot(A,solk)-S,vk)

#######################################

def velocity(w,EltCrack,Mesh,InCrack,muPrime,C,sigma0):
    
    (dpdxLft,dpdxRgt,dpdyBtm,dpdyTop)=pressure_gradient(w,C,sigma0,Mesh,EltCrack,InCrack)
    
    vel = np.zeros((8,Mesh.NumberOfElts),dtype=np.float64)
    vel[0,EltCrack] = -((w[EltCrack]+w[Mesh.NeiElements[EltCrack,0]])/2)**2/muPrime[EltCrack]*dpdxLft
    vel[1,EltCrack] = -((w[EltCrack]+w[Mesh.NeiElements[EltCrack,1]])/2)**2/muPrime[EltCrack]*dpdxRgt
    vel[2,EltCrack] = -((w[EltCrack]+w[Mesh.NeiElements[EltCrack,2]])/2)**2/muPrime[EltCrack]*dpdyBtm
    vel[3,EltCrack] = -((w[EltCrack]+w[Mesh.NeiElements[EltCrack,3]])/2)**2/muPrime[EltCrack]*dpdyTop

    vel[4,EltCrack] = (vel[2,Mesh.NeiElements[EltCrack,0]]+vel[3,Mesh.NeiElements[EltCrack,0]]+vel[2,EltCrack]+vel[3,EltCrack])/4
    vel[5,EltCrack] = (vel[2,Mesh.NeiElements[EltCrack,1]]+vel[3,Mesh.NeiElements[EltCrack,1]]+vel[2,EltCrack]+vel[3,EltCrack])/4
    vel[6,EltCrack] = (vel[0,Mesh.NeiElements[EltCrack,2]]+vel[1,Mesh.NeiElements[EltCrack,2]]+vel[0,EltCrack]+vel[1,EltCrack])/4
    vel[7,EltCrack] = (vel[0,Mesh.NeiElements[EltCrack,3]]+vel[1,Mesh.NeiElements[EltCrack,3]]+vel[0,EltCrack]+vel[1,EltCrack])/4

    return vel

#######################################

def pressure_gradient(w,C,sigma0,Mesh,EltCrack,InCrack):
    
    pf          = np.zeros((Mesh.NumberOfElts,),dtype=np.float64)
    pf[EltCrack]= np.dot(C[np.ix_(EltCrack,EltCrack)],w[EltCrack]) + sigma0[EltCrack]
        
    dpdxLft     = (pf[EltCrack]-pf[Mesh.NeiElements[EltCrack,0]])*InCrack[Mesh.NeiElements[EltCrack,0]]
    dpdxRgt     = (pf[Mesh.NeiElements[EltCrack,1]]-pf[EltCrack])*InCrack[Mesh.NeiElements[EltCrack,1]]
    dpdyBtm     = (pf[EltCrack]-pf[Mesh.NeiElements[EltCrack,2]])*InCrack[Mesh.NeiElements[EltCrack,2]]
    dpdyTop     = (pf[Mesh.NeiElements[EltCrack,3]]-pf[EltCrack])*InCrack[Mesh.NeiElements[EltCrack,3]]
    
    return (dpdxLft,dpdxRgt,dpdyBtm,dpdyTop)

#######################################
#  in the future the following should be move to a separate files containing the fluid models....
def FF_YangJoseph(ReNum,rough):
    
    ff = np.full((len(ReNum),),np.inf,dtype=np.float64)

    lam = np.where(abs(ReNum)<2100)[0]
    ff[lam] = 16/ReNum[lam]

    turb = np.where(abs(ReNum)>=2100)[0]
    lamdaS = (-((-64/ReNum[turb] + 0.000083*ReNum[turb]**0.75)/(1 + 2320**50/ReNum[turb]**50)**0.5) - 64/ReNum[turb] + 0.3164/ReNum[turb]**0.25)/(1 + 3810**15/ReNum[turb]**15)**0.5 + (-((-((-64/ReNum[turb] + 0.000083*ReNum[turb]**0.75)/(1 + 2320**50/ReNum[turb]**50)**0.5) - 64/ReNum[turb] + 0.3164/ReNum[turb]**0.25)/(1 + 3810**15/ReNum[turb]**15)**0.5) - (-64/ReNum[turb] + 0.000083*ReNum[turb]**0.75)/(1 + 2320**50/ReNum[turb]**50)**0.5 - 64/ReNum[turb] + 0.1537/ReNum[turb]**0.185)/(1 + 1680700000000000000000000/ReNum[turb]**5)**0.5 + (-((-((-64/ReNum[turb] + 0.000083*ReNum[turb]**0.75)/(1 + 2320**50/ReNum[turb]**50)**0.5) - 64/ReNum[turb] + 0.3164/ReNum[turb]**0.25)/(1 + 3810**15/ReNum[turb]**15)**0.5) - (-((-((-64/ReNum[turb] + 0.000083*ReNum[turb]**0.75)/(1 + 2320**50/ReNum[turb]**50)**0.5) - 64/ReNum[turb] + 0.3164/ReNum[turb]**0.25)/(1 + 3810**15/ReNum[turb]**15)**0.5) - (-64/ReNum[turb] + 0.000083*ReNum[turb]**0.75)/(1 + 2320**50/ReNum[turb]**50)**0.5 - 64/ReNum[turb] + 0.1537/ReNum[turb]**0.185)/(1 + 1680700000000000000000000/ReNum[turb]**5)**0.5 - (-64/ReNum[turb] + 0.000083*ReNum[turb]**0.75)/(1 + 2320**50/ReNum[turb]**50)**0.5 - 64/ReNum[turb] + 0.0753/ReNum[turb]**0.136)/(1 + 4000000000000/ReNum[turb]**2)**0.5 + (-64/ReNum[turb] + 0.000083*ReNum[turb]**0.75)/(1 + 2320**50/ReNum[turb]**50)**0.5 + 64/ReNum[turb]
    lamdaR = ReNum[turb]**(-0.2032 + 7.348278/rough[turb]**0.96433953)*(-0.022 + (-0.978 + 0.92820419*rough[turb]**0.03569244 - 0.00255391*rough[turb]**0.8353877)/(1 + 265550686013728218770454203489441165109061383639474724663955742569518708077419167245843482753466249/rough[turb]**50)**0.5 + 0.00255391*rough[turb]**0.8353877) + (-(ReNum[turb]**(-0.2032 + 7.348278/rough[turb]**0.96433953)*(-0.022 + (-0.978 + 0.92820419*rough[turb]**0.03569244 - 0.00255391*rough[turb]**0.8353877)/(1 + 265550686013728218770454203489441165109061383639474724663955742569518708077419167245843482753466249/rough[turb]**50)**0.5 + 0.00255391*rough[turb]**0.8353877)) + 0.01105244*ReNum[turb]**(-0.191 + 0.62935712/rough[turb]**0.28022284)*rough[turb]**0.23275646 + (ReNum[turb]**(0.015 + 0.26827956/rough[turb]**0.28852025)*(0.0053 + 0.02166401/rough[turb]**0.30702955) - 0.01105244*ReNum[turb]**(-0.191 + 0.62935712/rough[turb]**0.28022284)*rough[turb]**0.23275646 + (ReNum[turb]**0.002*(0.011 + 0.18954211/rough[turb]**0.510031) - ReNum[turb]**(0.015 + 0.26827956/rough[turb]**0.28852025)*(0.0053 + 0.02166401/rough[turb]**0.30702955) + (0.0098 - ReNum[turb]**0.002*(0.011 + 0.18954211/rough[turb]**0.510031) + 0.17805185/rough[turb]**0.46785053)/(1 + (8.733801045300249e10*rough[turb]**0.90870686)/ReNum[turb]**2)**0.5)/(1 + (6.44205549308073e15*rough[turb]**5.168887)/ReNum[turb]**5)**0.5)/(1 + (1.1077593467238922e13*rough[turb]**4.9771653)/ReNum[turb]**5)**0.5)/(1 + (2.9505925619934144e14*rough[turb]**3.7622822)/ReNum[turb]**5)**0.5
    ff[turb] = np.asarray(lamdaS + (lamdaR-lamdaS)/(1+(ReNum[turb]/(45.196502*rough[turb]**1.2369807+1891))**-5)**0.5,float)/4
    return ff

#######################################

def Elastohydrodynamic_ResidualFun_ExtendedFP(solk, interItr,*args):

    (A, S, vk) = MakeEquationSystemExtendedFP(solk,interItr,*args)
    return (np.dot(A,solk)-S,vk)

#######################################

def Picard_Newton(Res_fun, sys_fun, guess, TypValue, interItr, relax, Tol, maxitr, *args):
    """
    Mixed Picard Newton solver for nonlinear systems.
        
    :param Res_fun: The function calculating the residual
    :param sys_fun: The function giving the system A,b for the Picard solver to solve the linear system of the form Ax=b
    :param guess:   The initial guess
    :param TypValue:Typical value of the variable to estimate the Epsilon to calculate jacobian
    :param interItr:Initial value of the variable exchanged between the iterations, if any 
    :param relax:   The relaxation factor
    :param Tol:     Tolerance
    :param maxitr:  Maximum number of iterations
    :param args:    arguments given to the residual and systems functions
    :return:        solution
    """
    solk = guess
    k = 1
    norm = 1
    normlist = np.ones((maxitr,), float)

    tryNewton = False

    newton = 0

    while norm > Tol:

        solkm1 = solk
        if k % 100 == 0 or tryNewton:
            (Fx,interItr) = Res_fun(solk,interItr, *args)
            if newton %3 == 0:
                Jac = Jacobian(Res_fun, solk, interItr, TypValue, *args)
            dx = np.linalg.solve(Jac, -Fx)
            solk = solkm1 + dx
            newton += 1
        else:
            # try:
            (A, b, interItr) = sys_fun(solk,interItr,*args)
            solk = (1 - relax) * solkm1 + relax * np.linalg.solve(A, b)
            # except np.linalg.LinAlgError:  # if condition number too high
            #     A = A + 1e-13 * np.identity(A.shape[0])
            #     solk = np.linalg.solve(A, b)

        norm = np.linalg.norm(abs(solk   - solkm1))/np.linalg.norm(abs(solkm1))

        normlist[k] = norm
        # residual = np.linalg.norm((np.dot(A,solk)-b))

        if norm > normlist[k - 1] and normlist[k - 1]<2e-4:  # WTF IS THAT ?
             break
        k = k + 1

        if k >= maxitr:
            print('Picard iteration not converged after ' + repr(maxitr) + ' iterations')
            plt.plot(np.arange(100),normlist,'.')
            plt.axis([0,100,0,0.0016])
            plt.pause(0.01)
            solk = np.full((len(solk),), np.nan, dtype=np.float64)
            break

    print('Iterations = ' + repr(k) + ', exiting norm = ' + repr(norm))
    return (solk,interItr)

#######################################
# ARHHHHHHH DON T USE pointers !!!!!!

def Jacobian(Residual_function, x,interItr, TypValue,*args):
    (Fx,interItr) = Residual_function(x,interItr,*args)
    Jac = np.zeros((len(x), len(x)), dtype=np.float64)
    for i in range(0, len(x)):
        Epsilon = np.finfo(float).eps ** 0.5 * max(x[i],TypValue[i])
        xip = np.copy(x)
        # xin = np.copy(x)
        xip[i] = xip[i] + Epsilon
        # xin[i] = xin[i]-Epsilon
        (Fxi,interItr) = Residual_function(xip,interItr,*args)
        Jac[:, i] = (Fxi - Fx) / Epsilon
        # Jac[:,i] = (Residual_function(xip,interItr,*args)[0] - Residual_function(xin,interItr,*args)[0])/(2*Epsilon)
    return Jac