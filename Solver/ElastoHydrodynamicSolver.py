# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 14:43:38 2016

@author: Haseeb
"""

import numpy as np
import scipy.sparse.linalg as spla
from scipy.linalg import cho_factor
from scipy.linalg import cho_solve
import matplotlib.pyplot as plt
from Utility import *

def FiniteDiff_operator_laminar(w,EltCrack,muPrime,Mesh,InCrack):
    
    FinDiffOprtr    = np.zeros((w.size,w.size),dtype=np.float64)
    dx      = Mesh.hx
    dy      = Mesh.hy
    
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
    mu      = muPrime/12
    
    wLftEdge = (w[EltCrack]+w[Mesh.NeiElements[EltCrack,0]])/2
    wRgtEdge = (w[EltCrack]+w[Mesh.NeiElements[EltCrack,1]])/2
    wBtmEdge = (w[EltCrack]+w[Mesh.NeiElements[EltCrack,2]])/2
    wTopEdge = (w[EltCrack]+w[Mesh.NeiElements[EltCrack,3]])/2
    
    (dpdxLft,dpdxRgt,dpdyBtm,dpdyTop) = pressure_gradient(w,C,sigma0,Mesh,EltCrack,InCrack)

    ReLftEdge = 4/3 * rho*wLftEdge*(vkm1[0,EltCrack]**2+vkm1[4,EltCrack]**2)**0.5/mu[EltCrack]
    ReRgtEdge = 4/3 * rho*wRgtEdge*(vkm1[1,EltCrack]**2+vkm1[5,EltCrack]**2)**0.5/mu[EltCrack]
    ReBtmEdge = 4/3 * rho*wBtmEdge*(vkm1[2,EltCrack]**2+vkm1[6,EltCrack]**2)**0.5/mu[EltCrack]
    ReTopEdge = 4/3 * rho*wTopEdge*(vkm1[3,EltCrack]**2+vkm1[7,EltCrack]**2)**0.5/mu[EltCrack]
    
    rough     = 10000*np.ones((EltCrack.size,),np.float64)
    ffLftEdge = FF_YangJoseph(ReLftEdge,rough)
    ffRgtEdge = FF_YangJoseph(ReRgtEdge,rough)
    ffBtmEdge = FF_YangJoseph(ReBtmEdge,rough)
    ffTopEdge = FF_YangJoseph(ReTopEdge,rough)
#    ffLftEdge = 3/4 * 16*mu[EltCrack]/(rho*(vkm1[0,EltCrack]**2+vkm1[4,EltCrack]**2)**0.5*wLftEdge)
#    ffRgtEdge = 3/4 * 16*mu[EltCrack]/(rho*(vkm1[1,EltCrack]**2+vkm1[5,EltCrack]**2)**0.5*wRgtEdge)
#    ffBtmEdge = 3/4 * 16*mu[EltCrack]/(rho*(vkm1[2,EltCrack]**2+vkm1[6,EltCrack]**2)**0.5*wBtmEdge)
#    ffTopEdge = 3/4 * 16*mu[EltCrack]/(rho*(vkm1[3,EltCrack]**2+vkm1[7,EltCrack]**2)**0.5*wTopEdge)
    
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
    
    maxRe = max(ReLftEdge)
    
    ffLftEdge = FF_YangJoseph(ReLftEdge,rough)
    ffRgtEdge = FF_YangJoseph(ReRgtEdge,rough)
    ffBtmEdge = FF_YangJoseph(ReBtmEdge,rough)
    ffTopEdge = FF_YangJoseph(ReTopEdge,rough)
    
#    ffLftEdge = 3/4 * 16*mu[EltCrack]/(rho*(vk[0,EltCrack]**2+vk[4,EltCrack]**2)**0.5*wLftEdge)
#    ffRgtEdge = 3/4 * 16*mu[EltCrack]/(rho*(vk[1,EltCrack]**2+vk[5,EltCrack]**2)**0.5*wRgtEdge)
#    ffBtmEdge = 3/4 * 16*mu[EltCrack]/(rho*(vk[2,EltCrack]**2+vk[6,EltCrack]**2)**0.5*wBtmEdge)
#    ffTopEdge = 3/4 * 16*mu[EltCrack]/(rho*(vk[3,EltCrack]**2+vk[7,EltCrack]**2)**0.5*wTopEdge)
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

def MakeEquationSystemExtendedFP(delwK,EltChannel,EltsTipNew,wLastTS,wTip,EltCrack,Mesh,dt,Q,C,muPrime,rho,InCrack,LeakOff,sigma0,vkm1):
    
    
    Ccc     = C[np.ix_(EltChannel,EltChannel)]
    Cct     = C[np.ix_(EltChannel,EltsTipNew)]
    
    A       = np.zeros((EltChannel.size+EltsTipNew.size,EltChannel.size+EltsTipNew.size),dtype=np.float64)
    S       = np.zeros((EltChannel.size+EltsTipNew.size,),dtype=np.float64)
    
    wcNplusOne              = np.copy(wLastTS)
    wcNplusOne[EltChannel]  = wcNplusOne[EltChannel]+delwK
    wcNplusOne[EltsTipNew]  = wTip

    (FinDiffOprtr,vk)    = FiniteDiff_operator_turbulent(wcNplusOne,EltCrack,muPrime,Mesh,InCrack,rho,vkm1,C,sigma0)
                                                                                
#    FinDiffOprtr    = FiniteDiff_operator_laminar(wcNplusOne,EltCrack,muPrime,Mesh,InCrack)
        
    condCC  = FinDiffOprtr[np.ix_(EltChannel,EltChannel)]
    condCT  = FinDiffOprtr[np.ix_(EltChannel,EltsTipNew)]
    condTC  = FinDiffOprtr[np.ix_(EltsTipNew,EltChannel)]
    condTT  = FinDiffOprtr[np.ix_(EltsTipNew,EltsTipNew)]    
    
    Channel     = np.arange(EltChannel.size)
    Tip         = Channel.size+np.arange(EltsTipNew.size)
    
    A[np.ix_(Channel,Channel)]   = np.identity(Channel.size) - dt*np.dot(condCC,Ccc)
    A[np.ix_(Channel,Tip)]       = -dt*condCT
    A[np.ix_(Tip,Channel)]       = -dt*np.dot(condTC,Ccc)
    A[np.ix_(Tip,Tip)]           = -dt*condTT

   
    S[Channel]  = dt*np.dot(condCC,np.dot(Ccc,wLastTS[EltChannel])+np.dot(Cct,wTip)+sigma0[EltChannel]) + dt/Mesh.hx/Mesh.hy*Q[EltChannel] - LeakOff[EltChannel]/Mesh.hx/Mesh.hy
    S[Tip]      = -(wTip-wLastTS[EltsTipNew]) + dt*np.dot(condTC,np.dot(Ccc,wLastTS[EltChannel])+np.dot(Cct,wTip)+sigma0[EltChannel]) - LeakOff[EltsTipNew]/Mesh.hx/Mesh.hy 
    
    return (A,S,vk)

######################################

def MakeEquationSystemSameFP(delwk,w,EltCrack,NeiElements,Q,myC,dt,EltArea,muPrime,mesh,InCrack,LeakOff,sigma0):
    
    wnPlus1 = np.copy(w)
    wnPlus1[EltCrack] = wnPlus1[EltCrack]+delwk
    con     = FiniteDiff_operator_laminar(wnPlus1,EltCrack,muPrime,mesh,InCrack)
    con     = con[np.ix_(EltCrack,EltCrack)]
    CCrack  = myC[np.ix_(EltCrack,EltCrack)]
    
    A = np.identity(EltCrack.size)-dt*np.dot(con,CCrack)
    S = dt*np.dot(con,np.dot(CCrack,w[EltCrack])+sigma0[EltCrack]) + dt/EltArea*Q[EltCrack] - LeakOff[EltCrack]/EltArea
    return (A,S)

#######################################

def ElastoHydrodynamicSolver_SameFP(guess,Tol,w,NeiElements,EltCrack,dt,Q,myC,EltArea,muPrime,mesh,InCrack,DLeakOff,sigma0):
    maxitr  = 100    
    solk    = guess
    k       = 0
    norm    = 1 
    while norm>Tol:
        (A,S)   = MakeEquationSystemSameFP(solk,w,EltCrack,NeiElements,Q,myC,dt,EltArea,muPrime,mesh,InCrack,DLeakOff,sigma0)
        solkm1  = solk
#        print('condtion = '+repr(np.linalg.cond(A))+ ' precision = '+repr(2.220446049250313e-16))        
        solk    = np.linalg.solve(A,S)
        norm    = np.linalg.norm(abs(solk/solkm1-1))
        k       = k+1
        if k>maxitr:
            raise SystemExit('Picard iteration not converged after '+repr(maxitr)+' iterations, norm = '+repr(norm))
    print('Iterations = '+repr(k)+', exiting norm Picard method= '+repr(norm))    
    return solk
    
def ElastoHydrodynamicSolver_ExtendedFP(guess,Tol,EltChannel,EltCrack_k,EltsTipNew,wLastTS,wTip,Mesh,dt,Q,C,muPrime,rho,InCrack,DLeakOff,sigma0):
    maxitr  = 100   
    solk    = guess
    k       = 0
    norm    = 1
    relax   = 1
    normlist = np.zeros((maxitr,),float)
    
    delwK   = solk[np.arange(EltChannel.size)]
    wcNplusOne              = np.copy(wLastTS)
    wcNplusOne[EltChannel]  = wcNplusOne[EltChannel]+delwK
    wcNplusOne[EltsTipNew]  = wTip
    vk = velocity(wcNplusOne,EltCrack_k,Mesh,InCrack,muPrime,C,sigma0)

    while norm>Tol:
    
        delwK   = solk[np.arange(EltChannel.size)]
        delwK   = abs(delwK) # Keep the solution positive
        (A,S,vk)   = MakeEquationSystemExtendedFP(delwK,EltChannel,EltsTipNew,wLastTS,wTip,EltCrack_k,Mesh,dt,Q,C,muPrime,rho,InCrack,DLeakOff,sigma0,vk)

  #        print('positive definite '+repr(np.linalg.eigvals(A)))
#        Eig = np.linalg.eigvals(A)
#        print('min '+("%04.03e" % Eig.min())+' max '+("%04.03e" % Eig.max()))
#        print('min diagonal '+("%04.03e" % min(A.diagonal())))
        solkm1  = solk

        try:        
#            M2 = spla.spilu(A)
#            M_x = lambda x: M2.solve(x)
#            M = spla.LinearOperator(A.shape, M_x)
#            solk = spla.gmres(A,S,M=M)[0]
            solk    = (1-relax)*solkm1+relax*np.linalg.solve(A,S)
        except np.linalg.LinAlgError: #if condition number too high
            A = A + 1e-13*np.identity(A.shape[0])
            solk    = np.linalg.solve(A,S)
            
        norm    = np.linalg.norm(abs(solk/solkm1-1))
        normlist[k] = norm
        if norm>normlist[k-1] and norm<1e-4:
            break
        k       = k+1
        if k>=maxitr:
            print('Picard iteration not converged after '+repr(maxitr)+' iterations')
            print('norms of the last iterations'+repr(normlist))
            solk = np.full((len(solk),),np.nan,dtype=np.float64)
            break

    print('Iterations = '+repr(k)+', exiting norm Picard method= '+repr(norm))
    return solk

###################################################

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

def pressure_gradient(w,C,sigma0,Mesh,EltCrack,InCrack):
    
    pf          = np.zeros((Mesh.NumberOfElts,),dtype=np.float64)
    pf[EltCrack]          = np.dot(C[np.ix_(EltCrack,EltCrack)],w[EltCrack]) + sigma0[EltCrack]
        
    dpdxLft     = (pf[EltCrack]-pf[Mesh.NeiElements[EltCrack,0]])*InCrack[Mesh.NeiElements[EltCrack,0]]
    dpdxRgt     = (pf[Mesh.NeiElements[EltCrack,1]]-pf[EltCrack])*InCrack[Mesh.NeiElements[EltCrack,1]]
    dpdyBtm     = (pf[EltCrack]-pf[Mesh.NeiElements[EltCrack,2]])*InCrack[Mesh.NeiElements[EltCrack,2]]
    dpdyTop     = (pf[Mesh.NeiElements[EltCrack,3]]-pf[EltCrack])*InCrack[Mesh.NeiElements[EltCrack,3]]
    
    return (dpdxLft,dpdxRgt,dpdyBtm,dpdyTop)
    
def FF_YangJoseph(ReNum,rough):
    
    ff = np.full((len(ReNum),),np.inf,dtype=np.float64)
    nonzero = np.where(abs(ReNum)>1e-5)[0]    

    lamdaS =   (-((-64/ReNum[nonzero] + 0.000083*ReNum[nonzero]**0.75)/(1 + 2320**50/ReNum[nonzero]**50)**0.5) - 64/ReNum[nonzero] + 0.3164/ReNum[nonzero]**0.25)/(1 + 3810**15/ReNum[nonzero]**15)**0.5 + (-((-((-64/ReNum[nonzero] + 0.000083*ReNum[nonzero]**0.75)/(1 + 2320**50/ReNum[nonzero]**50)**0.5) - 64/ReNum[nonzero] + 0.3164/ReNum[nonzero]**0.25)/(1 + 3810**15/ReNum[nonzero]**15)**0.5) - (-64/ReNum[nonzero] + 0.000083*ReNum[nonzero]**0.75)/(1 + 2320**50/ReNum[nonzero]**50)**0.5 - 64/ReNum[nonzero] + 0.1537/ReNum[nonzero]**0.185)/(1 + 1680700000000000000000000/ReNum[nonzero]**5)**0.5 + (-((-((-64/ReNum[nonzero] + 0.000083*ReNum[nonzero]**0.75)/(1 + 2320**50/ReNum[nonzero]**50)**0.5) - 64/ReNum[nonzero] + 0.3164/ReNum[nonzero]**0.25)/(1 + 3810**15/ReNum[nonzero]**15)**0.5) - (-((-((-64/ReNum[nonzero] + 0.000083*ReNum[nonzero]**0.75)/(1 + 2320**50/ReNum[nonzero]**50)**0.5) - 64/ReNum[nonzero] + 0.3164/ReNum[nonzero]**0.25)/(1 + 3810**15/ReNum[nonzero]**15)**0.5) - (-64/ReNum[nonzero] + 0.000083*ReNum[nonzero]**0.75)/(1 + 2320**50/ReNum[nonzero]**50)**0.5 - 64/ReNum[nonzero] + 0.1537/ReNum[nonzero]**0.185)/(1 + 1680700000000000000000000/ReNum[nonzero]**5)**0.5 - (-64/ReNum[nonzero] + 0.000083*ReNum[nonzero]**0.75)/(1 + 2320**50/ReNum[nonzero]**50)**0.5 - 64/ReNum[nonzero] + 0.0753/ReNum[nonzero]**0.136)/(1 + 4000000000000/ReNum[nonzero]**2)**0.5 + (-64/ReNum[nonzero] + 0.000083*ReNum[nonzero]**0.75)/(1 + 2320**50/ReNum[nonzero]**50)**0.5 + 64/ReNum[nonzero]
    lamdaR =   ReNum[nonzero]**(-0.2032 + 7.348278/rough[nonzero]**0.96433953)*(-0.022 + (-0.978 + 0.92820419*rough[nonzero]**0.03569244 - 0.00255391*rough[nonzero]**0.8353877)/(1 + 265550686013728218770454203489441165109061383639474724663955742569518708077419167245843482753466249/rough[nonzero]**50)**0.5 + 0.00255391*rough[nonzero]**0.8353877) + (-(ReNum[nonzero]**(-0.2032 + 7.348278/rough[nonzero]**0.96433953)*(-0.022 + (-0.978 + 0.92820419*rough[nonzero]**0.03569244 - 0.00255391*rough[nonzero]**0.8353877)/(1 + 265550686013728218770454203489441165109061383639474724663955742569518708077419167245843482753466249/rough[nonzero]**50)**0.5 + 0.00255391*rough[nonzero]**0.8353877)) + 0.01105244*ReNum[nonzero]**(-0.191 + 0.62935712/rough[nonzero]**0.28022284)*rough[nonzero]**0.23275646 + (ReNum[nonzero]**(0.015 + 0.26827956/rough[nonzero]**0.28852025)*(0.0053 + 0.02166401/rough[nonzero]**0.30702955) - 0.01105244*ReNum[nonzero]**(-0.191 + 0.62935712/rough[nonzero]**0.28022284)*rough[nonzero]**0.23275646 + (ReNum[nonzero]**0.002*(0.011 + 0.18954211/rough[nonzero]**0.510031) - ReNum[nonzero]**(0.015 + 0.26827956/rough[nonzero]**0.28852025)*(0.0053 + 0.02166401/rough[nonzero]**0.30702955) + (0.0098 - ReNum[nonzero]**0.002*(0.011 + 0.18954211/rough[nonzero]**0.510031) + 0.17805185/rough[nonzero]**0.46785053)/(1 + (8.733801045300249e10*rough[nonzero]**0.90870686)/ReNum[nonzero]**2)**0.5)/(1 + (6.44205549308073e15*rough[nonzero]**5.168887)/ReNum[nonzero]**5)**0.5)/(1 + (1.1077593467238922e13*rough[nonzero]**4.9771653)/ReNum[nonzero]**5)**0.5)/(1 + (2.9505925619934144e14*rough[nonzero]**3.7622822)/ReNum[nonzero]**5)**0.5
    ff[nonzero] = np.asarray(lamdaS + (lamdaR-lamdaS)/(1+(ReNum[nonzero]/(45.196502*rough[nonzero]**1.2369807+1891))**-5)**0.5,float)/4
    
    return ff    
    
    