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

def FiniteDiff_operator(w,EltCrack,muPrime,Mesh,InCrack):
    
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
def MakeEquationSystemExtendedFP(delwK,EltChannel,EltsTipNew,wLastTS,wTip,EltCrack,Mesh,dt,Q,C,muPrime,InCrack,LeakOff,sigma0):
    
    
    Ccc     = C[np.ix_(EltChannel,EltChannel)]
    Cct     = C[np.ix_(EltChannel,EltsTipNew)]
    
    A       = np.zeros((EltChannel.size+EltsTipNew.size,EltChannel.size+EltsTipNew.size),dtype=np.float64)
    S       = np.zeros((EltChannel.size+EltsTipNew.size,),dtype=np.float64)
    
    wcNplusOne              = np.copy(wLastTS)
    wcNplusOne[EltChannel]  = wcNplusOne[EltChannel]+delwK
    wcNplusOne[EltsTipNew]  = wTip

    cond    = FiniteDiff_operator(wcNplusOne,EltCrack,muPrime,Mesh,InCrack)
    
    condCC  = cond[np.ix_(EltChannel,EltChannel)]
    condCT  = cond[np.ix_(EltChannel,EltsTipNew)]
    condTC  = cond[np.ix_(EltsTipNew,EltChannel)]
    condTT  = cond[np.ix_(EltsTipNew,EltsTipNew)]    
    
    Channel     = np.arange(EltChannel.size)
    Tip         = Channel.size+np.arange(EltsTipNew.size)
    
    A[np.ix_(Channel,Channel)]   = np.identity(Channel.size) - dt*np.dot(condCC,Ccc)
    A[np.ix_(Channel,Tip)]       = -dt*condCT
    A[np.ix_(Tip,Channel)]       = -dt*np.dot(condTC,Ccc)
    A[np.ix_(Tip,Tip)]           = -dt*condTT

   
    S[Channel]  = dt*np.dot(condCC,np.dot(Ccc,wLastTS[EltChannel])+np.dot(Cct,wTip)+sigma0[EltChannel]) + dt/Mesh.hx/Mesh.hy*Q[EltChannel] - LeakOff[EltChannel]/Mesh.hx/Mesh.hy
    S[Tip]      = -(wTip-wLastTS[EltsTipNew]) + dt*np.dot(condTC,np.dot(Ccc,wLastTS[EltChannel])+np.dot(Cct,wTip)+sigma0[EltChannel]) - LeakOff[EltsTipNew]/Mesh.hx/Mesh.hy 
    
    return (A,S)

######################################

def MakeEquationSystemSameFP(delwk,w,EltCrack,NeiElements,Q,myC,dt,EltArea,muPrime,mesh,InCrack,LeakOff,sigma0):
    
    wnPlus1 = np.copy(w)
    wnPlus1[EltCrack] = wnPlus1[EltCrack]+delwk
    con     = FiniteDiff_operator(wnPlus1,EltCrack,muPrime,mesh,InCrack)
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
    
def ElastoHydrodynamicSolver_ExtendedFP(guess,Tol,EltChannel,EltCrack_k,EltsTipNew,wLastTS,wTip,Mesh,dt,Q,C,muPrime,InCrack,DLeakOff,sigma0):
    maxitr  = 100   
    solk    = guess
    k       = 0
    norm    = 1 
    while norm>Tol:
        delwK   = solk[np.arange(EltChannel.size)]
        delwK   = abs(delwK) # Keep the solution positive
        (A,S)   = MakeEquationSystemExtendedFP(delwK,EltChannel,EltsTipNew,wLastTS,wTip,EltCrack_k,Mesh,dt,Q,C,muPrime,InCrack,DLeakOff,sigma0)

#        (A,S)   = MakeEquationSystemExtendedFP(delwK,EltChannel,EltsTipNew,wLastTS,wTip,EltCrack_k,Mesh.NeiElements,dt,Q,C,muPrime,InCrack,Mesh.hx,Mesh.hy)
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
            solk    = np.linalg.solve(A,S)
        except np.linalg.LinAlgError: #if condition number too high
            A = A + 1e-13*np.identity(A.shape[0])
            solk    = np.linalg.solve(A,S)
            
        norm    = np.linalg.norm(abs(solk/solkm1-1))
        k       = k+1 
        if k>maxitr:
            raise SystemExit('Picard iteration not converged after '+repr(maxitr)+' iterations')
    if np.isnan(solk).any():
        raise SystemExit('delw sol is not evaluated correctly '+repr(solk))
    print('Iterations = '+repr(k)+', exiting norm Picard method= '+repr(norm))
    return solk
    