# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Tue Nov  1 15:22:00 2016.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2017. All rights reserved.
See the LICENSE.TXT file for more details.

Tip inversion for different flow regimes. The functions take width opening and gives distance from tip calculated using the given propagation regime
"""

import math
import numpy as np
from scipy.optimize import brentq
from scipy.optimize import newton
import matplotlib.pyplot as plt


def TipAsym_viscStor_Res(dist,*args):
    """Residual function for viscocity dominate regime, without leak off"""
    
    (wEltRibbon,Kprime,Eprime,muPrime,Cbar,DistLstTSEltRibbon,dt) = args
    return wEltRibbon - (18*3**0.5*(dist-DistLstTSEltRibbon)/dt * muPrime/Eprime)**(1/3)*dist**(2/3)


##########################################    

def TipAsym_viscLeakOff_Res(dist,*args):
    """Residual function for viscocity dominated regime, with leak off"""
    (wEltRibbon,Kprime,Eprime,muPrime,Cbar,DistLstTSEltRibbon,dt) = args
    return wEltRibbon - 4/(15*np.tan(np.pi/8))**0.25 *(Cbar*muPrime/Eprime)**0.25 * ((dist-DistLstTSEltRibbon)/dt)**0.125 * dist**(5/8)


######################################

def f(K,Cb,C1):
    return 1/(3*C1) * (1-K**3 - 3*Cb*(1-K**2)/2 + 3*Cb**2*(1-K) - 3*Cb**3*np.log((Cb+1)/(Cb+K)))
    
def TipAsym_Universal_delt_Res(dist,*args):
    """More precise function to be minimized to find root for universal Tip assymptote (see Donstov and Pierce)"""
    
    (wEltRibbon,Kprime,Eprime,muPrime,Cbar,DistLstTSEltRibbon,dt) = args    
    
    Vel = (dist-DistLstTSEltRibbon)/dt
    Kh = Kprime*dist**0.5/(Eprime*wEltRibbon)
    Ch = 2*Cbar*dist**0.5/(Vel**0.5*wEltRibbon)
    sh = muPrime*Vel*dist**2/(Eprime*wEltRibbon**3)
    
    g0 = f(Kh,0.9911799823*Ch,10.392304845)
    delt= 10.392304845*(1+0.9911799823*Ch)*g0
    
    C1 = 4*(1-2*delt)/(delt*(1-delt)) * np.tan(math.pi*delt)
    C2 = 16*(1-3*delt)/(3*delt*(2-3*delt)) * np.tan(3*math.pi*delt/2)
    b  = C2/C1
    
    return sh-f(Kh,Ch*b,C1)

def TipAsym_Universal_zero_Res(dist,*args):
    """Function to be minimized to find root for universal Tip assymptote (see Donstov and Pierce 2017)"""
    (wEltRibbon,Kprime,Eprime,muPrime,Cbar,DistLstTSEltRibbon,dt) = args    
    
    Vel = (dist-DistLstTSEltRibbon)/dt
    Kh = Kprime*dist**0.5/(Eprime*wEltRibbon)
    Ch = 2*Cbar*dist**0.5/(Vel**0.5*wEltRibbon)
    g0 = f(Kh,0.9911799823*Ch,6*3**0.5)
    sh = muPrime*Vel*dist**2/(Eprime*wEltRibbon**3)
   
    return sh - g0
    
#########################################################################################    
    
#def TipAsymInversion_Universal_test(w,EltRibbon,Kprime,Eprime,muPrime,Cbar,DistLstTS,dt):
#    """Tip assymptote inversion from width to distance to tip for Universal case (see Donstov and Pierce 2017)"""
#    
#    dist = np.zeros((EltRibbon.size))
#    for i in range(0,len(EltRibbon)):
#        TipAsmptargs        = (w[EltRibbon[i]],Kprime[EltRibbon[i]],Eprime,muPrime[EltRibbon[i]],Cbar[EltRibbon[i]],-DistLstTS[EltRibbon[i]],dt)
#        minbnd = -DistLstTS[EltRibbon[i]]*(1+1*np.finfo(float).eps)
##        maxbnd = 5*(w[EltRibbon[i]]/(Kprime[EltRibbon[i]]/Eprime))**2
#        maxbnd = -DistLstTS[EltRibbon[i]]*(1+1e16*np.finfo(float).eps)
#        
##        if TipAsym_Universal_zero_Res(minbnd,*TipAsmptargs)*TipAsym_Universal_zero_Res(maxbnd,*TipAsmptargs)>0:
##            print ('wrong a b')
#        
#        x = np.linspace(minbnd,maxbnd,100)
#        g0 = np.zeros((100,),)
#        sol = np.zeros((100,),)
#        for j in range(0,100):
#            sol[j]=TipAsym_Universal_zero_Res(x[j],*TipAsmptargs)
#            
#            Vel = (x[j]+DistLstTS[EltRibbon[i]])/dt
#            Kh = Kprime[EltRibbon[i]]*x[j]**0.5/(Eprime*w[EltRibbon[i]])
#            Ch = 2*Cbar[EltRibbon[i]]*x[j]**0.5/(Vel**0.5*w[EltRibbon[i]])
#            g0[j] = f(Kh,0.9911799823*Ch,6*3**0.5)
#    
#        plt.plot(x[2:100],sol[2:100] , 'r')
#        plt.pause(0.01)
#        
#        plt.plot(x,g0 , )
#        plt.pause(0.01)
#        
##        print(repr(TipAsym_Universal_zero_Res(minbnd,*TipAsmptargs)))
#        dist[i]=brentq(TipAsym_Universal_delt_Res,minbnd , maxbnd,TipAsmptargs)
##        dist[i]=newton(TipAsym_Universal_zero_Res, 2,None, TipAsmptargs)
##        dist[i]=newton(TipAsym_Universal_delt_Res, 2.0,None, TipAsmptargs)
#        
#        
#    return dist
    
#########################################


def TipAsym_MKTransition_Res(dist,*args):
    """Residual function for viscocity to toughness regime with transition, without leak off"""
    (wEltRibbon,Kprime,Eprime,muPrime,Cbar,DistLstTSEltRibbon,dt) = args
    return wEltRibbon - (1 + 18*3**0.5*Eprime**2*(dist-DistLstTSEltRibbon)/dt*muPrime*dist**0.5/Kprime**3)**(1/3) * Kprime/Eprime*dist**0.5

    
#######################################
def FindBracket_dist(w,EltRibbon,Kprime,Eprime,muPrime,Cprime,DistLstTS,dt,ResFunc):
    """ Find the valid bracket for the root evaluation function. Also returns list of ribbon cells that are not propagating"""
    
    stagnant    = np.where(Kprime[EltRibbon]*(-DistLstTS[EltRibbon])**0.5/(Eprime*w[EltRibbon])>1) #propagation condition
    moving      = np.arange(EltRibbon.shape[0])[~np.in1d(EltRibbon,EltRibbon[stagnant])]
    
    a = -DistLstTS[EltRibbon[moving]]*(1+1e5*np.finfo(float).eps)
    b = 10*(w[EltRibbon[moving]]/(Kprime[EltRibbon[moving]]/Eprime))**2
    
    for i in range(0,len(moving)):

        TipAsmptargs    = (w[EltRibbon[moving[i]]],Kprime[EltRibbon[moving[i]]],Eprime,muPrime[EltRibbon[moving[i]]],Cprime[EltRibbon[moving[i]]],-DistLstTS[EltRibbon[moving[i]]],dt)        
        Res_a = ResFunc(a[i],*TipAsmptargs)
        Res_b = ResFunc(b[i],*TipAsmptargs)

        cnt   = 0
        mid   = b[i] 
        while Res_a*Res_b>0:
            mid     = (a[i]+2*mid)/3 #weighted 
            Res_a   = ResFunc(mid,*TipAsmptargs)
            a[i] = mid            
            cnt     += 1
            if cnt >= 30: # Should assume not propagating. not set to check how frequently it happens.
                raise SystemExit('Tip Inversion: front distance bracket cannot be found')
            
    
    return (moving,a,b)
            
#######################################
def TipAsymInversion(w,EltRibbon,Kprime,Eprime,regime,muPrime=None,Cprime=None,DistLstTS=None,dt=None):  
    """ Evaluate distance from the front using tip assymptotics of the given regime, given the fracture width in the ribbon cells"""
    
    if regime == 'U':
        ResFunc = TipAsym_Universal_zero_Res
#        ResFunc = TipAsym_Universal_delt_Res
    elif regime == 'Kt':
        return 0 # to be implementd
    elif regime == 'M':
        ResFunc = TipAsym_viscStor_Res
    elif regime == 'Mt':
        ResFunc = TipAsym_viscLeakOff_Res
    elif regime == 'MK':
        ResFunc = TipAsym_MKTransition_Res
    elif regime == 'K':
        return w[EltRibbon]**2 * (Eprime/Kprime[[EltRibbon]])**2
    
    
    (moving,a,b) = FindBracket_dist(w,EltRibbon,Kprime,Eprime,muPrime,Cprime,DistLstTS,dt,ResFunc)
    dist = -DistLstTS[EltRibbon]
    for i in range(0,len(moving)):
        TipAsmptargs    = (w[EltRibbon[moving[i]]],Kprime[EltRibbon[moving[i]]],Eprime,muPrime[EltRibbon[moving[i]]],Cprime[EltRibbon[moving[i]]],-DistLstTS[EltRibbon[moving[i]]],dt)
        try:
            dist[moving[i]]         = brentq(ResFunc,a[i],b[i],TipAsmptargs)
        except RuntimeError:
            dist[moving[i]] = np.nan
 
    return dist