# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri Oct 14 18:27:39 2016.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2017. All rights reserved.
See the LICENSE.TXT file for more details.


Functions to calculate tip volumes, given the propagation regime

    regime -- A  gives the area (fill fraction)
    regime -- K  gives tip volume according to the square root assymptote
    regime -- M  gives tip volume according to the viscocity dominated assymptote 
    regime -- Lk is used to calculate the leak off given the distance of the front l (note, its not tip volume) 
    regime -- Mt gives tip volume according to the viscocity, Leak-off assymptote 
    regime -- U  gives tip volume according to the Universal assymptote (Donstov and Pierce, 2017)
    regime -- MK gives tip volume according to the M-K transition assymptote
    
"""
import numpy as np
from importlib.machinery import SourceFileLoader
import matplotlib.pyplot as plt
from TipInversion import f
from scipy.optimize import brentq

 
def TipAsym_UniversalW_zero_Res(w,*args):
    """Function to be minimized to find root for universal Tip assymptote (see Donstov and Pierce 2017)"""
    (dist,Kprime,Eprime,muPrime,Cbar,Vel) = args     

    Kh = Kprime*dist**0.5/(Eprime*w)
    Ch = 2*Cbar*dist**0.5/(Vel**0.5*w)
    g0 = f(Kh,0.9911799823*Ch,6*3**0.5)
    sh = muPrime*Vel*dist**2/(Eprime*w**3)
   
    return sh - g0
    
def TipAsym_UniversalW_delt_Res(w,*args):                              
    """Residual for the General assymptote (not sure if its actual residual but zero will give the solution)"""
    
    (dist,Kprime,Eprime,muPrime,Cbar,Vel) = args    
    
    Kh = Kprime*dist**0.5/(Eprime*w)
    Ch = 2*Cbar*dist**0.5/(Vel**0.5*w)
    sh = muPrime*Vel*dist**2/(Eprime*w**3)
    
    g0 = f(Kh,0.9911799823*Ch,10.392304845)
    delt= 10.392304845*(1+0.9911799823*Ch)*g0
    
    C1 = 4*(1-2*delt)/(delt*(1-delt)) * np.tan(np.pi*delt)
    C2 = 16*(1-3*delt)/(3*delt*(2-3*delt)) * np.tan(3*np.pi*delt/2)
    b  = C2/C1
    
    return sh-f(Kh,Ch*b,C1)

 
def MomentsTipAssympGeneral(dist, Kprime, Eprime, muPrime, Cbar, Vel, stagnant, KIPrime): 
    """Moments of the General tip assymptote to calculate the volume integral (see Donstov and Pierce, 2017)"""
        
    TipAsmptargs    = (dist,Kprime,Eprime,muPrime,Cbar,Vel)

#    x = np.linspace(a,b,100)
#    sol = np.zeros((100,),)
#    for j in range(0,100):
#        sol[j]=TipAsym_UniversalW_delt_Res(x[j],*TipAsmptargs)
#    
#    plt.plot(x,sol , 'r')
#    plt.pause(0.01)
#    plt.close("all")
    if stagnant:
        w   = KIPrime*dist**0.5/Eprime
    else:
        (a,b)   = FindBracket_w(dist,Kprime,Eprime,muPrime,Cbar,Vel)
        w       = brentq(TipAsym_UniversalW_zero_Res,a,b,TipAsmptargs)#root finding between the upper and lower bounds on width
        if w<-1e-15:
            w = abs(w)
    
    if Vel<1e-6:
        delt = 1/6
    else:
        Kh = Kprime*dist**0.5/(Eprime*w)
        Ch = 2*Cbar*dist**0.5/(Vel**0.5*w)
        g0 = f(Kh,0.9911799823*Ch,10.392304845)
        delt= 10.392304845*(1+0.9911799823*Ch)*g0
    
    M0 = 2*w*dist/(3+delt)

    M1 = 2*w*dist**2/(5+delt)
    if np.isnan(M0) or np.isnan(M1):
        raise SystemExit('Moment(s) of assymptote are nan')
    return (M0,M1)

    
def Pdistance(x, y, slope, intercpt): 
    """distance of a point from a line"""

    return (slope*x -y + intercpt)/(slope**2+1)**0.5;

def VolumeTriangle(dist, em, regime, Kprime, Eprime, muPrime, Cbar, Vel, stagnant, KIPrime): 
    """Volume  of the triangle defined by perpendicular distance (dist) and em (em=1/sin(alpha)cos(alpha), where alpha is the angle of the perpendicular)
    The regime variable identifies the propagation regime    
    """
    

    if regime == 'A':
        return dist**2*em/2
        
    elif regime == 'K':   
        return 4/15 * Kprime/Eprime * dist**2.5*em
        
    elif regime == 'M':
        return 0.7081526678 * (Vel*muPrime/Eprime)**(1/3) * em * dist**(8/3)
        
    elif regime == 'Lk':
        return 2/5 * Vel**-0.5 * dist**(5/2) * em
        
    elif regime == 'Mt':
        return 256/273/(15*np.tan(np.pi/8))**0.25 * (Cbar*muPrime/Eprime)**0.25 * em * Vel**0.125 * dist**(21/8)
        
    elif regime == 'U':
        (M0,M1) = MomentsTipAssympGeneral(dist, Kprime, Eprime, muPrime, Cbar, Vel, stagnant, KIPrime)
        return em*(dist*M0 - M1) 

    elif regime == 'MK': 
        return (3.925544049000839e-9*em*Kprime*(1.7320508075688772*Kprime**9*(Kprime**6 - 1872.*dist*Eprime**4*muPrime**2*Vel**2) + (1. + (31.17691453623979*(dist)**0.5*Eprime**2*muPrime*Vel)/Kprime**3)**0.3333333333333333*(-1.7320508075688772*Kprime**15 + 18.*(dist)**0.5*Eprime**2*Kprime**12*muPrime*Vel + 2868.2761373340604*dist*Eprime**4*Kprime**9*muPrime**2*Vel**2 - 24624.*dist**1.5*Eprime**6*Kprime**6*muPrime**3*Vel**3 + 464660.73424811783*dist**2*Eprime**8*Kprime**3*muPrime**4*Vel**4 + 5.7316896e7*dist**2.5*Eprime**10*muPrime**5*Vel**5)))/(Eprime**11*muPrime**5*Vel**5)
        
        
def Area(dist, regime, Kprime, Eprime, muPrime, Cbar, Vel, stagnant, KIPrime): 
    """Gives Area under the tip depending on the regime identifier ;  
    used in case of 0 or 90 degree angle; can be used for 1d case"""
    if regime == 'A': 
        return dist
        
    elif regime == 'K':
        return 2/3 * Kprime/Eprime * dist**1.5    

    elif regime == 'M':
        return 1.8884071141 * (Vel*muPrime/Eprime)**(1/3) * dist**(5/3)
        
    elif regime == 'Lk':
        return 2/3 * Vel**-0.5 * dist**1.5
        
    elif regime == 'Mt':
        return 32/13/(15*np.tan(np.pi/8))**0.25 * (Cbar*muPrime/Eprime)**0.25 * Vel**0.125 * dist**(13/8)    
    
    elif regime == 'U':
        (M0,M1) = MomentsTipAssympGeneral(dist, Kprime, Eprime, muPrime, Cbar, Vel, stagnant, KIPrime)
        return M0 
        
    elif regime == 'MK': 
        return (7.348618459729571e-6*Kprime*(-1.7320508075688772*Kprime**9 + (1. + (31.17691453623979*(dist)**0.5*Eprime**2*muPrime*Vel)/Kprime**3)**0.3333333333333333*(1.7320508075688772*Kprime**9 - 18.*(dist)**0.5*Eprime**2*Kprime**6*muPrime*Vel + 374.12297443487745*dist*Eprime**4*Kprime**3*muPrime**2*Vel**2 + 81648.*dist**1.5*Eprime**6*muPrime**3*Vel**3)))/(Eprime**7*muPrime**3*Vel**3)
        
def VolumeIntegral(alpha, l, dx, dy, regime, Kprime, Eprime, muPrime, Cbar, Vel, stagnant=[], KIPrime=[]):
    """Calculate Volume integrals of the grid cells according to the tip assymptote given by the variable regime"""
    if len(stagnant)==0:
        stagnant    = np.zeros((alpha.size,),bool)
        KIPrime     = np.zeros((alpha.size,),float)
        
    volume = np.zeros((len(l),),float)
    for i in range(0,len(l)):

        if abs(alpha[i])<1e-10:
            if l[i]<=dx:
                volume[i]  = Area(l[i],regime, Kprime[i], Eprime, muPrime[i], Cbar[i], Vel[i], stagnant[i], KIPrime[i])*dy
            else:
                volume[i]  = (Area(l[i],regime, Kprime[i], Eprime, muPrime[i], Cbar[i], Vel[i], stagnant[i], KIPrime[i])-Area(l[i]-dx,regime, Kprime[i], Eprime, muPrime[i], Cbar[i], Vel[i], stagnant[i], KIPrime[i]))*dy
        
        elif abs(alpha[i] - np.pi/2)<1e-10:
            if l[i]<=dy:
                volume[i]  = Area(l[i],regime, Kprime[i], Eprime, muPrime[i], Cbar[i], Vel[i], stagnant[i], KIPrime[i])*dx
            else:
                volume[i]  = (Area(l[i],regime, Kprime[i], Eprime, muPrime[i], Cbar[i], Vel[i], stagnant[i], KIPrime[i])-Area(l[i]-dy,regime, Kprime[i], Eprime, muPrime[i], Cbar[i], Vel[i], stagnant[i], KIPrime[i]))*dx
        else:
            yIntrcpt = l[i]/np.cos(np.pi/2-alpha[i]);
            grad     = -1/np.tan(alpha[i]);
            m        = 1/(np.sin(alpha[i])*np.cos(alpha[i]));  
    
            TriVol   =  VolumeTriangle(l[i],m,regime, Kprime[i], Eprime, muPrime[i], Cbar[i], Vel[i], stagnant[i], KIPrime[i]);
                
            lUp = Pdistance(0, dy, grad, yIntrcpt); #distance of the front from the upper left vertex of the grid cell
            if lUp > 0: #upper vertex of the triangle is higher than the grid cell height
                UpTriVol = VolumeTriangle(lUp,m,regime, Kprime[i], Eprime, muPrime[i], Cbar[i], Vel[i], stagnant[i], KIPrime[i]);
            else:
                UpTriVol = 0;
                
            lRt = Pdistance(dx, 0, grad, yIntrcpt); #distance of the front from the lower right vertex of the grid cell
            if lRt > 0: #right vertex of the triangle is wider than the grid cell width
                RtTriVol = VolumeTriangle(lRt,m,regime, Kprime[i], Eprime, muPrime[i], Cbar[i], Vel[i], stagnant[i], KIPrime[i]);
            else:
                RtTriVol = 0;
                
            IntrsctTriDist = Pdistance(dx, dy, grad, yIntrcpt); #distance of the front from the upper right vertex of the grid cell   
            if IntrsctTriDist > 0: # front has passed the grid cell
                IntrsctTri = VolumeTriangle(IntrsctTriDist,m,regime, Kprime[i], Eprime, muPrime[i], Cbar[i], Vel[i], stagnant[i], KIPrime[i])
            else:
                IntrsctTri = 0;
    

            volume[i]  = TriVol - UpTriVol - RtTriVol + IntrsctTri

    return volume
    
def FindBracket_w(dist,Kprime,Eprime,muPrime,Cprime,Vel):
    
    a = (1e5*np.finfo(float).eps)*dist**0.5*Kprime/Eprime        #lower bound on width
    b = 10000*dist**0.5*Kprime/Eprime   
    
    TipAsmptargs    = (dist,Kprime,Eprime,muPrime,Cprime,Vel)
    Res_a = TipAsym_UniversalW_zero_Res(a,*TipAsmptargs)
    Res_b = TipAsym_UniversalW_zero_Res(b,*TipAsmptargs)

    cnt   = 0
    mid   = b
    while Res_a*Res_b>0:
        mid     = (a+2*mid)/3 #weighted 
        Res_a   = TipAsym_UniversalW_zero_Res(mid,*TipAsmptargs)
#        a = mid            
        cnt     += 1
        if cnt >= 50:
#            x = np.linspace((1e5*np.finfo(float).eps)*dist**0.5*Kprime/Eprime ,maxbnd,100)
#            g0 = np.zeros((100,),)
#            sol = np.zeros((100,),)
#            for j in range(0,100):
#                sol[j]=TipAsym_Universal_zero_Res(x[j],*TipAsmptargs)
#                
#                Vel = (x[j]+DistLstTS[EltRibbon[i]])/dt
#                Kh = Kprime[EltRibbon[i]]*x[j]**0.5/(Eprime*w[EltRibbon[i]])
#                Ch = 2*Cbar[EltRibbon[i]]*x[j]**0.5/(Vel**0.5*w[EltRibbon[i]])
#                g0[j] = f(Kh,0.9911799823*Ch,6*3**0.5)
            raise SystemExit('fracture width bracket cannot be found')
            
    
    return (a,b)