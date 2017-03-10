# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 16:22:33 2016

@author: Haseeb
"""

from Domain import *
from Utility import *
from AnalyticalSolution import *
from ElastoHydrodynamicSolver import *
from TipInversion import *
from LevelSet import *
from VolIntegral import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pickle

class Fracture(Domain):
    """ Class defining propagating fracture; Is extension of Domain class.
        
        Instance variables:
            Eprime (float):             elastic modulus (E'), assumed to uniform for complete domain (see e.g. Detournay 2016, Annual Review)
            Kprime (ndarray-float):     toughness (K' parameter) for each cell in the domain (see e.g. Detournay 2016, Annual Review)
            sigmaO (ndarray-float):     in-situ stress field
            Cprime (ndarray-float):     Carter's leak off coefficient (C' parameter) for each cell in the domain (see e.g. Detournay 2016, Annual Review)
            muPrime (ndarray-float):    viscocity (mu' parameter) for each cell in the domain (see e.g. Detournay 2016, Annual Review)
            w (ndarray-float):          fracture opening (width)
            p (ndarray-float):          fracture pressure 
            EltChannel (ndarray-int):   list of cells currently in the channel region
            EltCrack (ndarray-int):     list of cells currently in the crack region
            EltRibbon (ndarray-int):    list of cells currently in the Ribbon region
            EltTip (ndarray-int):       list of cells currently in the Tip region
            v (ndarray-float):          propagation velocity for each cell in the tip cells.
            alpha (ndarray-float):      angle prescribed by perpedicular on the fracture front (see Pierce 2015, Computation Methods Appl. Mech)
            l (ndarray-float):          length of perpedicular on the fracture front (see Pierce 2015, Computation Methods Appl. Mech)
            ZeroVertex (ndarray-float): Vertex from which the perpendicular is drawn (can have value from 0 to 3)
            FillF (ndarray-float):      filling fraction of each tip cell
            CellStatus (ndarray-int):   specifies which region each element currently belongs to
            initRad (float):            starting radius
            initTime (float):           starting time
            sgndDist (ndarray-float):   signed minimun distace from fractire front of each cell in the domain
            Q (ndarray-float):          injection rate into each cell of the domain
            
        functions:
            InitializeRadialFracture:   set initial conditions of a radial fracture to start simulation
            InitializePKN               set initial conditions of a PKN fracture
            PlotFracture:               plot given variable of the fracture
            PrintFractureTrace:         plot current regions and front position of the fracture
            Propagate:                  propagate fracture for the given time step 
            
    """
    
    def __init__(self,mesh,Eprime,Kprime,sigma0,Cprime,muPrime,rho,Q0):
        """ Constructor for the Fracture class
            Arguments:
                mesh (CartesianMesh):           A mesh describing the domain
                Eprime (float):                 Initial value of elastic modulus
                (Kprime,sigmaO,Cprime,muPrime): Arguments giving the initial values of the parameters
                                                These can be ndarray of a single value. If single value is given, each
                                                cell will get the same value
                Q0 (float or ndarray):          Initial injection rate. can be an array or a single value. If single value
                                                is given, it will be injection in the center of the fracture. 
        """
        
        super().__init__(mesh,Eprime,Kprime,sigma0,Cprime)
#        (self.Eprime,self.mesh,self.Kprime,self.sigmaO,self.Cprime) = (domain.Eprime,domain.mesh,domain.Kprime,domain.sigmaO,domain.Cprime)
        (self.w,self.p,self.time) = (np.zeros((mesh.NumberOfElts),float),np.zeros((mesh.NumberOfElts),float),0.0)
        (self.EltChannel,self.EltRibbon,self.EltCrack) = (np.asarray([],dtype=np.int32),np.asarray([],dtype=np.int32),np.asarray([],dtype=np.int32))
        (self.EltTip,self.alpha,self.l,self.ZeroVertex,self.v) = ([],[],[],[],[])
        (self.FillF,self.CellStatus) = ([],[])
        (self.initRad,self.initTime) = ([],[])
        self.sgndDist = []
        (self.Leakedoff,self.Tarrival) = (np.zeros((mesh.NumberOfElts),float),[])
        self.InCrack = []
        self.FractEvol = np.empty((1,4),float)
        self.exitstatus = 0
        self.rho = rho
        
        if isinstance(Q0, np.ndarray): #check if float or ndarray
            self.Q = Q0
        else:
            self.Q = np.zeros((mesh.NumberOfElts,),float)
            self.Q[mesh.CenterElts] = Q0/len(mesh.CenterElts)
            
        if isinstance(muPrime, np.ndarray): #check if float or ndarray
            self.muPrime = muPrime
        else:
            self.muPrime = muPrime*np.ones((mesh.NumberOfElts,),float)
            
            
#############################################
        
    def InitializeRadialFracture(self,initValue,initType,regime):
        """ Initialize the fracture, according to the given initial value and the propagation regime. Either initial radius or time can be given 
            as the initial value. The function sets up the fracture front and other fracture parameters according to the given ragime at the 
            given time or radius.
            
            Arguments:
                initValue (float):      initial value
                initType (string):      Possible values:
                                            time        -- indicating the given value is initial time
                                            radius      -- indicating the given value is initial radius
                regime (string):        Possible values:
                                            K   -- indicating toughness dominated regime, without leak off
                                            M   -- indicating viscocity dominated regime, without leak off
                                            Mt  -- indicating viscocity dominated regime, with leak off
        """
        if initType == 'time':
            self.time = initValue
            if regime == 'K':
                (self.initRad,self.p,self.w,v) = KvertexSolutionTgiven(np.mean(self.Kprime), np.mean(self.Eprime), sum(self.Q), self.mesh, initValue)
            elif regime == 'M':
                (self.initRad,self.p,self.w,v) = MvertexSolutionTgiven(self.Eprime, sum(self.Q), np.mean(self.muPrime), self.mesh,initValue)
            elif regime == 'Mt':
                (self.initRad,self.p,self.w,v) = MTvertexSolutionTgiven(np.mean(self.Eprime), np.mean(self.Cprime), sum(self.Q), np.mean(self.muPrime), self.mesh,initValue)
            else:
                print('regime '+regime+' not supported')
                return
        elif initType == 'radius':
            self.initRad = initValue
            if regime == 'K':
                (self.time,self.p,self.w,v) = KvertexSolutionRgiven(np.mean(self.Kprime), np.mean(self.Eprime), sum(self.Q), self.mesh, initValue)
            elif regime == 'M':
                (self.time,self.p,self.w,v) = MvertexSolutionRgiven(self.Eprime, sum(self.Q), np.mean(self.muPrime), self.mesh, initValue)
            elif regime == 'Mt':
                (self.time,self.p,self.w,v) = MTvertexSolutionRgiven(self.Eprime, np.mean(self.Cprime), sum(self.Q), np.mean(self.muPrime), self.mesh, initValue)
            else:
                print('regime '+regime+' not supported')
                return
        else:
            print('initType '+initType+' not supported')
        # level set value at middle of the elements
        phiMid = np.empty([self.mesh.NumberOfElts, 1],dtype=float)
        for e in range(0,self.mesh.NumberOfElts) :
            phiMid[e]=RadiusLevelSet(self.mesh.CenterCoor[e],self.initRad)    
        # level set value at vertices of the element
        phiVertices=np.empty([len(self.mesh.VertexCoor), 1],dtype=float)
        for i in range(0,len(self.mesh.VertexCoor)) :
          phiVertices[i]=RadiusLevelSet(self.mesh.VertexCoor[i],self.initRad)                
        # finding elements containing at least one vertices inside the fracture, i.e. with a value of the level <0
          # avoiding loop on elements....
    
        psum=np.sum(phiVertices[self.mesh.Connectivity[:]]<0,axis=1) # array of Length (number of elements) containig the sum of vertices with neg level set value)
        EltTip =(np.where(np.logical_and(psum>0,psum<4)))[0] # indices of tip element which by definition have less than 4 but at least 1 vertices inside the level set
        EltCrack=(np.where(psum>0))[0] # # indices of cracked element
        EltChannel=(np.where(psum==4))[0] # indices of channel element / fully cracked
    
        # find the ribbon elements: Channel Elements having at least 
        # on common vertices with a Tip element
        #
        # loop on ChannelElement, and on TipElement
        testribbon = np.empty([len(EltChannel), 1],dtype=float)
        for e in range(0,len(EltChannel)) :
            for i in range(0,len(EltTip)):
                if (len(np.intersect1d(self.mesh.Connectivity[EltChannel[e]],self.mesh.Connectivity[EltTip[i]]))>0) :
                    testribbon[e]=1
                    break
                else :
                    testribbon[e]=0            
        EltRibbon=EltChannel[(np.reshape(testribbon,len(EltChannel))==1)]   # EltChannel is (N,) testribbon is (N,1)
        
          
        # Get the initial Filling fraction as well as location of the intersection of the crack front with the edges of the mesh  
        # we loop over all the tip element  (partially fractured element)
          
        EltArea=self.mesh.EltArea
    
        FillF=np.empty([len(EltTip)],dtype=float)  # a vector containing the filling fraction of each Tip Elements
        I=np.empty([len(EltTip),2],dtype=float)    # a vector containing the coordinantes of the intersection of the front with the edges of each Tip Element - I point
        J=np.empty([len(EltTip),2],dtype=float)    # a vector containing the coordinantes of the intersection of the front with the edges of each Tip Element - J point
        
        for i in range(0,len(EltTip)) :
        
            ptsV=self.mesh.VertexCoor[self.mesh.Connectivity[EltTip[i]]]  ;#
            levelV=np.reshape(phiVertices[self.mesh.Connectivity[EltTip[i]]],4) ; # level set value at the vertices of this element
            s=np.argsort(levelV); # sort the level set  
            furthestin=s[0];   # vertex the furthest inside the fracture
            InsideFrac= 1 * (levelV<0.)  ; # array of 0 and 1 
    
            if np.sum(InsideFrac)==1 :
                # case 1 vertex in the fracture
                Ve=np.where(InsideFrac==1)[0]  # corresponding vertex indices in the reference element
                x = np.sqrt(self.initRad**2-(ptsV[Ve,1][0])**2) # zero of the level set in x direction (same y as Ve)
                y = np.sqrt(self.initRad**2-(ptsV[Ve,0][0])**2) # zero of the level set in y direction (same x as Ve)
                if (x<np.around(ptsV[0,0],8)) | (x> np.around(ptsV[1,0],8)) :      # note the np.around(,8)  to avoid running into numerical precision issue
                    x=-x           
                if (y<np.around(ptsV[0,1],8)) | (y > np.around(ptsV[3,1],8) ) :
                    y=-y
                        
                if (Ve==0 | Ve==2) :      
                    # case it is 0 node or 2 node
                    I[i]=np.array([ x, ptsV[Ve,1][0] ])
                    J[i]=np.array([ ptsV[Ve,0][0], y ])
                else :
                    J[i]=np.array([ x, ptsV[Ve,1][0] ])
                    I[i]=np.array([ ptsV[Ve,0][0], y ])
                
                # the filling fraction is simple to compute - area of the triangle Ve-IJ - divided by EltArea
                FillF[i] = 0.5*np.linalg.norm(I[i]-ptsV[Ve])*np.linalg.norm(J[i]-ptsV[Ve])/EltArea
    
    
            if  np.sum(InsideFrac)==2 :         
            # case of 2 vertices inside the fracture (and 2 outside)
                Ve=np.where(InsideFrac==1)[0]
                if (np.sum(Ve == np.array([0,1]))==2) | (np.sum(Ve == np.array([2,3]))==2)  : 
                    # case where the front is mostly horizontal i.e. [0-1] or [2,3]
                    y1 = np.sqrt(self.initRad**2-(ptsV[Ve[0],0])**2)
                    y2 = np.sqrt(self.initRad**2-(ptsV[Ve[1],0])**2)
                    if (y1<np.around(ptsV[0,1],8)) | (y1 > np.around(ptsV[3,1],8)) :
                        y1=-y1  
                    if (y2<np.around(ptsV[0,1],8)) | (y2 > np.around(ptsV[3,1],8)) :
                        y2=-y2  
                    if (furthestin == 0)  | (furthestin == 2 ) :
                        I[i]=np.array([(ptsV[Ve[0],0]),y1]);
                        J[i]=np.array([(ptsV[Ve[1],0]),y2]);
                        FillF[i]= 0.5*(np.linalg.norm(I[i]-ptsV[Ve[0]])+np.linalg.norm(J[i]-ptsV[Ve[1]]))*(np.linalg.norm(ptsV[Ve[0]]-ptsV[Ve[1]]))/EltArea
                    else :
                        J[i]=np.array([(ptsV[Ve[0],0]),y1]);
                        I[i]=np.array([(ptsV[Ve[1],0]),y2]);
                        FillF[i]= 0.5*(np.linalg.norm(I[i]-ptsV[Ve[1]])+np.linalg.norm(J[i]-ptsV[Ve[0]]))*(np.linalg.norm(ptsV[Ve[0]]-ptsV[Ve[1]]))/EltArea
                else :
                    # case where the front is mostly vertical i.e. [0-3] or [1,2]
                    x1 = np.sqrt(self.initRad**2-(ptsV[Ve[0],1])**2)
                    x2 = np.sqrt(self.initRad**2-(ptsV[Ve[1],1])**2)
                    if (x1 < (np.around(ptsV[0,0],8))) | (x1 > (np.around(ptsV[1,0],8))) :
                        x1=-x1
                    if (x2 < np.around(ptsV[0,0],8)) | (x2 > np.around(ptsV[1,0],8)) :
                        x2=-x2
                    if (furthestin == 0) | (furthestin == 2 ) :
                        I[i]=np.array([x1,(ptsV[Ve[0],1])]);
                        J[i]=np.array([x2,(ptsV[Ve[1],1]) ]);
                        FillF[i]= 0.5*(np.linalg.norm(I[i]-ptsV[Ve[0]])+np.linalg.norm(J[i]-ptsV[Ve[1]]))*(np.linalg.norm(ptsV[Ve[0]]-ptsV[Ve[1]]))/EltArea
                    else :
                        J[i]=np.array([x1,(ptsV[Ve[0],1])]);
                        I[i]=np.array([x2,(ptsV[Ve[1],1]) ]);
                        FillF[i]= 0.5*(np.linalg.norm(I[i]-ptsV[Ve[1]])+np.linalg.norm(J[i]-ptsV[Ve[0]]))*(np.linalg.norm(ptsV[Ve[0]]-ptsV[Ve[1]]))/EltArea
                
            if  np.sum(InsideFrac)==3 :     
                # only one vertices outside the fracture
                # we redo the same than for case 1 but Ve now corresponds to the only vertex outside the fracture
                Ve=np.where(InsideFrac==0)[0]
                x = np.sqrt(self.initRad**2-(ptsV[Ve,1][0])**2)
                y = np.sqrt(self.initRad**2-(ptsV[Ve,0][0])**2)
                if (x<np.around(ptsV[0,0],8)) | (x> np.around(ptsV[1,0],8)) :
                    x=-x
                if (y<np.around(ptsV[0,1],8)) | (y > np.around(ptsV[3,1],8)) :
                    y=-y       
                if (Ve==0 | Ve==2) :      
              # case it is 
                    J[i]=np.array([ x, ptsV[Ve,1][0] ])
                    I[i]=np.array([ ptsV[Ve,0][0], y ])
                else :
                    I[i]=np.array([ x, ptsV[Ve,1][0] ])
                    J[i]=np.array([ ptsV[Ve,0][0], y ])
          
                FillF[i] =1.- 0.5*np.linalg.norm(I[i]-ptsV[Ve])*np.linalg.norm(J[i]-ptsV[Ve])/EltArea
        
        # Type of each cell        
        CellStatus             = np.zeros((self.mesh.NumberOfElts,),dtype=np.uint8)
        CellStatus[:]          = 0  
        CellStatus[EltChannel] = 1
        CellStatus[EltTip]     = 2
        CellStatus[EltRibbon]  = 3
        
        InCrack                 = np.zeros((self.mesh.NumberOfElts,),dtype=np.uint8)
        InCrack[EltCrack]       = 1
        
        self.sgndDist = RadiusLevelSet(self.mesh.CenterCoor,self.initRad)
        
#        self.ZeroVertex = np.zeros((len(EltTip)),int)
#        for i in range(0,len(EltTip)):
#            if self.mesh.CenterCoor[i,0]<=0 and self.mesh.CenterCoor[i,1]<=0:
#                self.ZeroVertex[i] = 0
#            elif self.mesh.CenterCoor[i,0]>=0 and self.mesh.CenterCoor[i,1]<=0:
#                self.ZeroVertex[i] = 1
#            elif self.mesh.CenterCoor[i,0]<=0 and self.mesh.CenterCoor[i,1]>=0:
#                self.ZeroVertex[i] = 3
#            elif self.mesh.CenterCoor[i,0]>=0 and self.mesh.CenterCoor[i,1]>=0:
#                self.ZeroVertex[i] = 2
        
        (self.EltTip, self.l, self.alpha, CSt)= TrackFront(self.sgndDist, EltChannel, self.mesh)
        self.FillF = FillF[np.arange(EltTip.shape[0])[np.in1d(EltTip,self.EltTip)]]
        (self.EltChannel,self.EltRibbon,self.EltCrack) = (EltChannel,EltRibbon,EltCrack)
        (self.Ffront,self.CellStatus,self.InCrack) = (np.concatenate((I,J),axis=1),CellStatus,InCrack)
        self.v = v*np.ones((len(self.l)),float)
        
        distFrmCentr = (self.mesh.CenterCoor[:,0]**2 + self.mesh.CenterCoor[:,1]**2)**0.5
        self.Tarrival = np.zeros((self.mesh.NumberOfElts,),dtype=np.float) 
        self.Tarrival[:] = np.nan
        # using Mtilde solution to initialize arrival time
        self.Tarrival[self.EltChannel] = self.Cprime[self.EltChannel]**2*distFrmCentr[self.EltChannel]**4*np.pi**4/sum(self.Q)**2/4
        self.Leakedoff = np.zeros((self.mesh.NumberOfElts,),dtype=float)
        self.Leakedoff[self.EltChannel]= 2*self.Cprime[self.EltChannel]*self.mesh.EltArea*(self.time-self.Tarrival[self.EltChannel])**0.5
        self.Leakedoff[self.EltTip]    = 2*self.Cprime[self.EltTip]*VolumeIntegral(self.alpha, self.l, self.mesh.hx, self.mesh.hy, 'Lk', self.Kprime[self.EltTip], self.Eprime, self.muPrime[self.EltTip], self.Cprime[self.EltTip], self.v)

#        D=np.resize(self.Leakedoff,(self.mesh.ny,self.mesh.nx))
#        plt.matshow(D)
#        cm.colorbar
#        plt.axis('equal')
#        plt.pause(0.01)
###############################################################################

    def InitializePKN(self,initValue,initType,h):
        """ Initialize the fracture, according to the given initial value and the propagation regime. Either initial radius or time can be given 
            as the initial value. The function sets up the fracture front and other fracture parameters according to the PKN fracture geometry for the 
            given time or radius.
            
            Arguments:
                initValue (float):      initial value
                initType (string):      Possible values:
                                            time        -- indicating the given value is initial time
                                            radius      -- indicating the given value is initial radius
                h (float):              PKN fracture height
        """
        (sol_l,self.p,self.w,v,PKN) = PKNSolution(self.Eprime,sum(self.Q),np.mean(self.muPrime),self.mesh,initValue,h)
    
        Cdist = (self.mesh.CenterCoor[:,0]**2+self.mesh.CenterCoor[:,1]**2)**0.5
        grt = np.where(abs(self.mesh.CenterCoor[:,0])+self.mesh.hx/2>sol_l)
        lss = np.where(abs(self.mesh.CenterCoor[:,0])-self.mesh.hx/2<sol_l)
        tip = np.intersect1d(grt,lss)
        tiph = tip[np.where(abs(self.mesh.CenterCoor[tip,1])<h/2)[0]]

        grt = np.where(abs(self.mesh.CenterCoor[:,1])+self.mesh.hy/2>h/2)
        lss = np.where(abs(self.mesh.CenterCoor[:,1])-self.mesh.hy/2<h/2)
        tip = np.intersect1d(grt,lss)
        tipv = tip[np.where(abs(self.mesh.CenterCoor[tip,0])<sol_l)[0]]
        self.EltTip = np.append(tipv,tiph)
        
        eltsRibbon  = np.array([],int)
        for i in range(0,len(self.EltTip)):
            neighbors  = np.asarray(Neighbors(self.EltTip[i],self.mesh.nx,self.mesh.ny))
    
            if Cdist[neighbors[0]]<= Cdist[neighbors[1]]:
                eltsRibbon = np.append(eltsRibbon, neighbors[0])
            
            else:
                eltsRibbon = np.append(eltsRibbon, neighbors[1])
                        
            if Cdist[neighbors[2]]<= Cdist[neighbors[3]]:
                eltsRibbon = np.append(eltsRibbon, neighbors[2])
            
            else:
                eltsRibbon = np.append(eltsRibbon, neighbors[3])
            
        eltsRibbon = np.unique(eltsRibbon)
        for i in range(0,len(self.EltTip)):
            eltsRibbon = np.delete(eltsRibbon,np.where(eltsRibbon==self.EltTip[i]))
        
        self.EltRibbon  = eltsRibbon
        self.EltCrack   = PKN
        
        self.EltChannel = self.EltCrack
        for i in range(0,len(self.EltTip)):
            self.EltChannel = np.delete(self.EltChannel,np.where(self.EltChannel==self.EltTip[i]))
        self.v = np.asarray([v],float)
        
        src = np.where(abs(self.mesh.CenterCoor[:,0])<self.mesh.hx)[0]
        src = src[np.where(abs(self.mesh.CenterCoor[src,1])<h/2-self.mesh.hy)[0]]
        Q0 = sum(self.Q)        
        self.Q= np.zeros((self.mesh.NumberOfElts),float)
        self.Q[src] = Q0/len(src)
        
        sgndDist                    = 1e10*np.ones((self.mesh.NumberOfElts,),float); # uncalculated cells get very large value
        sgndDist[self.EltChannel]   = 0
#        sgndDist[self.EltRibbon]    = -0.001*TipAsymInversion(self.w,self.EltRibbon,self.Kprime,self.Eprime,'K')
        
        for i in range(0,len(self.EltRibbon)):
            neighbors  = np.asarray(Neighbors(self.EltRibbon[i],self.mesh.nx,self.mesh.ny))
            if np.where(self.EltTip==neighbors[0])[0].size>0:
                sgndDist[self.EltRibbon[i]]=-(sol_l-abs(self.mesh.CenterCoor[self.EltRibbon[i],0]))
            if np.where(self.EltTip==neighbors[1])[0].size>0:
                sgndDist[self.EltRibbon[i]]=-(sol_l-abs(self.mesh.CenterCoor[self.EltRibbon[i],0]))
            if np.where(self.EltTip==neighbors[2])[0].size>0:
                sgndDist[self.EltRibbon[i]]=-(h/2-abs(self.mesh.CenterCoor[self.EltRibbon[i],1]))
            if np.where(self.EltTip==neighbors[3])[0].size>0:
                sgndDist[self.EltRibbon[i]]=-(h/2-abs(self.mesh.CenterCoor[self.EltRibbon[i],1]))
        
        
        self.ZeroVertex=np.zeros((len(self.EltTip),),int)
        for i in range(0,len(self.EltTip)):
            if self.mesh.CenterCoor[self.EltTip[i],0] < 0 and self.mesh.CenterCoor[self.EltTip[i],1]<0:
                self.ZeroVertex[i] = 0
            if self.mesh.CenterCoor[self.EltTip[i],0] > 0 and self.mesh.CenterCoor[self.EltTip[i],1]<0:
                self.ZeroVertex[i] = 1
            if self.mesh.CenterCoor[self.EltTip[i],0] < 0 and self.mesh.CenterCoor[self.EltTip[i],1]>0:
                self.ZeroVertex[i] = 3
            if self.mesh.CenterCoor[self.EltTip[i],0] > 0 and self.mesh.CenterCoor[self.EltTip[i],1]>0:
                self.ZeroVertex[i] = 2
                    
        SolveFMM(sgndDist, self.EltRibbon, self.EltChannel, self.mesh)
        
#        D=np.resize(sgndDist,(self.mesh.ny,self.mesh.nx))
#        plt.matshow(D)
##        cm.colorbar
#        plt.axis('equal')
#        plt.pause(0.01)
        
        (ElmntTip, self.l, self.alpha, CellStatusNew) = TrackFront(sgndDist, self.EltChannel, self.mesh)
        self.FillF = VolumeIntegral(self.alpha, self.l, self.mesh.hx, self.mesh.hy, 'A', self.Kprime[ElmntTip], self.Eprime, self.muPrime[ElmntTip], self.Cprime[ElmntTip], np.ones((ElmntTip.size,),))/self.mesh.EltArea # Calculate filling fraction for current iteration
                
        self.InCrack                 = np.zeros((self.mesh.NumberOfElts,),dtype=np.uint8)
        self.InCrack[self.EltCrack]  = 1
        self.sgndDist = sgndDist
        self.time = initValue
        self.EltTip = ElmntTip
        
    def PlotFracture(self,Elem_Identifier,Parameter_Identifier,analytical=0,evol=False,identify=[]):
        """3D plot of all elements given in the form of a list;
        
            Arguments:
                Elem_Identifier(string):        elements to be printed; possible options:
                                                    complete
                                                    channel
                                                    crack
                                                    ribbon
                Parameter_Identifier(string):   parameter to be ploted; possible options:
                                                    width
                                                    pressure
                                                    viscocity
                                                    footPrint
                analytical (float):             radius of fracture footprint calculated analytically; not plotted if not given
                evol (boolean):                 fracture evolution plot flag. Set to true will print fracture evolution with time
                identify (ndarray):             Give the cells in the list with different color to identify
        """    
    
        if Elem_Identifier == 'complete':
            Elts = np.arange(self.mesh.NumberOfElts)
        elif Elem_Identifier == 'channel':
            Elts = self.EltChannel
        elif Elem_Identifier == 'crack':
            Elts = self.EltCrack
        elif Elem_Identifier == 'ribbon':
            Elts = self.EltRibbon
        elif Elem_Identifier == 'tip':
            Elts = self.EltTip
        else:
            print('invalid element identifier')
            return
        
        values = np.zeros((self.mesh.NumberOfElts),float)
        if Parameter_Identifier == 'width':
            values[Elts] = self.w[Elts]
        elif Parameter_Identifier == 'pressure':
            values[Elts] = self.p[Elts]
        elif Parameter_Identifier == 'muPrime':
            values[Elts] = self.muPrime[Elts]
        elif Parameter_Identifier == 'footPrint':
            self.PrintFractureTrace(analytical,evol,identify)
            return
        else:
            print('invalid parameter identifier')
            return
            
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(self.mesh.CenterCoor[:,0],self.mesh.CenterCoor[:,1],values, cmap=cm.jet, linewidth=0.2)
        plt.show()
        plt.pause(0.0001)

        

######################################


    def PrintFractureTrace(self, rAnalytical, evol, identify):
        """ Print fracture front and different regions of the fracture
            Arguments:
                rAnalytical (float):    radius of fracture footprint calculated analytically
                evol (boolean):         print fracture evolution flag (see PlotFracture function)
                identify (ndarray):     list of cells to be identified (see PlotFracture function)
        """
    
        pnt1 = np.zeros((2,len(self.l))) #fracture front intersection with the grid lines
        pnt2 = np.zeros((2,len(self.l)))
    
    #    pnt3 = np.zeros((2,len(l))) #Perpendicular to the fracture front inside the grid cells
    #    pnt4 = np.zeros((2,len(l)))
    
    
        for i in range(0,len(self.l)):
            if self.alpha[i]!=0 and self.alpha[i]!=math.pi/2:
                yIntrcpt = self.l[i]/math.cos(math.pi/2-self.alpha[i]);
                grad     = -1/math.tan(self.alpha[i]);
    
                if Pdistance(0, self.mesh.hy, grad, yIntrcpt)<=0:
                    pnt1[0,i] = 0
                    pnt1[1,i] = yIntrcpt
                else:
                    pnt1[0,i] = (self.mesh.hy-yIntrcpt)/grad
                    pnt1[1,i] = self.mesh.hy
    
                if Pdistance(self.mesh.hx, 0, grad, yIntrcpt)<=0:
                    pnt2[0,i] = -yIntrcpt/grad
                    pnt2[1,i] = 0
                else:
                    pnt2[0,i] = self.mesh.hx
                    pnt2[1,i] = yIntrcpt + grad*self.mesh.hx
    
    #        pnt3[0,i] = self.l[i]*math.cos(self.alpha[i])
    #        pnt3[1,i] = math.tan(self.alpha[i])*self.l[i]*math.cos(self.alpha[i])
    
            if self.alpha[i]==0:
                pnt1[0,i] = self.l[i]
                pnt1[1,i] = self.mesh.hy
                pnt2[0,i] = self.l[i]
                pnt2[1,i] = 0
    
            if self.alpha[i]==math.pi/2:
                pnt1[0,i] = 0
                pnt1[1,i] = self.l[i]
                pnt2[0,i] = self.mesh.hx
                pnt2[1,i] = self.l[i]
    
#   to plot the perpendicular on the front.
    #        pnt1[0,i] = pnt1[0,i] + self.mesh.VertexCoor[self.mesh.Connectivity[self.EltTip[i],0],0]
    #        pnt1[1,i] = pnt1[1,i] + self.mesh.VertexCoor[self.mesh.Connectivity[self.EltTip[i],0],1]
    #        pnt2[0,i] = pnt2[0,i] + self.mesh.VertexCoor[self.mesh.Connectivity[self.EltTip[i],0],0]
    #        pnt2[1,i] = pnt2[1,i] + self.mesh.VertexCoor[self.mesh.Connectivity[self.EltTip[i],0],1]
    #
    #        pnt3[0,i] = pnt3[0,i] + self.mesh.VertexCoor[self.mesh.Connectivity[self.EltTip[i],0],0]
    #        pnt3[1,i] = pnt3[1,i] + self.mesh.VertexCoor[self.mesh.Connectivity[self.EltTip[i],0],1]
    #        pnt4[0,i] = self.mesh.VertexCoor[self.mesh.Connectivity[self.EltTip[i],0],0]
    #        pnt4[1,i] = self.mesh.VertexCoor[self.mesh.Connectivity[self.EltTip[i],0],1]
    
            if self.ZeroVertex[i]==0:
                pnt1[0,i] = pnt1[0,i] + self.mesh.VertexCoor[self.mesh.Connectivity[self.EltTip[i],self.ZeroVertex[i]],0]
                pnt1[1,i] = pnt1[1,i] + self.mesh.VertexCoor[self.mesh.Connectivity[self.EltTip[i],self.ZeroVertex[i]],1]
                pnt2[0,i] = pnt2[0,i] + self.mesh.VertexCoor[self.mesh.Connectivity[self.EltTip[i],self.ZeroVertex[i]],0]
                pnt2[1,i] = pnt2[1,i] + self.mesh.VertexCoor[self.mesh.Connectivity[self.EltTip[i],self.ZeroVertex[i]],1]
    
            if self.ZeroVertex[i]==1:
                pnt1[0,i] = -pnt1[0,i] + self.mesh.VertexCoor[self.mesh.Connectivity[self.EltTip[i],self.ZeroVertex[i]],0]
                pnt1[1,i] = pnt1[1,i] + self.mesh.VertexCoor[self.mesh.Connectivity[self.EltTip[i],self.ZeroVertex[i]],1]
                pnt2[0,i] = -pnt2[0,i] + self.mesh.VertexCoor[self.mesh.Connectivity[self.EltTip[i],self.ZeroVertex[i]],0]
                pnt2[1,i] = pnt2[1,i] + self.mesh.VertexCoor[self.mesh.Connectivity[self.EltTip[i],self.ZeroVertex[i]],1]
    
            if self.ZeroVertex[i]==3:
                pnt1[0,i] = pnt1[0,i] + self.mesh.VertexCoor[self.mesh.Connectivity[self.EltTip[i],self.ZeroVertex[i]],0]
                pnt1[1,i] = -pnt1[1,i] + self.mesh.VertexCoor[self.mesh.Connectivity[self.EltTip[i],self.ZeroVertex[i]],1]
                pnt2[0,i] = pnt2[0,i] + self.mesh.VertexCoor[self.mesh.Connectivity[self.EltTip[i],self.ZeroVertex[i]],0]
                pnt2[1,i] = -pnt2[1,i] + self.mesh.VertexCoor[self.mesh.Connectivity[self.EltTip[i],self.ZeroVertex[i]],1]
    
            if self.ZeroVertex[i]==2:
                pnt1[0,i] = -pnt1[0,i] + self.mesh.VertexCoor[self.mesh.Connectivity[self.EltTip[i],self.ZeroVertex[i]],0]
                pnt1[1,i] = -pnt1[1,i] + self.mesh.VertexCoor[self.mesh.Connectivity[self.EltTip[i],self.ZeroVertex[i]],1]
                pnt2[0,i] = -pnt2[0,i] + self.mesh.VertexCoor[self.mesh.Connectivity[self.EltTip[i],self.ZeroVertex[i]],0]
                pnt2[1,i] = -pnt2[1,i] + self.mesh.VertexCoor[self.mesh.Connectivity[self.EltTip[i],self.ZeroVertex[i]],1]
                
        
        tmp = np.transpose(pnt1)
        tmp = np.hstack((tmp,np.transpose(pnt2)))
        if evol:
            self.FractEvol=np.vstack((self.FractEvol,tmp))
            PlotMeshFractureTrace(self.mesh,self.EltTip,self.EltChannel,self.EltRibbon,self.FractEvol[:,0:2],self.FractEvol[:,2:4],rAnalytical,self.sigma0,identify)
        else:
            PlotMeshFractureTrace(self.mesh,self.EltTip,self.EltChannel,self.EltRibbon,tmp[:,0:2],tmp[:,2:4],rAnalytical,self.sigma0,identify)
            
        


######################################     
    def Propagate(self,dt,C,tol_frntPos,tol_Picard,regime='U',maxitr=25):
        """ Propagate fracture one time step
            Arguments:
                dt (float):         time step
                C (ndarray-float):  Elasticity matrix
                tol_frntPos(float): tolerance for the front position iteration. The front position is assumed to be converged
                                    if the norm of current iteration is below this tolerance.
                tol_Picard(float):  tolerance for Picard iteration.
                regime (string):    propagation regime. Possible options:
                                        regime -- A  gives the area (fill fraction)
                                        regime -- K  gives tip volume according to the square root assymptote
                                        regime -- M  gives tip volume according to the viscocity dominated assymptote 
                                        regime -- Lk is used to calculate the leak off given the distance of the front l (note, its not tip volume) 
                                        regime -- Mt gives tip volume according to the viscocity, Leak-off assymptote 
                                        regime -- U  gives tip volume according to the Universal assymptote (Donstov and Pierce, 2017)
                                        regime -- MK gives tip volume according to the M-K transition assymptote
                maxitr (int)        maximum iterations to find front position (default 25).
        """
        
        norm    = 1000
        print('Elememts currently in crack:  ' + repr(len(self.EltCrack)))
    
        C_EltTip = C[np.ix_(self.EltTip,self.EltTip)] 
        for e in range(0,len(self.EltTip)) :
            r=self.FillF[e]-.25;
            if r<0.1 :
                r=0.1
            ac=(1-r)/r;
            C[self.EltTip[e],self.EltTip[e]]=C[self.EltTip[e],self.EltTip[e]]*(1.+ac*np.pi/4.)
        

        guess = dt*sum(self.Q)/self.EltCrack.size*np.ones((self.EltCrack.size,), float) # average injected fluid over footprint
        print('Solving non linear system with same footprint')
        
        DLkOff              = np.zeros((self.mesh.NumberOfElts,),float)
#        TarrivalNode        = self.time-dt - self.l/self.v
#        TarrivalMid         = ((self.time-dt)+TarrivalNode)/2
#        if ((self.time-dt-TarrivalMid)<0).any():
#            TarrivalMid[np.where((T-dt-TarrivalMid)<0)]=T-dt
#        DLkOff[self.EltTip]         = 2*self.Cprime[self.EltTip]*self.FillF*self.mesh.EltArea* ((self.time-TarrivalMid)**0.5-(self.time-dt-TarrivalMid)**0.5)
#        DLkOff[self.EltChannel]     = 2*self.Cprime[self.EltChannel]*self.mesh.EltArea*((self.time - self.Tarrival[self.EltChannel])**0.5 - (self.time-dt - self.Tarrival[self.EltChannel])**0.5)

        sol     =   ElastoHydrodynamicSolver_SameFP(guess,tol_Picard,self.w,self.mesh.NeiElements,self.EltCrack,dt,self.Q,C,self.mesh.EltArea,self.muPrime,self.mesh,self.InCrack,DLkOff,self.sigma0)
                                            
        C[np.ix_(self.EltTip,self.EltTip)]=C_EltTip  #retain origional C (without fill fraction correction)

        w_k          = np.copy(self.w)
        w_k[self.EltCrack] = w_k[self.EltCrack]+sol

        
        print('Starting iteration')
        itrcount = 1
        
        FillFrac_km1 = []    # filling fraction last iteration; used to calculate norm
        DLkOffEltChannel = DLkOff[self.EltChannel]
    #   Fracture front loop
        while (norm>tol_frntPos) :
                    
            #Initialization the signed distance in the ribbon element - by inverting the tip asymptotics    
            sgndDist_k                  = 1e10*np.ones((self.mesh.NumberOfElts,),float); # uncalculated cells get very large value
            sgndDist_k[self.EltChannel] = 0
            sgndDist_k[self.EltRibbon]  = -TipAsymInversion(w_k,self.EltRibbon,self.Kprime,self.Eprime,regime,self.muPrime,self.Cprime,self.sgndDist,dt)
            if np.isnan(sgndDist_k[self.EltRibbon]).any():
                print('Tip inversion is not correct'+'\n time step failed .............')
                self.exitstatus = 7
                return
            
            SolveFMM(sgndDist_k, self.EltRibbon, self.EltChannel, self.mesh)    # solve Eikonal eq via Fast Marching Method starting from the element close to the ribbon elt (i.e. the Tip element of the last time step)
                
#            PrintDomain(np.arange(self.mesh.NumberOfElts),sgndDist_k,self.mesh)
#            plt.pause(1)
            if max(sgndDist_k)==1e10:
                print('FMM not worked properly = '+repr(np.where(sgndDist_k==1e10))+'\ntime step failed .............')
                self.exitstatus = 2
                return

            print('Calculating filling fraction of tip elements with new front location...')
            (EltsTipNew, l_k, alpha_k, CellStatus)= TrackFront(sgndDist_k, self.EltChannel, self.mesh)  # gets the new tip elements & \ell_k & alpha_k (also containing the elements which are fully filled after the front is moved outward)
            
            #check if still in the grid            
            tipNeighb = self.mesh.NeiElements[EltsTipNew,:]
            for i in range(0,len(EltsTipNew)):
                if (np.where(tipNeighb[i,:]==EltsTipNew[i])[0]).size>0:
                    raise SystemExit('Reached end of the grid. exiting....')
            
            InCrack_k   = np.zeros((self.mesh.NumberOfElts,),dtype=np.uint8)
            InCrack_k[self.EltChannel] = 1
            InCrack_k[EltsTipNew] = 1

            Vel_k =  -(sgndDist_k[EltsTipNew] - self.sgndDist[EltsTipNew])/dt

            
            FillFrac_k     = VolumeIntegral(alpha_k, l_k, self.mesh.hx, self.mesh.hy, 'A', self.Kprime[EltsTipNew], self.Eprime, self.muPrime[EltsTipNew], self.Cprime[EltsTipNew], Vel_k)/self.mesh.EltArea # Calculate filling fraction for current iteration
            # some of the list are redundant to calculate on each iteration            
            (EltChannel_k, EltTip_k, EltCrack_k, EltRibbon_k, zrVertx_k, CellStatus_k) = UpdateLists(self.EltChannel, EltsTipNew, FillFrac_k, sgndDist_k, self.mesh) # Evaluate the element lists for current iteration
            
            NewTipinTip = np.arange(EltsTipNew.shape[0])[np.in1d(EltsTipNew, EltTip_k)]  # EletsTipNew may contain fully filled elements also  
            
            if len(FillFrac_km1)==len(FillFrac_k):
                norm = np.linalg.norm(FillFrac_k-FillFrac_km1)
            else:
                norm = 1
            print('Norm with new filling fraction = '+repr(norm))
            
#            highStress = np.append(np.where(self.mesh.CenterCoor[:,0]>0.048/2+self.mesh.hx),np.where(self.mesh.CenterCoor[:,0]<-0.048/2-self.mesh.hx))
##           np.where(self.Kprime!=self.Kprime[self.mesh.CenterElts[0]])
#            
#            for i in range(0,len(EltsTipNew)):
#                if np.where(highStress==EltsTipNew[i])[0].size > 0:
#                    alpha_k[i]=0
                    
            nan = np.logical_or(np.isnan(alpha_k),np.isnan(l_k))
            if nan.any():
#                problem = np.where(nan)[0]
#                for i in range(0,len(problem)):
#                    neighbors  = np.asarray(Neighbors(EltsTipNew[problem[i]],self.mesh.nx,self.mesh.ny))
#                    inTip = np.asarray([],int)
#                    for j in range(0,len(neighbors)):
#                        inTip = np.append(inTip,np.where(EltsTipNew==neighbors[j]))
#                    alpha_k[problem[i]]=np.mean(alpha_k[inTip])
                    
                    
                print('Front is not tracked correctly, '+ 'problem in cell(s) '+repr(EltsTipNew[np.where(nan)])+'\ntime step failed .............')
                self.exitstatus = 3
                return
            
          
            if norm>tol_frntPos:
                stagnant     = abs(1-sgndDist_k[EltsTipNew]/self.sgndDist[EltsTipNew]) < 1e-8  #tip cells whose distance from front has not changed.
                if stagnant.any():
                    KIPrime = StressIntensityFactor(w_k,sgndDist_k,EltsTipNew,EltRibbon_k,stagnant,self.mesh,self.Eprime)       # calculate stress intensity factor for stagnant cells
                    if np.isnan(KIPrime).any():
                        np.where(np.isnan(KIPrime))
                        print('Ribbon element not found in the enclosure of tip cell. tip cell '+repr(EltsTipNew[np.where(np.isnan(KIPrime))])+'\n time step failed .............')
                        self.exitstatus = 8
                        return
                    wTip        = VolumeIntegral(alpha_k, l_k, self.mesh.hx, self.mesh.hy, regime, self.Kprime[EltsTipNew], self.Eprime, self.muPrime[EltsTipNew], self.Cprime[EltsTipNew], Vel_k, stagnant, KIPrime)/self.mesh.EltArea    
                else:
                    wTip        = VolumeIntegral(alpha_k, l_k, self.mesh.hx, self.mesh.hy, regime, self.Kprime[EltsTipNew], self.Eprime, self.muPrime[EltsTipNew], self.Cprime[EltsTipNew], Vel_k)/self.mesh.EltArea
                

                smallNgtvWTip = np.where(np.logical_and(wTip<0, wTip>-10**-4*np.mean(wTip))) # check if the tip volume has gone into negative
                if np.asarray(smallNgtvWTip).size>0:
#                    warnings.warn("Small negative volume integral(s) received, ignoring "+repr(wTip[smallngtvwTip])+' ...')
                    wTip[smallNgtvWTip]= abs(wTip[smallNgtvWTip])
                
                if (wTip<-10**-4*np.mean(wTip)).any():
                    print('wTip not right'+'\n time step failed .............')
                    self.exitstatus = 4
                    return
   
                DLkOff              = np.zeros((self.mesh.NumberOfElts,),float)
#                LkOffTipTn          = 2*self.Cprime[EltsTipNew]*VolumeIntegral(alpha_k, l_k, self.mesh.hx, self.mesh.hy, 'Lk', self.Kprime[EltsTipNew], self.Eprime, self.muPrime[EltsTipNew], self.Cprime[EltsTipNew], Vel_k)
#                DLkOff[EltsTipNew]  = LkOffTipTn-self.Leakedoff[EltsTipNew]
#                DLkOff[self.EltChannel] = DLkOffEltChannel
#                PrintDomain(np.arange(self.mesh.NumberOfElts),DLkOff,self.mesh)
                
         
                guess = np.zeros((self.EltChannel.size+EltsTipNew.size,), float)
                pguess = self.p[EltsTipNew]
                
        
                guess[np.arange(self.EltChannel.size)] = dt*sum(self.Q)/self.EltCrack.size*np.ones((self.EltCrack.size,), float)        
                guess[self.EltChannel.size+np.arange(EltsTipNew.size)] = pguess
    
                print('Not converged, solving non linear system with extended footprint')
                sol     = ElastoHydrodynamicSolver_ExtendedFP(guess,tol_Picard,self.EltChannel,EltCrack_k,EltsTipNew,self.w,wTip,self.mesh,dt,self.Q,C,self.muPrime,InCrack_k,DLkOff,self.sigma0)

#                if itrcount>30:
#                    if norm_km1-norm<1e-5:
#                        norm = 1e-4
##                    self.PlotFracture('complete','footPrint')
#                    plt.pause(1)
#                    print(repr(FillFrac_k)+repr(FillFrac_km1))
#                    sol[np.arange(self.EltChannel.size)] = 1.5*sol[np.arange(self.EltChannel.size)]
#                    itrcount=1
                w_k[self.EltChannel] =  self.w[self.EltChannel] + sol[np.arange(self.EltChannel.size)]
                if np.isnan(w_k).any() or (w_k<0).any():
                    print('width not correct. Solution obtained in the last iteration'+repr(sol)+'\n time step failed .............')
                    self.exitstatus = 5
                    return
    #            pTip_k      = sol[self.EltChannel.size+np.arange(EltsTipNew.size)]

                itrcount = itrcount +1 
                print('\niteration ' + repr(itrcount))

                if itrcount>=maxitr:
                    print('did not converge after '+repr(maxitr)+' iterations'+'\n time step failed .............')
                    self.exitstatus = 6
                    return
                FillFrac_km1 = np.copy(FillFrac_k)
            else:
                print('converged, exiting loop...')
                break
            
#        NewChannelinTip  = np.arange(EltsTipNew.shape[0])[~np.in1d(EltsTipNew, EltTip_k)]
#        EltsChannelNew  = EltsTipNew[NewChannelinTip]
#        lmbda           = self.mesh.hx*np.cos(alpha_k[NewChannelinTip])+self.mesh.hy*np.sin(alpha_k[NewChannelinTip])
#        self.Tarrival[EltsChannelNew] = (2*self.time - (l_k[NewChannelinTip]-lmbda)/Vel_k[NewChannelinTip] - l_k[NewChannelinTip]/Vel_k[NewChannelinTip])/2
#        self.Leakedoff += DLkOff     
        
        w_k[EltsTipNew]             = wTip
        self.w                      = w_k
        self.FillF                  = FillFrac_k[NewTipinTip]      
        (self.EltChannel, self.EltTip, self.EltCrack, self.EltRibbon, self.ZeroVertex) = (EltChannel_k, EltTip_k, EltCrack_k, EltRibbon_k, zrVertx_k)
        self.p[self.EltCrack]       = np.dot(C[np.ix_(self.EltCrack,self.EltCrack)],self.w[self.EltCrack])
        self.sgndDist               = sgndDist_k
        (self.alpha,self.l,self.v)  = (alpha_k[NewTipinTip],l_k[NewTipinTip],Vel_k[NewTipinTip])
        self.InCrack                = InCrack_k
        self.time += dt
        self.exitstatus = 1
        return
        
  
    def SaveFracture(self,filename):
        with open(filename, 'wb') as output:
            pickle.dump(self, output, -1)
    

            
