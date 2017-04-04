# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Thu Dec 22 16:22:33 2016.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2017. All rights reserved.
See the LICENSE.TXT file for more details.
"""


# imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pickle

# local import ....

from src.Utility import *
from src.HFAnalyticalSolutions import *
from src.ElastoHydrodynamicSolver import *
from src.TipInversion import *
from src.LevelSet import *
from src.VolIntegral import *
#from src.Domain import *
from src.Properties import *
from src.CartesianMesh import *


# TOdo : merge the __init__ with the actual initialization !!!!
# todo : clean up , comments
class Fracture():
    """ Class defining propagating fracture;
        
        Instance variables:

            muPrime (ndarray-float):    12 viscosity (mu' parameter) for each cell in the domain
            w (ndarray-float):          fracture opening (width)
            p (ndarray-float):          fracture pressure
            time (float):               time after the start of injection
            EltChannel (ndarray-int):   list of cells currently in the channel region
            EltCrack (ndarray-int):     list of cells currently in the crack region
            EltRibbon (ndarray-int):    list of cells currently in the Ribbon region
            EltTip (ndarray-int):       list of cells currently in the Tip region
            v (ndarray-float):          propagation velocity for each cell in the tip cells.
            alpha (ndarray-float):      angle prescribed by perpendicular on the fracture front
                                        (see Pierce 2015, Computation Methods Appl. Mech)
            l (ndarray-float):          length of perpendicular on the fracture front
                                        (see Pierce 2015, Computation Methods Appl. Mech)
            ZeroVertex (ndarray-float): Vertex from which the perpendicular is drawn (can have value from 0 to 3)
            FillF (ndarray-float):      filling fraction of each tip cell
            CellStatus (ndarray-int):   specifies which region each element currently belongs to
            initRad (float):            starting radius
            initTime (float):           starting time
            sgndDist (ndarray-float):   signed minimun distance from fracture front of each cell in the domain
            Q (ndarray-float):          injection rate into each cell of the domain
            FractEvol (ndarray-float):  array containing the coordinates of the individual fracture front lines;
                            used for printing fracture evolution through time
            InCrack (ndarray-int):      array specifying whether the cell is inside or outside the fracture.
            
                                            
        functions:
            InitializeRadialFracture:   set initial conditions of a radial fracture to start simulation
            InitializePKN               set initial conditions of a PKN fracture
            PlotFracture:               plot given variable of the fracture
            PrintFractureTrace:         plot current regions and front position of the fracture

            
    """
    
    def __init__(self,mesh,Fluid,Solid_Properties):
        """ Constructor for the Fracture class
            Arguments:
                mesh (CartesianMesh):           A mesh describing the domain

         """

        self.mesh =mesh

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

        self.TimeStep = 0;

        # local viscosity at grid point  (is needed by the current scheme, could be useful as a tangent value for NL viscosity
        # or case of multiple fluids....
        # becareful with copy there ! WE SHOULD DO A DEEP COPY HERE TO AVOID PROBLEMS !
        self.muPrime = Fluid.muprime *(np.ones((mesh.NumberOfElts),float) )
        self.rho = Fluid.density *(np.ones((mesh.NumberOfElts),float) )
        # normal stress.... ( should we try to avoid this one ? )
        # becareful with copy there ! WE SHOULD DO A DEEP COPY HERE TO AVOID PROBLEMS !
        self.SigmaO = Solid_Properties.SigmaO

#############################################

    def InitializeRadialFracture(self,initValue,initType,regime,Solid,Fluid,Injection):
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
        # todo ensure to use the properties close to the inj. point, first rate etc.   ... for generalization

        if initType == 'time':
            self.time = initValue
            if regime == 'K':
                (self.initRad,self.p,self.w,v) = KvertexSolutionTgiven(np.mean(Solid.Kprime), Solid.Eprime,
                                                                       Injection.injectionrate, self.mesh, initValue)
            elif regime == 'M':
                (self.initRad,self.p,self.w,v) = MvertexSolutionTgiven(Solid.Eprime, Injection.injectionrate,
                                                                       Fluid.muprime, self.mesh,initValue)
            elif regime == 'Mt':
                (self.initRad,self.p,self.w,v) = MTvertexSolutionTgiven(Solid.Eprime, np.mean(Solid.Cprime),
                                                                        Injection.injectionrate,Fluid.muprime,
                                                                        self.mesh,initValue)
            else:
                print('regime '+regime+' not supported')
                return
        elif initType == 'radius':
            self.initRad = initValue
            if regime == 'K':
                (self.time,self.p,self.w,v) = KvertexSolutionRgiven(np.mean(Solid.Kprime),Solid.Eprime,
                                                                    Injection.injectionrate,self.mesh, initValue)
            elif regime == 'M':
                (self.time,self.p,self.w,v) = MvertexSolutionRgiven(Solid.Eprime, Injection.injectionrate,Fluid.muprime,
                                                                    self.mesh, initValue)
            elif regime == 'Mt':
                (self.time,self.p,self.w,v) = MTvertexSolutionRgiven(Solid.Eprime, np.mean(Solid.Cprime),
                                                                     Injection.injectionrate,
                                                                     Fluid.muprime, self.mesh, initValue)
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

        # array of Length (number of elements) containig the sum of vertices with neg level set value)
        psum=np.sum(phiVertices[self.mesh.Connectivity[:]]<0,axis=1)
        # indices of tip element which by definition have less than 4 but at least 1 vertices inside the level set
        EltTip =(np.where(np.logical_and(psum>0,psum<4)))[0]
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
        
          
        #Get the initial Filling fraction as well as location of the intersection of the crack front with the edges
        #                               of the mesh
        # we loop over all the tip element  (partially fractured element)
          
        EltArea=self.mesh.EltArea
        # a vector containing the filling fraction of each Tip Elements     -> should be of all elements
        FillF=np.empty([len(EltTip)],dtype=float)
        # a vector containing the coordinantes of the intersection of the front with the edges of each Tip Element - I point
        I=np.empty([len(EltTip),2],dtype=float)
        # a vector containing the coordinantes of the intersection of the front with the edges of each Tip Element - J point
        J=np.empty([len(EltTip),2],dtype=float)

        for i in range(0,len(EltTip)) :
        
            ptsV=self.mesh.VertexCoor[self.mesh.Connectivity[EltTip[i]]]  ;#
            # level set value at the vertices of this element
            levelV=np.reshape(phiVertices[self.mesh.Connectivity[EltTip[i]]],4) ;
            s=np.argsort(levelV); # sort the level set  
            furthestin=s[0];   # vertex the furthest inside the fracture
            InsideFrac= 1 * (levelV<0.)  ; # array of 0 and 1 
    
            if np.sum(InsideFrac)==1 :
                # case 1 vertex in the fracture
                Ve=np.where(InsideFrac==1)[0]  # corresponding vertex indices in the reference element
                x = np.sqrt(self.initRad**2-(ptsV[Ve,1][0])**2) # zero of the level set in x direction (same y as Ve)
                y = np.sqrt(self.initRad**2-(ptsV[Ve,0][0])**2) # zero of the level set in y direction (same x as Ve)
                # note the np.around(,8)  to avoid running into numerical precision issue
                if (x<np.around(ptsV[0,0],8)) | (x> np.around(ptsV[1,0],8)) :

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
                        FillF[i]= 0.5*(np.linalg.norm(I[i]-ptsV[Ve[0]])+np.linalg.norm(J[i]-ptsV[Ve[1]]))\
                                  *(np.linalg.norm(ptsV[Ve[0]]-ptsV[Ve[1]]))/EltArea
                    else :
                        J[i]=np.array([(ptsV[Ve[0],0]),y1]);
                        I[i]=np.array([(ptsV[Ve[1],0]),y2]);
                        FillF[i]= 0.5*(np.linalg.norm(I[i]-ptsV[Ve[1]])+np.linalg.norm(J[i]-ptsV[Ve[0]]))\
                                  *(np.linalg.norm(ptsV[Ve[0]]-ptsV[Ve[1]]))/EltArea
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
                        FillF[i]= 0.5*(np.linalg.norm(I[i]-ptsV[Ve[0]])+np.linalg.norm(J[i]-ptsV[Ve[1]]))\
                                  *(np.linalg.norm(ptsV[Ve[0]]-ptsV[Ve[1]]))/EltArea
                    else :
                        J[i]=np.array([x1,(ptsV[Ve[0],1])]);
                        I[i]=np.array([x2,(ptsV[Ve[1],1]) ]);
                        FillF[i]= 0.5*(np.linalg.norm(I[i]-ptsV[Ve[1]])+np.linalg.norm(J[i]-ptsV[Ve[0]]))\
                                  *(np.linalg.norm(ptsV[Ve[0]]-ptsV[Ve[1]]))/EltArea
                
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
        
        (self.EltTip, self.l, self.alpha, CSt)= TrackFront(self.sgndDist, EltChannel, self.mesh)
        self.FillF = FillF[np.arange(EltTip.shape[0])[np.in1d(EltTip,self.EltTip)]]
        (self.EltChannel,self.EltRibbon,self.EltCrack) = (EltChannel,EltRibbon,EltCrack)
        (self.Ffront,self.CellStatus,self.InCrack) = (np.concatenate((I,J),axis=1),CellStatus,InCrack)
        self.v = v*np.ones((len(self.l)),float)
        
        self.ZeroVertex=np.zeros((len(self.EltTip),),int)
        for i in range(0,len(self.EltTip)):
            if self.mesh.CenterCoor[self.EltTip[i],0] < 0 and self.mesh.CenterCoor[self.EltTip[i],1]<0:
                self.ZeroVertex[i] = 2
            if self.mesh.CenterCoor[self.EltTip[i],0] > 0 and self.mesh.CenterCoor[self.EltTip[i],1]<0:
                self.ZeroVertex[i] = 3
            if self.mesh.CenterCoor[self.EltTip[i],0] < 0 and self.mesh.CenterCoor[self.EltTip[i],1]>0:
                self.ZeroVertex[i] = 1
            if self.mesh.CenterCoor[self.EltTip[i],0] > 0 and self.mesh.CenterCoor[self.EltTip[i],1]>0:
                self.ZeroVertex[i] = 0
        
        distFrmCentr = (self.mesh.CenterCoor[:,0]**2 + self.mesh.CenterCoor[:,1]**2)**0.5
        self.Tarrival = np.zeros((self.mesh.NumberOfElts,),dtype=np.float) 
        self.Tarrival[:] = np.nan
        # using Mtilde solution to initialize arrival time
        self.Tarrival[self.EltChannel] = Solid.Cprime[self.EltChannel]**2*distFrmCentr[self.EltChannel]**4*np.pi**4/Injection.injectionrate**2/4
        self.Leakedoff = np.zeros((self.mesh.NumberOfElts,),dtype=float)
        self.Leakedoff[self.EltChannel]= 2*Solid.Cprime[self.EltChannel]*self.mesh.EltArea*(self.time-self.Tarrival[self.EltChannel])**0.5
        self.Leakedoff[self.EltTip]    = 2*Solid.Cprime[self.EltTip]*VolumeIntegral(self.alpha, self.l, self.mesh.hx, self.mesh.hy, 'Lk',
                                                                                   Solid.Kprime[self.EltTip], Solid.Eprime,
                                                                                   self.muPrime[self.EltTip], Solid.Cprime[self.EltTip], self.v)
        # why does the log file is open here ???? it should be out of this function !!
        f = open('log', 'w+')
        from time import gmtime, strftime
        f.write('log file, program run at: ' + strftime("%Y-%m-%d %H:%M:%S", gmtime())+'\n\n\n')

#        D=np.resize(self.Leakedoff,(self.mesh.ny,self.mesh.nx))
#        plt.matshow(D)
#        cm.colorbar
#        plt.axis('equal')
#        plt.pause(0.01)
###############################################################################

# WHY IS IT SO DIFFFERENT FOR PKN ????
# todo : clean up
    def InitializePKN(self,initValue,initType,frac_height,Solid,Fluid,Injection):
        """ Initialize the fracture, according to the given initial value and the propagation regime.
                    Either initial radius or time can be given
            as the initial value. The function sets up the fracture front and other fracture parameters
                    according to the PKN fracture geometry for the
            given time or radius.
            
            Arguments:
                initValue (float):      initial value
                initType (string):      Possible values:
                                            time        -- indicating the given value is initial time
                                            radius      -- indicating the given value is initial radius
                frac_height (float):              PKN fracture height
        """
        Qo = Injection.injectionrate

        (sol_l,self.p,self.w,v,PKN) = PKNSolution(Solid.Eprime,Qo,Fluid.muPrime,self.mesh,initValue,frac_height)
    
        Cdist = (self.mesh.CenterCoor[:,0]**2+self.mesh.CenterCoor[:,1]**2)**0.5
        grt = np.where(abs(self.mesh.CenterCoor[:,0])+self.mesh.hx/2>sol_l)
        lss = np.where(abs(self.mesh.CenterCoor[:,0])-self.mesh.hx/2<sol_l)
        tip = np.intersect1d(grt,lss)
        tiph = tip[np.where(abs(self.mesh.CenterCoor[tip,1])<frac_height/2)[0]]

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
        src = src[np.where(abs(self.mesh.CenterCoor[src,1])<frac_height/2-self.mesh.hy)[0]]

        # Q0 = sum(self.Q)
        # self.Q= np.zeros((self.mesh.NumberOfElts),float)
        # self.Q[src] = Q0/len(src)

        # uncalculated cells get very large value
        sgndDist                    = 1e10*np.ones((self.mesh.NumberOfElts,),float);
        sgndDist[self.EltChannel]   = 0

        for i in range(0,len(self.EltRibbon)):
            neighbors  = np.asarray(Neighbors(self.EltRibbon[i],self.mesh.nx,self.mesh.ny))
            if np.where(self.EltTip==neighbors[0])[0].size>0:
                sgndDist[self.EltRibbon[i]]=-(sol_l-abs(self.mesh.CenterCoor[self.EltRibbon[i],0]))
            if np.where(self.EltTip==neighbors[1])[0].size>0:
                sgndDist[self.EltRibbon[i]]=-(sol_l-abs(self.mesh.CenterCoor[self.EltRibbon[i],0]))
            if np.where(self.EltTip==neighbors[2])[0].size>0:
                sgndDist[self.EltRibbon[i]]=-(frac_height/2-abs(self.mesh.CenterCoor[self.EltRibbon[i],1]))
            if np.where(self.EltTip==neighbors[3])[0].size>0:
                sgndDist[self.EltRibbon[i]]=-(frac_height/2-abs(self.mesh.CenterCoor[self.EltRibbon[i],1]))
        
        # WHAT THE HECK is the FMM is doing here ? it is an initialization !
        SolveFMM(sgndDist, self.EltRibbon, self.EltChannel, self.mesh)

        
        (ElmntTip, self.l, self.alpha, CellStatusNew) = TrackFront(sgndDist, self.EltChannel, self.mesh)
        # Calculate filling fraction for current iteration
        self.FillF = VolumeIntegral(self.alpha, self.l, self.mesh.hx, self.mesh.hy, 'A', Solid.Kprime[ElmntTip],
                                    Solid.Eprime, self.muPrime[ElmntTip],
                                    Solid.Cprime[ElmntTip], np.ones((ElmntTip.size,),))/self.mesh.EltArea

                
        self.InCrack                 = np.zeros((self.mesh.NumberOfElts,),dtype=np.uint8)
        self.InCrack[self.EltCrack]  = 1
        self.sgndDist = sgndDist
        self.time = initValue
        self.EltTip = ElmntTip
        
        self.ZeroVertex=np.zeros((len(self.EltTip),),int)
        for i in range(0,len(self.EltTip)):
            if self.mesh.CenterCoor[self.EltTip[i],0] < 0 and self.mesh.CenterCoor[self.EltTip[i],1]<0:
                self.ZeroVertex[i] = 3
            if self.mesh.CenterCoor[self.EltTip[i],0] > 0 and self.mesh.CenterCoor[self.EltTip[i],1]<0:
                self.ZeroVertex[i] = 2
            if self.mesh.CenterCoor[self.EltTip[i],0] < 0 and self.mesh.CenterCoor[self.EltTip[i],1]>0:
                self.ZeroVertex[i] = 1
            if self.mesh.CenterCoor[self.EltTip[i],0] > 0 and self.mesh.CenterCoor[self.EltTip[i],1]>0:
                self.ZeroVertex[i] = 0
                
#------------------------------------------------------------------------------------------------------
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
                analytical (float):             radius of fracture footprint calculated analytically;
                                                not plotted if not given
                evol (boolean):                 fracture evolution plot flag. Set to true will print fracture
                                                evolution with time
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
    # todo : clean up, avoid dependency on sigmaO etc.
    def PrintFractureTrace(self, rAnalytical, evol, identify ):
        """ Print fracture front and different regions of the fracture
            Arguments:
                rAnalytical (float):    radius of fracture footprint calculated analytically
                evol (boolean):         print fracture evolution flag (see PlotFracture function)
                identify (ndarray):     list of cells to be identified (see PlotFracture function)
                Mat_Properties :: solid material properties object (containing the sigma0 on each element)

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
            PlotMeshFractureTrace(self.mesh,self.EltTip,self.EltChannel,self.EltRibbon,self.FractEvol[:,0:2],
                                  self.FractEvol[:,2:4],rAnalytical,self.SigmaO,identify)
        else:
            PlotMeshFractureTrace(self.mesh,self.EltTip,self.EltChannel,self.EltRibbon,tmp[:,0:2],tmp[:,2:4],
                                  rAnalytical,self.SigmaO,identify)

##################################################################################################################

    def SaveFracture(self,filename):
        with open(filename, 'wb') as output:
            pickle.dump(self, output, -1)

