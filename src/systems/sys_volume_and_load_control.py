# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Wed Dec 28 14:43:38 2016.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# External imports
import numpy as np
from scipy.sparse.linalg import LinearOperator

def MakeEquationSystem_volumeControl_symmetric(w_lst_tmstp, wTip_sym, EltChannel_sym, EltTip_sym, C_s, dt, Q, sigma_o,
                                                          ElemArea, LkOff, vol_weights, sym_elements, dwTip):
    """
    This function makes the linear system of equations to be solved by a linear system solver. The system is assembled
    with the extended footprint (treating the channel and the extended tip elements distinctly). The the volume of the
    fracture is imposed to be equal to the fluid injected into the fracture (see Zia and Lecampion 2018).
    """

    Ccc = C_s[np.ix_(EltChannel_sym, EltChannel_sym)]
    Cct = C_s[np.ix_(EltChannel_sym, EltTip_sym)]

    A = np.hstack((Ccc, -np.ones((EltChannel_sym.size, 1),dtype=np.float64)))
    weights = vol_weights[EltChannel_sym]
    weights = np.concatenate((weights, np.array([0.0])))
    A = np.vstack((A, weights))

    S = - sigma_o[EltChannel_sym] - np.dot(Ccc, w_lst_tmstp[sym_elements[EltChannel_sym]]) - np.dot(Cct, wTip_sym)
    S = np.append(S, np.sum(Q) * dt / ElemArea - np.sum(dwTip) - np.sum(LkOff))

    return A, S

#-----------------------------------------------------------------------------------------------------------------------


def MakeEquationSystem_volumeControl_double_fracture(w_lst_tmstp, wTipFR0, wTipFR1, EltChannel0, EltChannel1, EltTip0,
                                                     EltTip1, sigma_o, C, dt, QFR0, QFR1, ElemArea, lkOff):
    """
    This function makes the linear system of equations to be solved by a linear system solver. The system is assembled
    with the extended footprint (treating the channel and the extended tip elements distinctly). The the volume of the
    fracture is imposed to be equal to the fluid injected into the fracture.
    """
    """
    Scheme of the system of equations that we are going to make

       CC0  CC01| 1 1   Dw0   sigma00
                | 1 1   Dw0   sigma00
       CC10 CC1 | 0 1 * Dw1 = sigma01
       --------------   ---   --------
       -1 -1 -1 0 0 0   Dp0   Q00*Dt/A0
       0  0  0 -1 0 0   Dp1   Q01*Dt/A1
    """
    wTip = np.concatenate((wTipFR0, wTipFR1))
    EltChannel = np.concatenate((EltChannel0, EltChannel1))
    EltTip = np.concatenate((EltTip0, EltTip1))
    Ccc = C[np.ix_(EltChannel, EltChannel)]  # elasticity Channel Channel
    Cct = C[np.ix_(EltChannel, EltTip)]

    varray0 = np.zeros((EltChannel.size, 1), dtype=np.float64)
    varray0[0:EltChannel0.size] = 1.
    varray1 = np.zeros((EltChannel.size, 1), dtype=np.float64)
    varray1[EltChannel0.size:EltChannel.size] = 1.

    A = np.hstack((Ccc, -varray0, -varray1))

    harray0 = np.zeros((1, EltChannel.size + 2), dtype=np.float64)
    harray0[0, 0:EltChannel0.size] = 1.
    harray1 = np.zeros((1, EltChannel.size + 2), dtype=np.float64)
    harray1[0, EltChannel0.size:EltChannel.size] = 1.

    A = np.vstack((A, harray0, harray1))

    S = - sigma_o[EltChannel] - np.dot(Ccc, w_lst_tmstp[EltChannel]) - np.dot(Cct, wTip)
    S = np.append(S, sum(QFR0) * dt / ElemArea - (sum(wTipFR0) - sum(w_lst_tmstp[EltTip0])) - np.sum(
        lkOff[np.concatenate((EltChannel0, EltTip0))]))
    S = np.append(S, sum(QFR1) * dt / ElemArea - (sum(wTipFR1) - sum(w_lst_tmstp[EltTip1])) - np.sum(
        lkOff[np.concatenate((EltChannel1, EltTip1))]))

    return A, S


# -----------------------------------------------------------------------------------------------------------------------


def MakeEquationSystem_volumeControl(w_lst_tmstp, wTip, EltChannel, EltTip, sigma_o, C, dt, Q, ElemArea, lkOff):
    """
    This function makes the linear system of equations to be solved by a linear system solver. The system is assembled
    with the extended footprint (treating the channel and the extended tip elements distinctly). The the volume of the
    fracture is imposed to be equal to the fluid injected into the fracture.
    """
    Ccc = C[np.ix_(EltChannel, EltChannel)]
    Cct = C[np.ix_(EltChannel, EltTip)]

    A = np.hstack((Ccc, -np.ones((EltChannel.size, 1), dtype=np.float64)))
    A = np.vstack((A, np.ones((1, EltChannel.size + 1), dtype=np.float64)))
    A[-1, -1] = 0

    S = - sigma_o[EltChannel] - np.dot(Ccc, w_lst_tmstp[EltChannel]) - np.dot(Cct, wTip)
    S = np.append(S, sum(Q) * dt / ElemArea - (sum(wTip) - sum(w_lst_tmstp[EltTip])) - np.sum(lkOff))

    return A, S

# -----------------------------------------------------------------------------------------------------------------------



def MakeEquationSystem_mechLoading(wTip, EltChannel, EltTip, C, EltLoaded, w_loaded):
    """
    This function makes the linear system of equations to be solved by a linear system solver. The system is assembled
    with the extended footprint (treating the channel and the extended tip elements distinctly). The given width is
    imposed on the given loaded elements.
    """

    Ccc = C[np.ix_(EltChannel, EltChannel)]
    Cct = C[np.ix_(EltChannel, EltTip)]

    A = np.hstack((Ccc, -np.ones((EltChannel.size, 1), dtype=np.float64)))
    A = np.vstack((A,np.zeros((1,EltChannel.size+1), dtype=np.float64)))
    A[-1, np.where(EltChannel == EltLoaded)[0]] = 1

    S = - np.dot(Cct, wTip)
    S = np.append(S, w_loaded)

    return A, S


#-----------------------------------------------------------------------------------------------------------------------

class Volume_Control_4_gmres(LinearOperator):
    """
    This class provides the dot product for a volume control system i.e.

       CC0  CC01|  1    w     sigma0
                |  1    w     sigma0
       CC10 CC1 |  1    w     sigma0
       --------------   ---   --------
       -1 -1 -1  0      p0    Q00*Dt/A


    this is left preconditioned i.e.:
    P(-1) . A . x = P(-1) . b

    P(-1) -> preconditioner
    A -> matrix of the system
    """

    def __init__(self,data):
        HmatC, EltChannel, D_i, S_i = data
        self.HmatC = HmatC
        self.HmatC._set_domain_IDX(EltChannel)
        self.HmatC._set_codomain_IDX(EltChannel)
        self.D_i = D_i
        self.S_i = S_i
        self.EltChannel = EltChannel
        self.NiC = EltChannel.size # number of elem in the channel

        # it is mandatory to define shape and dtype
        self.shape_ = (self.NiC + 1, self.NiC + 1)
        self.dtype_ = float
        super().__init__(self.dtype_, self.shape_)

    def _matvec(self, xk):

        yk = self.HmatC._matvec(xk[0:self.NiC])

        # construction of A*xk
        gp1 = yk - np.ones(self.NiC) * xk[-1]              # E*x_1+[1...1](vertical)x_2
        gp2 = np.sum(xk[0:self.NiC])                       # [1...1](horiz)x_1

        # construction of P^-1*(A*xk)
        b1=self.D_i * gp1 + self.S_i * self.D_i * gp2 * np.ones(self.NiC)  # D_e^-1*(E*x_1+[1...1](vertical)x_2)-S^-1*D_e^-1*[1...1](horiz)x_1
        b2=self.S_i * gp2                                     # S^-1*[1...1](horiz)x_1

        return np.concatenate((b1,np.asarray([b2])))
