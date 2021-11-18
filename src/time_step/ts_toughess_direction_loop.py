
# External imports
import numpy as np
import logging

# Internal Imports
from level_set.anisotropy import projection_from_ribbon_LS_gradient_at_tip
from level_set.anisotropy import projection_from_ribbon
from level_set.anisotropy import get_toughness_from_cellCenter_iter
from level_set.level_set_utils import get_front_region
from properties import instrument_start, instrument_close
from solid.elasticity_Transv_Isotropic import TI_plain_strain_modulus
from tip.tip_inversion import TipAsymInversion
from level_set.FMM import fmm


def toughness_direction_loop(w_k, sgndDist_k, Fr_lstTmStp, sim_properties, mat_properties, fluid_properties, timeStep, log, perfNode):

    front_region = None
    eval_region  = None

    itr = 0
    while itr < sim_properties.maxProjItrs:
        # get the current direction of propagation
        if sim_properties.paramFromTip or mat_properties.anisotropic_K1c or mat_properties.TI_elasticity or mat_properties.inv_with_heter_K1c:
            if sim_properties.projMethod == 'ILSA_orig':
                projection_method = projection_from_ribbon
                second_arg = Fr_lstTmStp.EltChannel
            elif sim_properties.projMethod == 'LS_grad':
                projection_method = projection_from_ribbon_LS_gradient_at_tip
                second_arg = Fr_lstTmStp.front_region
            elif sim_properties.projMethod == 'LS_continousfront':
                projection_method = projection_from_ribbon_LS_gradient_at_tip
                second_arg = Fr_lstTmStp.front_region

            if itr == 0:
                # first iteration
                alpha_ribbon_k = projection_method(Fr_lstTmStp.EltRibbon,
                                                   second_arg,
                                                   Fr_lstTmStp.mesh,
                                                   sgndDist_k,
                                                   global_alpha=mat_properties.inv_with_heter_K1c)
                alpha_ribbon_km1 = np.zeros(Fr_lstTmStp.EltRibbon.size, )
            elif not mat_properties.inv_with_heter_K1c:
                alpha_ribbon_k = 0.25 * alpha_ribbon_k + 0.75 * projection_method(Fr_lstTmStp.EltRibbon,
                                                                                  second_arg,
                                                                                  Fr_lstTmStp.mesh,
                                                                                  sgndDist_k,
                                                                                  global_alpha=mat_properties.inv_with_heter_K1c)
            else:
                alpha_ribbon_k = 0.0 * alpha_ribbon_k + 1. * projection_method(Fr_lstTmStp.EltRibbon,
                                                                               second_arg,
                                                                               Fr_lstTmStp.mesh,
                                                                               sgndDist_k,
                                                                               global_alpha=mat_properties.inv_with_heter_K1c)
                # alpha_ribbon_k[np.where(alpha_ribbon_k>2.*np.pi)[0]] = alpha_ribbon_k[np.where(alpha_ribbon_k>2.*np.pi)[0]] - 2. * np.pi
            # from utility import plot_as_matrix
            # K = np.zeros((Fr_lstTmStp.mesh.NumberOfElts,), )
            # K[Fr_lstTmStp.EltRibbon] = alpha_ribbon_k
            # plot_as_matrix(K, Fr_lstTmStp.mesh)
            if np.isnan(alpha_ribbon_k).any():
                exitstatus = 11
                return exitstatus, None

        if mat_properties.inv_with_heter_K1c:
            Kprime_k = get_toughness_from_cellCenter_iter(alpha_ribbon_k,
                                                          Fr_lstTmStp.mesh.CenterCoor[Fr_lstTmStp.EltRibbon],
                                                          mat_properties)
        else:
            Kprime_k = None

        # ----- plot to check -----
        # K = np.zeros((Fr_lstTmStp.mesh.NumberOfElts,), )
        # K[Fr_lstTmStp.EltRibbon] = Kprime_k.of(sgndDist_k[Fr_lstTmStp.EltRibbon],mesh=Fr_lstTmStp.mesh, ribbon=Fr_lstTmStp.EltRibbon)
        # from utility import plot_as_matrix
        # plot_as_matrix(K, Fr_lstTmStp.mesh)

        if mat_properties.TI_elasticity:  # and not mat_properties.inv_with_heter_K1c:
            Eprime_k = TI_plain_strain_modulus(alpha_ribbon_k,
                                               mat_properties.Cij)
            if np.isnan(Eprime_k).any():
                exitstatus = 11
                return exitstatus, None
        else:
            Eprime_k = None

        # Initialization of the signed distance in the ribbon element - by inverting the tip asymptotics
        # large float value. (algorithm requires inf)
        # todo: check the sgndDist initiation process
        sgndDist_k = 1e50 * np.ones((Fr_lstTmStp.mesh.NumberOfElts,), float)  # Initializing the cells with extremely

        perfNode_tipInv = instrument_start('tip inversion', perfNode)

        sgndDist_k[Fr_lstTmStp.EltRibbon] = - TipAsymInversion(w_k,
                                                               Fr_lstTmStp,
                                                               mat_properties,
                                                               fluid_properties,
                                                               sim_properties,
                                                               timeStep,
                                                               Kprime_k=Kprime_k,
                                                               Eprime_k=Eprime_k,
                                                               perfNode=perfNode_tipInv)

        status = True
        fail_cause = None
        # if tip inversion returns nan
        if np.isnan(sgndDist_k[Fr_lstTmStp.EltRibbon]).any():
            status = False
            fail_cause = 'tip inversion failed'
            exitstatus = 7
            # K = np.zeros((Fr_lstTmStp.mesh.NumberOfElts,), )
            # K[Fr_lstTmStp.EltRibbon] = sgndDist_k[Fr_lstTmStp.EltRibbon]
            # from utility import plot_as_matrix
            # plot_as_matrix(K, Fr_lstTmStp.mesh)
            # np.argwhere(np.isnan(sgndDist_k[Fr_lstTmStp.EltRibbon])).flatten()

        if perfNode_tipInv is not None:
            instrument_close(perfNode, perfNode_tipInv, None, len(Fr_lstTmStp.EltRibbon),
                             status, fail_cause, Fr_lstTmStp.time)
            perfNode.tipInv_data.append(perfNode_tipInv)

        if not status:
            return exitstatus, None

        # # Check for positive
        # if np.any(sgndDist_k[Fr_lstTmStp.EltRibbon]>0):
        #     log.debug("found a positive signed distance: it must not happen ")
        # Check if the front is receding
        # Define the level set in the ribbon elements as not to allow the fracture to reced.
        sgndDist_k[Fr_lstTmStp.EltRibbon] = np.minimum(sgndDist_k[Fr_lstTmStp.EltRibbon],
                                                       Fr_lstTmStp.sgndDist[Fr_lstTmStp.EltRibbon])

        # We calculate the front region
        front_region = get_front_region(Fr_lstTmStp.mesh, Fr_lstTmStp.EltRibbon, sgndDist_k[Fr_lstTmStp.EltRibbon])

        # the search region outwards from the front position at last time step
        pstv_region = np.setdiff1d(front_region, Fr_lstTmStp.EltChannel)

        # the search region inwards from the front position at last time step
        ngtv_region = np.setdiff1d(front_region, pstv_region)

        # Creating a fmm structure to solve the level set
        fmmStruct = fmm(Fr_lstTmStp.mesh)

        # We define the ribbon elements as the known elements and solve from there outwards to the domain boundary.
        fmmStruct.solveFMM((sgndDist_k[Fr_lstTmStp.EltRibbon], Fr_lstTmStp.EltRibbon),
                           np.unique(np.hstack((pstv_region, Fr_lstTmStp.EltRibbon))), Fr_lstTmStp.mesh)

        # We define the ribbon elements as the known elements and solve from there inwards (inside the fracture). To do
        # so, we need a sign change on the level set (positive inside)
        toEval = np.unique(np.hstack((ngtv_region, Fr_lstTmStp.EltRibbon)))
        fmmStruct.solveFMM((-sgndDist_k[Fr_lstTmStp.EltRibbon], Fr_lstTmStp.EltRibbon), toEval,
                           Fr_lstTmStp.mesh)

        # The solution stored in the object is the calculated level set. we need however to change the sign as to have
        # negative inside and positive outside.
        sgndDist_k = fmmStruct.LS
        sgndDist_k[toEval] = -sgndDist_k[toEval]

        eval_region = np.where(sgndDist_k[front_region] >= -Fr_lstTmStp.mesh.cellDiag)[0]

        # do it only once if not anisotropic
        if not (sim_properties.paramFromTip or mat_properties.anisotropic_K1c
                or mat_properties.TI_elasticity or mat_properties.inv_with_heter_K1c) or sim_properties.explicitProjection:
            break

        norm = np.linalg.norm(np.abs(np.sin(alpha_ribbon_k) - np.sin(alpha_ribbon_km1)) + np.abs(
            np.cos(alpha_ribbon_k) - np.cos(alpha_ribbon_km1)))
        if norm < sim_properties.toleranceProjection:
            log.debug("projection iteration converged after " + repr(itr - 1) + " iterations; exiting norm " +
                      repr(norm))
            break

        alpha_ribbon_km1 = np.copy(alpha_ribbon_k)
        log.debug("iterating on projection... norm " + repr(norm))
        itr += 1

    return front_region, eval_region, sgndDist_k