# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Thu Dec 22 17:18:37 2016.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019.
All rights reserved. See the LICENSE.TXT file for more details.
"""

import numpy as np
import matplotlib.pyplot as plt
import dill
import copy
from tip_inversion import TipAsymInversion


def plot_as_matrix(data, mesh, fig=None):

    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(111)

    ReMesh = np.resize(data, (mesh.ny, mesh.nx))
    cax = ax.matshow(ReMesh)
    fig.colorbar(cax)
    plt.show()

    return fig

#-----------------------------------------------------------------------------------------------------------------------
def ReadFracture(filename):
    with open(filename, 'rb') as input:
        return dill.load(input)

#-----------------------------------------------------------------------------------------------------------------------

def find_regime(w, Fr, Material_properties, fluid_properties, sim_properties, timeStep, Kprime, asymptote_universal):

    sim_parameters_tmp = copy.deepcopy(sim_properties)
    sim_parameters_tmp.set_tipAsymptote('K')
    asymptote_toughness = TipAsymInversion(w,
                                           Fr,
                                           Material_properties,
                                           fluid_properties,
                                           sim_parameters_tmp,
                                           timeStep,
                                           Kprime_k=Kprime)

    sim_parameters_tmp.set_tipAsymptote('M')
    asymptote_viscosity = TipAsymInversion(w,
                                         Fr,
                                         Material_properties,
                                         fluid_properties,
                                         sim_parameters_tmp,
                                         timeStep)

    regime = 1. - abs(asymptote_viscosity - asymptote_universal) / abs(asymptote_viscosity - asymptote_toughness)
    regime[np.where(regime < 0.)[0]] = 0.
    regime[np.where(regime > 1.)[0]] = 1.

    tmp = np.full(Fr.mesh.NumberOfElts, np.nan, dtype=float)
    tmp[Fr.EltRibbon] = regime

    return tmp

#-----------------------------------------------------------------------------------------------------------------------


def save_images_to_video(image_folder, video_name='movie'):

    import cv2
    import os

    if ".avi" not in video_name:
        video_name = video_name + '.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, -1, 1, (width,height))

    img_no = 0
    for image in images:
        print("adding image no " + repr(img_no))
        video.write(cv2.imread(os.path.join(image_folder, image)))
        cv2.waitKey(1)
        img_no += 1

    cv2.destroyAllWindows()
    video.release()
