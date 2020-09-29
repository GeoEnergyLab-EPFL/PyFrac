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

    return regime

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

#-----------------------------------------------------------------------------------------------------------------------
import sys
import logging
def logging_level(logging_level_string):
    """
    This function returns the pertinent logging level based on the string received as input.

    :param logging_level_string: string that defines the level of logging:
                                 'debug' - Detailed information, typically of interest only when diagnosing problems.
                                 'info' - Confirmation that things are working as expected.
                                 'warning' - An indication that something unexpected happened, or indicative of some problem in the near future (e.g. ‘disk space low’). The software is still working as expected.
                                 'error' - Due to a more serious problem, the software has not been able to perform some function.
                                 'critical' - A serious error, indicating that the program itself may be unable to continue running.
    :return: code representing the logging error
    """
    if logging_level_string in ['debug', 'Debug', 'DEBUG']:
        return logging.DEBUG
    elif logging_level_string in ['info', 'Info', 'INFO']:
        return logging.INFO
    elif logging_level_string in ['warning', 'Warning', 'WARNING']:
        return logging.WARNING
    elif logging_level_string in ['error', 'Error', 'ERROR']:
        return logging.ERROR
    elif logging_level_string in ['critical', 'Critical', 'CRITICAL']:
        return logging.CRITICAL
    else:
        SystemExit('Options are: debug, info, warning, error, critical')

def setup_logging_to(logging_to_file_level='debug',logging_to_console_level='debug'):
    """This function sets up the log, both to the file and to the console
        Note: from any module in the code you can use the logging capabilities. You just have to:

        1) import the module

        import logging

        2) create a child of the logger named 'PyFrac' defined in this function. Use a pertinent name as 'Pyfrac.frontrec'

        logger1 = logging.getLogger('PyFrac.frontrec')

        3) use the object to send messages in the module, such as

        logger1.debug('debug message')
        logger1.info('info message')
        logger1.warning('warn message')
        logger1.error('error message')
        logger1.critical('critical message')

    :param logging_to_file_level: string that defines the level of logging concerning the file:
                                 'debug'    - Detailed information, typically of interest only when diagnosing problems.
                                 'info'     - Confirmation that things are working as expected.
                                 'warning'  - An indication that something unexpected happened, or indicative of some
                                              problem in the near future (e.g. ‘disk space low’). The software is still
                                              working as expected.
                                 'error'    - Due to a more serious problem, the software has not been able to perform
                                              some function.
                                 'critical' - A serious error, indicating that the program itself may be unable to
                                              continue running.

    :param logging_to_console_level: string that defines the level of logging concerning the console. See above.
    :return: -
    """
    fileLvl = logging_level(logging_to_file_level)
    consoleLvl = logging_level(logging_to_console_level)

    logger = logging.getLogger('PyFrac')
    logger.setLevel(logging.DEBUG)

    # create file handler which logs debug messages to the file
    fh = logging.FileHandler('pyfrac.log',mode='w')
    fh.setLevel(fileLvl)
    # create console handler with a higher log level
    ch = logging.StreamHandler(stream = sys.stdout)
    ch.setLevel(consoleLvl)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',datefmt='%m-%d %H:%M')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)