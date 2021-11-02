# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Thu Dec 22 17:18:37 2016.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# External imports
import numpy as np
import matplotlib.pyplot as plt
import dill
import requests

class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
        self.threshold = 100
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            if self.niter == self.threshold:
                print('WARNING: GMRES has not converged in '+str(self.niter)+' iter, monitoring the residual')
            if self.niter > self.threshold:
                print('iter %3i\trk = %s' % (self.niter, str(rk)))

#-----------------------------------------------------------------------------------------------------------------------

def getMemUse():
    # some memory statistics
    import os, psutil

    process = psutil.Process(os.getpid())
    byte_use = process.memory_info().rss  # byte
    GiByte_use = byte_use / 1024 / 1024 / 1024  # GiB
    print("  -> Current memory use: " + str(GiByte_use) + " GiB")
    return GiByte_use

#-----------------------------------------------------------------------------------------------------------------------

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

def save_images_to_video(image_folder, video_name='movie'):

    import cv2
    import os
    log = logging.getLogger('PyFrac.save_images_to_video')
    if ".avi" not in video_name:
        video_name = video_name + '.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, -1, 1, (width,height))

    img_no = 0
    for image in images:
        log.info("adding image no " + repr(img_no))
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

#-----------------------------------------------------------------------------------------------------------------------

def setup_logging_to_console(verbosity_level='debug'):
    """This function sets up the log to the console
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

        4) IMPORTANT TO KNOW:
           1-If you want to log only to the console in the abobe example you have to use: logger1 = logging.getLogger('PyFrac_LC.frontrec')
           2-SystemExit and KeyboardInterrupt exceptions are never swallowed by the logging package .

    :param verbosity_level: string that defines the level of logging concerning the console:
                                 'debug'    - Detailed information, typically of interest only when diagnosing problems.
                                 'info'     - Confirmation that things are working as expected.
                                 'warning'  - An indication that something unexpected happened, or indicative of some
                                              problem in the near future (e.g. ‘disk space low’). The software is still
                                              working as expected.
                                 'error'    - Due to a more serious problem, the software has not been able to perform
                                              some function.
                                 'critical' - A serious error, indicating that the program itself may be unable to
                                              continue running.
    :return: -
    """
    consoleLvl = logging_level(verbosity_level)

    logger = logging.getLogger('PyFrac')
    logger.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler(stream = sys.stdout)
    ch.setLevel(consoleLvl)

    # create formatter and add it to the handlers
    formatterch = logging.Formatter(fmt='%(levelname)-8s:     %(message)s')
    ch.setFormatter(formatterch)

    # add the handlers to logger
    logger.addHandler(ch)

    log = logging.getLogger('PyFrac.general')
    log.info('Console logger set up correctly')

    # create a logger that logs only to the console and not on the file:
    logger_to_console = logging.getLogger('PyFrac_LC')
    logger_to_console.setLevel(logging.DEBUG)

    # add the handlers to logger
    logger_to_console.addHandler(ch)

    # usage example
    # logger_to_files = logging.getLogger('PyFrac_LF.set_logging_to_file')
    # logger_to_files.info('this comment will go only to the log file')

#-----------------------------------------------------------------------------------------------------------------------

def append_new_line(file_name, text_to_append):
    """Append given text as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)

#-----------------------------------------------------------------------------------------------------------------------

# SENDING A MESSAGE TO MONITOR THE SIMULATION
"""
see:
https://medium.com/@ManHay_Hong/how-to-create-a-telegram-bot-and-send-messages-with-python-4cf314d9fa3e

"""

def send_phone_message(bot_message):
    try:
        bot_token = 'the one that you have when you create a bot'
        bot_chatID = 'it is the field ID at https://api.telegram.org/bot<yourtoken>/getUpdates -> replace <yourtoken> with bot_token'



        send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message
        response = requests.get(send_text)
        return response.json()

    except Exception as e:
        # there may be many error coming...
        print(e);

#-----------------------------------------------------------------------------------------------------------------------
