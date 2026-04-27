# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2023-04-05 11:39:52
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-04-05 11:41:30

''' Utilities for communication with other computational engines '''

from instrulink.logger import logger

def connect_to_matlab():
    logger.info('connecting to MATLAB...')
    import matlab.engine
    shared_sessions = matlab.engine.find_matlab()
    if len(shared_sessions) == 0:
        raise ValueError('No shared MATLAB session. Type in "matlab.engine.shareEngine" in the MATLAB prompt.')
    matlab_eng = matlab.engine.connect_matlab()
    if not matlab_eng.is_SI_available():
        raise ValueError('ScanImage not available in MATLAB workspace')
    return matlab_eng
