# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-08-08 10:11:50
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-05-11 15:13:18

''' Acquire videos from FLIR camera and save them to disk '''

from usnmexps.utils import check_conda_env
check_conda_env()

import os
from argparse import ArgumentParser

from instrulink import logger
from instrulink.camera import *
from usnmexps.dialog import open_folder_dialog
from usnmexps.constants import *


def get_output_fpath(datadir, basename=None, ext='mp4'):
    '''
    Ask the user for a file basename and return the corresponding filepath
    
    :param datadir: directory in which to save the video file
    :param ext: file extension
    :return: full path to output video file
    '''
    # Ask for file base name if not provided
    if basename is None:
        basename = input('provide output file base name, or hit enter to exit:')
    # Return None if file basename is empty
    if len(basename) == 0:
        return None
    # Otherwise, compute & return output file path
    else:
        return os.path.join(datadir, f'{basename}.{ext}')


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument(
        '-d', '--duration', type=float, default=EXP_VIDEO_DUR, help='Nominal acquisition duration (s)')
    parser.add_argument(
        '-n', '--nacqs', type=int, default=EXP_NACQS, help='Number of consecutive acquisitions per condition') 
    args = parser.parse_args()

    # Grab camera
    cam = grab_camera()

    # Choose a directory to save videos
    videodir = open_folder_dialog()
    if videodir is None:
        logger.error('no video output directory chosen')
        quit()
    logger.info(f'videos will be saved in {videodir}')

    # Parse run order
    runorder_str = input('Provide a pre-determined run order, or hit Enter to continue: ')
    if len(runorder_str) > 0:
        runorder = [f'run{int(x):02d}' for x in runorder_str.split(',')]
        logger.info(f'acquiring videos automatically for the following runs: {runorder}')
        basenames = runorder + ['']
    else:
        logger.info('no run order provided, continuing in normal mode')
        basenames = [None] * 100

    # Get first run Ask user for video file name
    irun = 0
    output_fpath = get_output_fpath(videodir, basename=basenames[irun])
    irun += 1

    # As long as user does not exit
    while output_fpath is not None:
        # Acquire videos
        try:
            cam.acquire(
                output_fpath, args.duration, nacqs=args.nacqs, 
                trigger_source=TriggerSource.EXTERNAL)
        except KeyboardInterrupt as err:
            cam.stopCapture()
        # Ask user for video file name
        output_fpath = get_output_fpath(videodir, basename=basenames[irun])
        irun += 1

    # Disconnect camera
    cam.disconnect()

    # Exit program
    logger.info('exiting')