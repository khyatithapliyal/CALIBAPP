# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-02-25 15:26:13
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-02-17 14:48:31

'''
Load the amplifier calibration data and plot the input voltage to gain relationship
for all the calibrations listed.
'''

from usnmexps.utils import check_conda_env
check_conda_env()

import matplotlib.pyplot as plt
from argparse import ArgumentParser

from instrulink import logger
from usnmexps.dialog import open_file_dialog
from usnmexps.constants import *
from usnmexps.calib_utils import extract_amplifier_gain
from usnmexps.pltutils import plot_amplifier_gain_over_time
from usnmexps.utils import get_last_file, update_last_file

# Parse command line arguments
parser = ArgumentParser()
parser.add_argument(
    '--last', default=False, action='store_true', help='Plot from last opened file')
args = parser.parse_args()

# If specified, fetch last calibration file
if args.last:
    last_calib_fpath = get_last_file('amplifier')
    if last_calib_fpath is None:
        logger.error('no last amplifier calibration file found')
        quit()
    calibration_fpaths = [last_calib_fpath]

# Otherwise, get calibration files list
else:
    calibration_fpaths = open_file_dialog(
        'xlsx',
        dirname=AMPLIFIERS_FOLDER,
        title='Select amplifier calibration file', 
        multiple=True)
    if calibration_fpaths is None:
        quit()

# Parse amplifier theoretical gain from calibration file(s)
try:
    gain = extract_amplifier_gain(calibration_fpaths)
    logger.info(f'using amplifier gain = {gain} dB')
except ValueError as e:
    logger.warning(e)
    gain = None

# Plot amplifier gain over time
fig = plot_amplifier_gain_over_time(calibration_fpaths, refgain=gain)

# Update last calibration file
update_last_file('amplifier', calibration_fpaths[-1])

plt.show()