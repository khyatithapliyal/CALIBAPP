# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-02-25 15:26:13
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-04-09 20:15:19

'''
Load the amplifier calibration data and plot the input voltage to gain relationship
for all the calibrations listed.
'''

from usnmexps.utils import check_conda_env
check_conda_env()

import os
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from instrulink import logger
from usnmexps.dialog import open_file_dialog
from usnmexps.constants import *
from usnmexps.pltutils import plot_calibration_curves
from usnmexps.calib_utils import extract_amplifier_gain, vratio_to_gain, gain_to_vratio
from usnmexps.utils import get_last_file, update_last_file

# Parse command line arguments
parser = ArgumentParser()
parser.add_argument(
    '-d', '--details', default=False, action='store_true', help='Plot detailed traces')
parser.add_argument(
    '--gain', default=False, action='store_true', help='Plot amplifier gain')
parser.add_argument(
    '--logx', default=False, action='store_true', help='Turn on log x-scale')
parser.add_argument(
    '--logy', default=False, action='store_true', help='Turn on log y-scale')
parser.add_argument(
    '--last', default=False, action='store_true', help='Plot from last opened file')
parser.add_argument(
    '--yerr', default='sd', choices=['sd', 'se'], help='Error shading type')
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

# Determine output and conversion function
if args.gain:
    yout = GAIN_KEY
    convfunc = lambda Vin, Vout: vratio_to_gain(Vout / Vin)
    groundy = gain is not None and gain > 0
    yref = gain
else:
    yout = VOUT_KEY
    convfunc = None
    groundy = False
    if gain is not None:
        yref = lambda x: x * gain_to_vratio(gain)
    else:
        yref = None

# Plot calibration curves
fig = plot_calibration_curves(
    calibration_fpaths,
    details=args.details,
    logx=args.logx,
    logy=args.logy,
    ylabel=yout, 
    convfunc=convfunc,
    groundy=groundy,
    yref=yref
)

# Update last amplifier calibration file
update_last_file('amplifier', calibration_fpaths[-1])

plt.show()