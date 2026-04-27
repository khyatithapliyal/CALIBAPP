# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-02-25 15:26:13
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-03-21 18:17:38

'''
Load the transducer calibration data and plot the input voltage to output ultrasound pressure relationship
for all the calibrations listed.
'''

from usnmexps.utils import check_conda_env
check_conda_env()

import matplotlib.pyplot as plt
from argparse import ArgumentParser

from instrulink import logger
from usnmexps.dialog import open_file_dialog
from usnmexps.constants import *
from usnmexps.pltutils import plot_calibration_curves
from usnmexps.utils import get_last_file, update_last_file

# Parse command line arguments
parser = ArgumentParser()
parser.add_argument(
    '-m', '--mode', default='summary', choices=['summary', 'details', 'mean'],
    help='Display mode: summary (one line per date, auto-filters similar dates), '
         'details (all individual traces), mean (single overall mean)')
parser.add_argument(
    '-d', '--details', default=False, action='store_true', 
    help='(Legacy) Plot detailed traces. Equivalent to --mode details')
parser.add_argument(
    '--logx', default=False, action='store_true', help='Turn on log x-scale')
parser.add_argument(
    '--logy', default=False, action='store_true', help='Turn on log y-scale')
parser.add_argument(
    '--last', default=False, action='store_true', help='Plot from last opened file')
parser.add_argument(
    '--yerr', default='sd', choices=['sd', 'se'], help='Error shading type')
parser.add_argument(
    '--yout', default='P', choices=['P', 'cpl', 'all'], help='Output unit(s)')
args = parser.parse_args()

# Resolve display mode (legacy --details flag overrides --mode)
if args.details:
    args.mode = 'details'

# Parse output unit(s)
yout= {
    'P': P_KEY, 
    'cpl': CPLRATIODB_KEY,
    'all': [P_KEY, CPLRATIODB_KEY]
}[args.yout]

# If specified, fetch last calibration file
if args.last:
    last_calib_fpath = get_last_file('transducer')
    if last_calib_fpath is None:
        logger.error('no last transducer calibration file found')
        quit()
    calibration_fpaths = [last_calib_fpath]

# Otherwise, get calibration files list
else:
    calibration_fpaths = open_file_dialog(
        'xlsx',
        dirname=TRANSDUCERS_FOLDER,
        title='Select transducer calibration file', 
        multiple=True)
    if calibration_fpaths is None:
        quit()

# Plot calibration curves
fig = plot_calibration_curves(
    calibration_fpaths,
    mode=args.mode,
    logx=args.logx,
    logy=args.logy,
    yerr=args.yerr,
    ylabel=yout,
)

# Update last calibration file
update_last_file('transducer', calibration_fpaths[-1])

plt.show()