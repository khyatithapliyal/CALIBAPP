# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-03-02 15:00:16
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-05-12 19:11:03

'''
Load the hydrophone calibration data and plot the pressure-to-voltage conversion constant (Ml)
of the hydrophone as a function of ultrasound frequency.
'''

from usnmexps.utils import check_conda_env
check_conda_env()

import matplotlib.pyplot as plt
from argparse import ArgumentParser

from instrulink import logger
from usnmexps.constants import *
from usnmexps.calib_utils import *

# Parse command line arguments
parser = ArgumentParser()
parser.add_argument(
    '-f', '--freq', type=float, help='US frequency (MHz)')
parser.add_argument(
    '--logx', default=False, action='store_true', help='Turn on log x-scale')
parser.add_argument(
    '--logy', default=False, action='store_true', help='Turn on log y-scale')
args = parser.parse_args()

# Load hydrophone calibration data table from Excel file
try:
    hydrophone_fpath = get_hydrophone_calibration_file()
    hydrophone_data = get_hydrophone_calibration_data(hydrophone_fpath)
except ValueError as err:
    logger.error(err)
    quit()

# Plot calibration data
logger.info('plotting data...')
ax = hydrophone_data.plot(x=F_KEY, y=CONV_KEY, logx=args.logx, logy=args.logy)
fig = ax.get_figure()
ax.set_ylabel(CONV_KEY)
for sk in ['top', 'right']:
    ax.spines[sk].set_visible(False)
hydrophone_fname = os.path.basename(hydrophone_fpath)
hydrophone_fname = os.path.splitext(hydrophone_fname)[0].replace('hydrophone_' , '')
ax.set_title(f'{hydrophone_fname} - calibration data')

# If frequency specified, mark the value of the conversion constant at this frequency
if args.freq is not None:
    f = args.freq  # MHz
    logger.info(f'marking {CONV_KEY} value at f = {f:.3f} MHz...')
    Ml = get_hydrophone_conversion_constant(hydrophone_data, f)
    ax.axvline(f, c='k', ls='--')
    ax.axhline(Ml, c='k', ls='--')

# Show figure
plt.show()
