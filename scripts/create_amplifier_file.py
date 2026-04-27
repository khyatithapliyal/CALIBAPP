# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-09-02 12:32:05
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-03-21 14:27:10

'''
Generate a calibration file for a new amplifier at a specific ultrasound frequency (in MHz).
'''

from usnmexps.utils import check_conda_env
check_conda_env()

import pandas as pd
from argparse import ArgumentParser
import os

from instrulink import logger
from usnmexps.constants import *
from usnmexps.calib_utils import check_input_voltages
from usnmexps.utils import save_to_excel_file

# Parse command line arguments
parser = ArgumentParser()
parser.add_argument(
    '-f', '--freq', type=float, help='US frequency (MHz)')
args = parser.parse_args()
f = args.freq
if f is None:
    logger.error('Error: exactly 1 target frequency must be provided')
    quit()

# Input Vpp range
Vpps_in = np.linspace(INPUT_VPPS_CALIB[0], INPUT_VPPS_CALIB[-1], 11)
check_input_voltages(Vpps_in)

# Ask user to provide amplifier ID
amp_id = input('Please provide an amplifier ID:')
if len(amp_id) == 0:
    logger.error('Empty amplifier ID -> aborting')
    quit()

# Ask user to provide attenuator gain
att_gain = input('Please provide attenuator gain (in dB): ')
if len(att_gain) == 0:
    logger.error('no attenuator gain provided')
    quit()
try:
    att_gain = -abs(float(att_gain))  # Make sure gain (dB) is negative
except ValueError:
    logger.error('invalid attenuator gain -> aborting')
    quit()

# Construct calibration file path
fname = f'amplifier_{amp_id}_{att_gain:.2f}dB_{f:.2f}MHz.xlsx'
fpath = os.path.join(AMPLIFIERS_FOLDER, fname)

# Abort if file already exists
if os.path.isfile(fpath):
    logger.error(f'{fpath} already exists -> aborting save')

# Save file
logger.info(f'saving conversion table to {fname}')
df = pd.DataFrame({VIN_KEY: Vpps_in / MV_TO_V})
save_to_excel_file(fpath, df)
logger.info('done')