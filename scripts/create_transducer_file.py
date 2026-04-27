# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-04-26 18:35:44
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-03-21 14:26:48

'''
Generate a calibration file for a new transducer at a specific ultrasound frequency (in MHz).
'''

from usnmexps.utils import check_conda_env
check_conda_env()

import re
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
Vpps_in = INPUT_VPPS_CALIB
check_input_voltages(Vpps_in)

# Ask user to provide transducer ID
transducer_id = input('Please provide a transducer ID ("T" followed by an integer):')
if len(transducer_id) == 0:
    logger.error('Empty transducer ID -> aborting')
    quit()
if not re.match(f'^{TRANSDUCER_ID_PATTERN}$', transducer_id):
    logger.error(f'Invalid transducer ID format ({transducer_id}) -> aborting')
    quit()

# Ask user to provide calibration condition
calib_conds = dict(enumerate(CALIB_CONDS))
calib_str = '\n'.join([f' - {k}: {v}' for k, v in calib_conds.items()])
answer = input(f'Please select calibration condition from the list:\n{calib_str}\n')
try:
    answer = int(answer)
    calib_cond = calib_conds[answer]
except (ValueError, KeyError):
    logger.error(f'invalid calibration condition code: "{answer}" -> aborting')
    quit()
calib_cond = calib_cond.replace(' ', '')

# Construct calibration file path
fname = f'{transducer_id}_{calib_cond}_{f:.3f}MHz.xlsx'
fpath = os.path.join(TRANSDUCERS_FOLDER, fname)

# Abort if file already exists
if os.path.isfile(fpath):
    logger.error(f'{fpath} already exists -> aborting')
    quit()

# Save file
logger.info(f'saving conversion table to {fname}')
df = pd.DataFrame({VIN_KEY: Vpps_in / MV_TO_V})
save_to_excel_file(fpath, df)
logger.info('done')