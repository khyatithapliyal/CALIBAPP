# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-03-07 17:31:19
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-03-21 14:26:13

'''
Get the driving parameters to set on the waveform generator in order to run 
a defined USNM protocol with a specific transducer.
'''

from usnmexps.utils import check_conda_env
check_conda_env()

import pandas as pd
from scipy.interpolate import interp1d
import os
from argparse import ArgumentParser

from instrulink import logger
from usnmexps.calib_utils import *
from usnmexps.constants import *
from usnmexps.dialog import askyesno_dialog
from usnmexps.config import PROTOCOLS_FOLDER
from usnmexps.utils import save_to_excel_file

# Parse command line arguments
parser = ArgumentParser()
args = parser.parse_args()

# Get transducer calibration Excel file and extract US frequency
try:
    calibration_fpath = get_transducer_calibration_file()
    calibration_file = os.path.basename(calibration_fpath)
    f = extract_frequency(calibration_fpath)
    calibration_data = pd.read_excel(calibration_fpath, engine='openpyxl')
except ValueError as e:
    logger.error(e)
    quit()

# Get output keys
outkeys = list(filter(
    lambda k: re.match(CALIBRATION_POUT_PATTERN, k), calibration_data.keys())) 

# If more than 1 output keys available, ask user to select
if len(outkeys) > 1:
    # Get first candidates set
    candidates = outkeys.copy()

    # Add date aggregate options for dates that contain more than 1 measure (only if more than 1 date)
    pouts_by_date = parse_outputs_by_date(outkeys)
    date_agg = {k: v for k, v in pouts_by_date.items() if len(v) > 1}
    if len(date_agg) > 1:
        candidates = candidates + list(date_agg.values())

    # Add "all" option
    candidates.append('all')
    all_code = len(candidates)

    # Generate candidate options dict
    candidates = dict(zip(np.arange(len(candidates)) + 1, candidates))

    # Ask user to select output keys
    candidates_str = '\n'.join([f' - {k}: {v}' for k, v in candidates.items()])
    answer = input(f'Please select calibration output(s) from the list:\n{candidates_str}\n')
    try:
        codes = [int(x) for x in answer.split()]
    except ValueError:
        logger.error('please provide only integer values separated by space')
        quit()
    if len(codes) == 0:
        logger.error('no column selected')
        quit()

    # If "all" not selected, select outkeys subset 
    if all_code not in codes:
        subkeys = []
        for code in codes:
            colkey = candidates[code]
            if isinstance(colkey, list):
                subkeys += colkey
            else:
                subkeys.append(colkey)
        outkeys = subkeys

# Take average of all selected output keys
p_key = 'Pout avg (MPa)'
calibration_data[p_key] = calibration_data.loc[:, outkeys].mean(axis=1)
calibration_data = calibration_data.loc[:, [VIN_KEY, p_key]]

# Load protocol data table from Excel file
logger.info('loading protocol template data...')
protocol_data = pd.read_excel(
    os.path.join(PROTOCOLS_FOLDER, PROTOCOL_TEMPLATE), engine='openpyxl')

# Add fundamental frequency column
protocol_data.insert(0, F_KEY, f)

# Interpolate input voltages for each input pressure value in the protocol (MPa)
logger.info('interpolating input voltages...')
finterp = interp1d(calibration_data[p_key], calibration_data[VIN_KEY], kind='linear')
protocol_data[VIN_KEY] = finterp(protocol_data[P_KEY])

# Determine required number of cycles for each DC value in protocol
protocol_data[NCYCLES_PER_PULSE_KEY] = np.round((f * MHZ_TO_HZ / DEFAULT_PRF) * protocol_data[DC_KEY] * 1e-2).astype(int)

# Determine number of cycles for each stimulus
protocol_data[PRF_KEY] = DEFAULT_PRF
protocol_data[DUR_KEY] = EXP_DUR
protocol_data[NCYCLES_PER_BURST_KEY] = int(np.round(DEFAULT_PRF * EXP_DUR * MS_TO_S))

logger.info(f'detailed protocol:\n{protocol_data}')

# Ask whether to save the protocol to file
save = askyesno_dialog('Save protocol to file?')

# If specified, save detailed protocol parameters to Excel file
if save:
    fname = calibration_file.replace('transducer', 'protocol')
    fpath = os.path.join(PROTOCOLS_FOLDER, fname)
    if os.path.isfile(fpath):
        logger.error(f'"{fname}" already exists in {PROTOCOLS_FOLDER} -> aborting save')
        quit()
    logger.info(f'saving detailed protocol to "{fname}"...')
    save_to_excel_file(fpath, protocol_data)
    logger.info('done')