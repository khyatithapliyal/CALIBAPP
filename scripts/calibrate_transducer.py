# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-04-26 18:39:57
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-03-21 14:27:50

'''
Generate a transducer calibration curve automatically for a specific transducer and driving frequency.
'''

from usnmexps.utils import check_conda_env
check_conda_env()

import matplotlib.pyplot as plt
import pandas as pd
import os

from instrulink import logger, grab_generator, grab_oscilloscope, VisaError
from usnmexps.constants import *
from usnmexps.dialog import askyesno_dialog
from usnmexps.calib_utils import *
from usnmexps.calibrators import TransducerCalibrator
from usnmexps.utils import save_to_excel_file

# Parse command line arguments
parser = get_calibration_parser(NCYCLES_CALIB, add_vpp=False)
parser.add_argument(
    '--ncurves', type=int, default=1, help='Number of calibration curves to acquire')
args = parser.parse_args()
calib_args, calib_kwargs, exec_args = parse_calibrator_settings(args)
if args.ncurves < 1:
    raise ValueError('number of curves must be >= 1')
if args.ncurves > 1:
    savemode = 'auto'
else:
    savemode = 'ask'

# Get transducer calibration Excel file and extract US frequency
try:
    calibration_fpath = get_transducer_calibration_file()
    calibration_file = os.path.basename(calibration_fpath)
    f = extract_frequency(calibration_fpath)
    calibration_data = pd.read_excel(calibration_fpath, engine='openpyxl')
except ValueError as e:
    logger.error(e)
    quit()

# Load input Vpps from transducer calibration file
Vpps_in = calibration_data[VIN_KEY] * MV_TO_V  # Vpp
logger.info(f'loaded {Vpps_in.size} driving voltages ({Vpps_in.min()} - {Vpps_in.max()} Vpp)')

# Load hydrophone calibration data and interpolate conversion factor at transducer frequency
try:
    hydrophone_fpath = get_hydrophone_calibration_file()
    hydrophone_data = get_hydrophone_calibration_data(hydrophone_fpath)
    Ml = get_hydrophone_conversion_constant(hydrophone_data, f) # V/Pa
except ValueError as e:
    logger.error(e)
    quit()

# Attempt to grab waveform generator and oscilloscope, and quit if failed
try:
    wg = grab_generator()
    scope = grab_oscilloscope(key=args.scope)
except VisaError as e:
    logger.error(e)
    quit()

# As long as user specifies
run_calib = True
ncurves_acq = 0
while run_calib:
    try:
        # Create calibrator object
        calibrator = TransducerCalibrator(
            wg, scope, 
            f * MHZ_TO_HZ, 
            *calib_args,
            **calib_kwargs,
            Ml=Ml,
        )
        
        # Run calibration curve
        out = calibrator.run_io_sweep(
            Vpps_in, 
            *exec_args,
            acq_interval=args.acqinterval,
            npercond=args.npercond
        )
        if isinstance(out, tuple):
            MPas, cplratios = out
        else:
            MPas, cplratios = out, None
        
        # Increment calibration counter
        ncurves_acq += 1

    # Log error and quit, if any
    except (VisaError, ValueError, KeyboardInterrupt) as e:
        calibrator.disable_generator()
        logger.error(e)
        quit()
    
    if savemode == 'ask':
        # Ask whether to save the calibration curve
        save = askyesno_dialog('Save calibration curve to file?')
    else:
        save = True

    if save:
        # Add calibration curve as new column in dataframe
        Pkey = find_out_column_name(calibration_data.columns)
        calibration_data[Pkey] = MPas

        # If cplratios are available, add them as additional column
        if cplratios is not None:
            colprefix = ' '.join(Pkey.split(' ')[:-1])
            cplratio_key = f'{colprefix} cplratio'
            calibration_data[cplratio_key] = cplratios

        # Check file availability
        check_file_availability(calibration_fpath)
        
        # Save to file
        logger.info(f'saving pressure calibration curve as column "{Pkey}" in {calibration_file}')
        if cplratios is not None:
            logger.info(f'saving coupling ratio calibration curve as column "{cplratio_key}" in {calibration_file}')
        save_to_excel_file(calibration_fpath, calibration_data)
    
    if ncurves_acq >= args.ncurves:
        # Ask whether to run another calibration
        run_calib = askyesno_dialog('Run another calibration?')
    else:
        run_calib = True
    
# If specified, keep calibration figures open
if args.keepfigs:
    plt.show()

# Log completion
wg.disconnect()
logger.info('done.')