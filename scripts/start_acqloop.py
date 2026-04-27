# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-10-11 12:54:59
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-03-20 15:29:15

'''
Run continous acquisition for a specific transducer and driving frequency.
'''

from usnmexps.utils import check_conda_env
check_conda_env()

from instrulink import logger, grab_generator, grab_oscilloscope, VisaError
from usnmexps.constants import *
from usnmexps.calib_utils import *
from usnmexps.calibrators import TransducerCalibrator

# Parse command line arguments
parser = get_calibration_parser(NCYCLES_CALIB)
args = parser.parse_args()
calib_args, calib_kwargs, exec_args = parse_calibrator_settings(args)

# Get transducer calibration Excel file and extract US frequency
try:
    calibration_fpath = get_transducer_calibration_file()
    f = extract_frequency(calibration_fpath)
except ValueError as e:
    logger.error(e)
    quit()

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

# Run continuous acquisition
try:
    # Create calibrator object
    calibrator = TransducerCalibrator(
        wg, scope, 
        f * MHZ_TO_HZ, 
        *calib_args, 
        **calib_kwargs,
        Ml=Ml
    )
    
    # Run continuous acquisition
    calibrator.run_acquisition(
        *exec_args,
        acq_interval=args.acqinterval,
        adjust_scope_vscale=not args.vfix
    )

# Log error and quit, if any
except (VisaError, ValueError) as e:
    calibrator.disable_generator()
    logger.error(e)
    quit()

# Disconnect from instruments and log completion
scope.disconnect()
wg.disconnect()
logger.info('done.')