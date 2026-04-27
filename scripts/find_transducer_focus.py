# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-05-02 14:41:37
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-03-20 11:24:31

'''
Find acoustic focus automatically for a specific transducer and driving frequency.
'''

from usnmexps.utils import check_conda_env
check_conda_env()

import os
import matplotlib.pyplot as plt

from instrulink import logger, grab_generator, grab_oscilloscope, grab_manipulator, VisaError, SutterError
from usnmexps.constants import *
from usnmexps.calibrators import TransducerCalibrator
from usnmexps.calib_utils import *

# Parse command line arguments
parser = get_calibration_parser(NCYCLES_SCAN)
add_scan_args(
    parser, DELTA_FOCUS, default_order=FOCUS_CROSS_SCAN_ORDER)
args = parser.parse_args()
calib_args, calib_kwargs, exec_args = parse_calibrator_settings(args)
scan_args, scan_kwargs = parse_scan_args(args)

if ''.join(sorted(args.order)) != 'XYZ':
    raise ValueError(f'invalid scan order: {args.order} (constituents must be {{X, Y, Z}})')

# Get transducer calibration Excel file and extract US frequency
try:
    calibration_fpath = get_transducer_calibration_file()
    calibration_file = os.path.basename(calibration_fpath)
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
    
# Attempt to grab waveform generator, oscilloscope and micro-manipulator, and quit if failed
try:
    wg = grab_generator()
    scope = grab_oscilloscope(key=args.scope)
    mp = grab_manipulator()
except (VisaError, SutterError) as e:
    logger.error(e)

# Find acoustic focus
try:
    # Create calibrator object
    calibrator = TransducerCalibrator(
        wg, scope, 
        f * MHZ_TO_HZ,
        *calib_args,
        **calib_kwargs,
        Ml=Ml
    )
    
    # Find acoustic focus
    focus_xyz, focus_MPa = calibrator.find_acoustic_focus(
        mp, 
        *exec_args,
        *scan_args,
        VBASIS, 
        adjust_scope_vscale=not args.vfix,
        **scan_kwargs
    )

# Log error and quit if any
except (VisaError, ValueError, SutterError, KeyboardInterrupt) as e:
    calibrator.disable_generator()
    logger.error(e)
    quit()

# Disconnect from instruments and log completion
wg.disconnect()
scope.disconnect()
mp.disconnect()
logger.info('done.')

# Plot figures
plt.show()