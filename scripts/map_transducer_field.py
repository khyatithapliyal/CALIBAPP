# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-09-08 16:31:34
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-03-20 11:23:27

'''
Map acoustic field automatically for a specific transducer and driving frequency.
'''

from usnmexps.utils import check_conda_env
check_conda_env()

import os
import matplotlib.pyplot as plt

from instrulink import logger, grab_generator, grab_oscilloscope, grab_manipulator, VisaError, SutterError
from usnmexps.config import MAPPING_DATA_FOLDER
from usnmexps.constants import *
from usnmexps.dialog import open_folder_dialog, askyesno_dialog
from usnmexps.calibrators import TransducerCalibrator
from usnmexps.calib_utils import *
from usnmexps.pltutils import plot_acoustic_field

# Parse command line arguments
parser = get_calibration_parser(NCYCLES_SCAN)
add_scan_args(parser, DELTA_SCAN, res=RES_SCAN, default_order=MAP_SCAN_ORDER, mmode='brute')
parser.add_argument(
    '-s', '--save', default='amp', choices=('none', 'amp', 'full'),
    help='Save mode ("none", "amp" or "full")')
args = parser.parse_args()
calib_args, calib_kwargs, exec_args = parse_calibrator_settings(args)
scan_args, scan_kwargs = parse_scan_args(args)

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
    hydrophone_model = extract_hydrophone_model(hydrophone_fpath)
    hydrophone_data = get_hydrophone_calibration_data(hydrophone_fpath)
    Ml = get_hydrophone_conversion_constant(hydrophone_data, f) # V/Pa
except ValueError as e:
    logger.error(e)
    quit()

# Get mapping code
calibration_fcode = os.path.splitext(calibration_file)[0]
mapping_code = TransducerCalibrator.get_mapping_code(
    calibration_fcode, hydrophone_model, *scan_args[1:], mode=scan_kwargs['mmode'])
logger.info(f'mapping code: "{mapping_code}"')

# Set empty output paths
out_fpath = None
traces_dir = None

# If user opted not to save anything
if args.save == 'none':
    # Ask confirmation, and abort if needed
    cont = askyesno_dialog('You opted not to save any mapping data. Continue anyway?')
    if not cont:
        quit()

# If user opted to save
else:
    # Ask for output directory, and abort if none given
    outdir = open_folder_dialog(
        title='Select output directory', initialdir=MAPPING_DATA_FOLDER)
    if outdir is None:
        logger.error('no output directory chosen')
        quit()
    # Determine path to output mapping file
    out_fpath = os.path.join(outdir, f'{mapping_code}.csv')
    if os.path.exists(out_fpath):
        logger.error(f'"{out_fpath}" file already exists')
        quit()
    # If user opted for "full" save, determine path to directory that will contain traces
    if args.save == 'full':
        traces_dir = os.path.join(outdir, f'{mapping_code}_traces')
        if os.path.exists(traces_dir):
            logger.error(f'"{traces_dir}" directory already exists')
            quit()
        else:
            os.makedirs(traces_dir)

try:
    # Grab waveform generator, oscilloscope and micro-manipulator
    wg = grab_generator()
    scope = grab_oscilloscope(key=args.scope)
    mp = grab_manipulator()
except (VisaError, SutterError) as e:
    logger.error(e)
    quit()

try:
    # Create calibrator object
    calibrator = TransducerCalibrator(
        wg, scope, 
        f * MHZ_TO_HZ,
        *calib_args,
        **calib_kwargs,
        Ml=Ml
    )
    
    # Map acoustic field
    coords_per_dim, Pmat = calibrator.map_acoustic_field(
        mp,
        *exec_args,
        *scan_args,
        VBASIS,
        out_fpath=out_fpath, 
        traces_dir=traces_dir,
        **scan_kwargs
    )

# Log error and quit, if any
except (VisaError, ValueError, SutterError, KeyboardInterrupt) as e:
    calibrator.disable_generator()
    logger.error(e)
    quit()

# Disconnect from instruments and log completion
scope.disconnect()
wg.disconnect()
mp.disconnect()
logger.info('done.')

# Plot results
sliceaxis = 'XYZ'[np.argmin(args.nperax)]
fig = plot_acoustic_field(
    coords_per_dim, Pmat, sliceaxis, title=mapping_code, xyz_unit='mm')

plt.show()
