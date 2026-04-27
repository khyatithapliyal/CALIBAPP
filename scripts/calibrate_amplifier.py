# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-04-26 18:39:57
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-03-21 17:09:03

'''
Generate an amplifier calibration curve automatically for a specific driving frequency.
'''

from usnmexps.utils import check_conda_env
check_conda_env()

import matplotlib.pyplot as plt
from instrulink import logger, grab_generator, grab_oscilloscope, VisaError
from usnmexps.constants import *
from usnmexps.calib_utils import *
from usnmexps.dialog import askyesno_dialog
from usnmexps.calibrators import AmplifierCalibrator
from usnmexps.utils import save_to_excel_file

# Parse command line arguments
parser = get_calibration_parser(NCYCLES_CALIB, add_vpp=False)
args = parser.parse_args()
calib_args, calib_kwargs, exec_args = parse_calibrator_settings(args)

# Get amplifier calibration Excel file and extract frequency and attenuation gain
try:
    calibration_fpath = get_amplifier_calibration_file()
    calibration_file = os.path.basename(calibration_fpath)
    try:
        theoretical_gain = extract_amplifier_gain(calibration_fpath)
        logger.info(f'amplifier theoretical gain = {theoretical_gain} dB')
    except ValueError as e:
        logger.warning(e)
        theoretical_gain = None
    f = extract_frequency(calibration_fpath)
    att_gain = extract_gain(calibration_fpath)
    calibration_data = pd.read_excel(calibration_fpath, engine='openpyxl')
except ValueError as e:
    logger.error(e)
    quit()
vratio = gain_to_vratio(-abs(att_gain))
logger.info(f'output conversion constant = {vratio}')

# Load input Vpps from transducer calibration file
Vpps_in = calibration_data[VIN_KEY] * MV_TO_V  # Vpp
logger.info(f'loaded {Vpps_in.size} driving voltages ({Vpps_in.min()} - {Vpps_in.max()} Vpp)')

# Attempt to grab waveform generator and oscilloscope, and quit if failed
try:
    wg = grab_generator()
    scope = grab_oscilloscope(key=args.scope)
except VisaError as e:
    logger.error(e)
    quit()

# As long as user specifies
run_calib = True
while run_calib:
    try:
        # Create calibrator object
        calibrator = AmplifierCalibrator(
            wg, scope, 
            f * MHZ_TO_HZ,
            *calib_args,
            **calib_kwargs,
            output_conv_constant=vratio
        )

        # Run calibration curve
        Vpps_out = calibrator.run_io_sweep(
            Vpps_in, 
            *exec_args, 
            acq_interval=args.acqinterval,
            npercond=args.npercond, 
            adjust_scope_vscale=not args.vfix
        )

    # Log error and quit, if any
    except (VisaError, ValueError) as e:
        calibrator.disable_generator()
        logger.error(e)
        quit()

    # Compute voltage ratios and corresponding gains
    ratios = Vpps_out[1:] / Vpps_in[1:]
    gains = vratio_to_gain(ratios)

    # Log mean and std
    mu_ratio, sigma_ratio = np.nanmean(ratios), np.nanstd(ratios)
    mu_gain, sigma_gain = np.nanmean(gains), np.nanstd(gains)
    logger.info(f'voltage ratio = {mu_ratio:.3f} +/- {sigma_ratio:.3f}')
    logger.info(f'corresponding gains = {mu_gain:.2f} +/- {sigma_gain:.2f} dB')

    # Compute deviation from theoretical_gain (if any)
    if theoretical_gain is not None:
        gain_error = mu_gain - theoretical_gain
        # Log error warning if too large
        if abs(gain_error) > 2:
            logger.error(
                f'average measured gain deviates ({mu_gain:.2f} dB) significantly from theoretical gain ({theoretical_gain:.2f} dB)')

    # Ask whether to save the calibration curve
    save = askyesno_dialog('Save calibration curve to file?')

    if save:
        # Add calibration curve as new column in dataframe
        key = find_out_column_name(calibration_data.columns, key='Vout', unit='Vpp')
        calibration_data[key] = Vpps_out

        # Check file availability
        check_file_availability(calibration_fpath)

        # Save to file
        logger.info(f'saving calibration curve as column "{key}" in {calibration_file}')
        save_to_excel_file(calibration_fpath, calibration_data)
    
    # Ask whether to run another calibration
    run_calib = askyesno_dialog('Run another calibration?')

# If specified, keep calibration figures open
if args.keepfigs:
    plt.show()

logger.info('done.')