# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-09-06 16:20:55
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-06-10 17:01:06

'''
Set the driving parameters of the waveform generator to generate a
gated sine burst at a given frequency.
'''

from usnmexps.utils import check_conda_env
check_conda_env()

import time
from argparse import ArgumentParser

from instrulink import logger, VisaError, grab_generator
from usnmexps.constants import *

# Parse command line arguments
parser = ArgumentParser()
parser.add_argument(
    '-f', '--freq', type=float, default=2., help='Carrier frequency (MHz)')
parser.add_argument(
    '--PRF', type=float, default=DEFAULT_PRF, help='Pulse repetition frequency (Hz)')
parser.add_argument(
    '--BRF', type=float, default=1., help='Burst repetition frequency (Hz)')
parser.add_argument(
    '--DC', type=int, default=50, help='Burst duty cycle (%)')
parser.add_argument(
    '-d', '--duration', type=float, default=EXP_DUR, help='Burst duration (ms)')
parser.add_argument(
    '--vpp', type=float, default=REF_VPP_CALIB, help='Signal amplitude (in Vpp)')
parser.add_argument(
    '--tramp', type=float, default=0., help='Ramp time (ms)')
parser.add_argument(
    '--gtype', default='trig', choices=('mod', 'trig'), 
    help='Gating type, i.e. "mod" for modulation or "trig" for trigger (only if modulation period is specified)')
parser.add_argument(
    '--release', default=False, action='store_true', help='Release generator')
args = parser.parse_args()
params_str = '\n'.join([f'   - {x}' for x in [
    f'f = {args.freq:.2f} MHz',
    f'A = {args.vpp:.3f} Vpp',
    f'PRF = {args.PRF:.2f} Hz',
    f'DC = {args.DC:.1f} %',
    f'duration = {args.duration:.1f} ms',
    f'BRF = {args.BRF:.2f} Hz',
    f'ramp-time = {args.tramp:.1f} ms'
]])

try:
    # Grab function generator
    wg = grab_generator()
    
    # Set gated sine burst
    logger.info(f'setting gated sine burst parameters:\n{params_str}')
    wg.set_gated_sine_burst(
        args.freq * MHZ_TO_HZ,  # Hz
        args.vpp,  # Vpp
        args.duration * MS_TO_S,  # s
        args.PRF,  # Hz
        args.DC,  # %
        tramp=args.tramp * MS_TO_S,  # s
        gate_type=args.gtype
    )

    # Set sine burst on internal loop
    logger.info(f'starting sine burst...')
    wg.start_trigger_loop(1, T=1. / args.BRF)
    time.sleep(1.)
    
    # If "no-release" mode specified, wait for any user input to stop burst
    if not args.release:
        logger.info('press enter to stop sine burst...')
        input()
        logger.info('disabling generator outputs')
        wg.disable_output()

    # Release function generator
    wg.disconnect()
    
except VisaError as e:
    logger.error(e)
    quit()
