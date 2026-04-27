# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-09-06 16:20:55
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-03-14 11:54:31

'''
Set the driving parameters of the waveform generator to generate a
looping sine burst at a given frequency.
'''

from usnmexps.utils import check_conda_env
check_conda_env()

import time
from argparse import ArgumentParser

from instrulink import logger, grab_generator, VisaError
from usnmexps.constants import *

# Parse command line arguments
parser = ArgumentParser()
parser.add_argument(
    '-f', '--freq', type=float, default=2., help='Carrier frequency (MHz)')
parser.add_argument(
    '--BRF', type=float, default=DEFAULT_PRF, help='Burst repetition frequency (Hz)')
parser.add_argument(
    '-n', '--ncycles', type=int, default=NCYCLES_CALIB, help='Burst # cycles')
parser.add_argument(
    '--vpp', type=float, default=REF_VPP_CALIB, help='Signal amplitude (in Vpp)')
parser.add_argument(
    '--ich', type=int, default=2, help='Channel index')
parser.add_argument(
    '--ich_trig', type=int, default=-1, help='Triggering channel index')
parser.add_argument(
    '--tramp', type=float, default=0., help='Ramp time (ms)')
parser.add_argument(
    '--gtype', default='trig', choices=('mod', 'trig'), 
    help='Gating type, i.e. "mod" for modulation or "trig" for trigger (only if modulation period is specified)')
args = parser.parse_args()
ich_trig = args.ich_trig
if ich_trig == -1:
    ich_trig = None
    
params_str = '\n'.join([f'   - {x}' for x in [
    f'f = {args.freq:.2f} MHz',
    f'A = {args.vpp:.3f} Vpp',
    f'ncycles = {args.ncycles}',
    f'BRF = {args.BRF:.2f} Hz',
    f'ramp-time = {args.tramp:.1f} ms'
]])

try:
    # Grab function generator
    wg = grab_generator()
    
    # Set looping sine burst
    logger.info(f'setting looping sine burst parameters:\n{params_str}')
    wg.set_looping_sine_burst(
        args.ich,
        args.freq * MHZ_TO_HZ,  # Hz
        args.vpp,  # Vpp
        args.ncycles,
        args.BRF,  # Hz
        ich_trig=ich_trig,
        tramp=args.tramp * MS_TO_S,  # s
    )

    # Release function generator
    time.sleep(1.)
    wg.disconnect()
    
except VisaError as e:
    logger.error(e)
    quit()
