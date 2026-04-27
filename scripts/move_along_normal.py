# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2024-03-20 10:20:56
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-03-31 09:46:20

from usnmexps.utils import check_conda_env
check_conda_env()

from argparse import ArgumentParser

from instrulink import logger, grab_manipulator, SutterError
from usnmexps.constants import *
from usnmexps.scanners import Scanner

# Parse command line arguments
parser = ArgumentParser(
    description='Move the stage along vector normal to reference plane, by a specific distance')
parser.add_argument(
    '--theta', type=float, default=THETAY_SCAN, help='angle of scanning system (degrees)')
parser.add_argument(
    '-d', '--distance', type=float, default=DEFAULT_MOVEAWAY_DIST,
    help='translation distance (um, positive means away from reference plane)')
args = parser.parse_args()
theta = {'Y': args.theta * DEG_TO_RAD}
d = args.distance

# Log command line arguments
logger.info(f'theta = {theta}')
logger.info(f'distance = {d} um')

# Attempt to grab micro-manipulator, and quit if failed
try:
    mp = grab_manipulator()
except SutterError as e:
    logger.error(e)

# Initialize scanner with appropriate vector basis and angle
scanner = Scanner(mp=mp, vbasis=VBASIS, theta=theta)

# Move stage away from reference plane by specified distance
try:
    scanner.move_along_normal_vector(d)
except SutterError as e:
    logger.error(e)
    quit()

# Disconnect from micro-manipulator
mp.disconnect()
logger.info('done')
