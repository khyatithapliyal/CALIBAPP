# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2023-04-03 13:53:37
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-04-05 13:18:57

''' Convert ultrasonic intensity values to peak-pressure values '''

from usnmexps.utils import check_conda_env
check_conda_env()

from argparse import ArgumentParser
from usnmexps.utils import intensity_to_pressure

# Define command-line options
parser = ArgumentParser()
parser.add_argument(
    '-I', type=float, help='time-average intensity value (W/cm2)', required=True)
parser.add_argument(
    '--rho', type=float, default=1046.0, help='medium density (kg/m3)')
parser.add_argument(
    '-c', type=float, default=1546.3, help='speed of sound in medium (m/s)')
parser.add_argument(
    '--DC', type=float, default=100., help='Duty cycle (%%)')

# Parse command line arguments
args = parser.parse_args()
Ispta = args.I  # W/cm2
DC = args.DC
if DC < 0 or DC > 100:
    raise ValueError(f'Invalid duty cycle: {DC} (must be within [0% - 100%] interval)')

# Compute and print peak-pressure amplitude (in MPa)
print(f'computing peak pressure amplitude for Ispta = {Ispta} W/cm2, DC = {DC} %, rho = {args.rho} kg/m3, c = {args.c} m/s')
Isppa = Ispta / (DC / 100)  # W/cm2
P = intensity_to_pressure(Isppa * 1e4, rho=args.rho, c=args.c) * 1e-6  # MPa
print(f'P = {P:g} MPa')