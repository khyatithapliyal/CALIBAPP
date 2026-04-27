# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2023-04-03 13:53:37
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2024-07-25 16:53:07

''' Convert peak-pressure values to ultrasonic intensity values '''

from usnmexps.utils import check_conda_env
check_conda_env()

from argparse import ArgumentParser

from usnmexps.utils import pressure_to_intensity

# Define command-line options
parser = ArgumentParser()
parser.add_argument(
    '-p', type=float, help='peak-pressure amplitude (MPa)', required=True)
parser.add_argument(
    '--rho', type=float, default=1046.0, help='medium density (kg/m3)')
parser.add_argument(
    '-c', type=float, default=1546.3, help='speed of sound in medium (m/s)')
parser.add_argument(
    '--DC', type=float, default=100., help='Duty cycle (%%)')

# Parse command line arguments
args = parser.parse_args()
P = args.p  # MPa
DC = args.DC
if DC < 0 or DC > 100:
    raise ValueError(f'Invalid duty cycle: {DC} (must be within [0% - 100%] interval)')

# Compute and print intensity (in W/cm2)
print(f'computing intensity for P = {P} MPa, DC = {DC} %, rho = {args.rho} kg/m3, c = {args.c} m/s')
Isppa = pressure_to_intensity(P * 1e6, rho=args.rho, c=args.c) * 1e-4  # W/cm2
print(f'Isppa = {Isppa:g} W/cm2')
if DC < 100:
    Ispta = Isppa * DC / 100  # W/cm2
    print(f'Ispta = {Ispta:g} W/cm2')