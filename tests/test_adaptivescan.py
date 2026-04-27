# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-04-30 07:58:58
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-03-03 16:41:34

''' Test adaptive scanner '''

import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from instrulink import grab_manipulator, logger, SutterError

from usnmexps.constants import *
from usnmexps.transforms import MultiTransform, transform_func
from usnmexps.scanners import AdaptiveScanner
from usnmexps.utils import make_gauss, eval_on_grid, linvec, scan_proof
from usnmexps.pltutils import plot_slices

# Parse command-line arguments
parser = ArgumentParser()
parser.add_argument(
    '--res', type=float, default=.1, help='sampling resolution (mm)')
parser.add_argument(
    '--theta', type=float, default=0., help='angle of scanning system (degrees)')
parser.add_argument(
    '--maxiter', type=int, default=20, help='maximum number of iterations')
args = parser.parse_args()

# Define XYZ relative coordinates vectors
xybounds = (-1, 1)  # mm
zbounds = (0, 6)  # mm
x = linvec(*xybounds, args.res) * MM_TO_UM  # um
y = linvec(*xybounds, args.res) * MM_TO_UM  # um
z = linvec(*zbounds, args.res) * MM_TO_UM  # um

# Define scanner vector basis and scanning angle 
vbasis = [1, 1, -1]
theta = {'Y': args.theta * np.pi / 180}

# Grab micro-manipulator, if any
try:
    mp = grab_manipulator()
    p0 = mp.get_position()  # um
except (ValueError, SutterError) as e:
    logger.error(e)
    mp = None
    p0 = np.zeros(3)  # um

# Define evaluation function as a gaussian peaking within grid limits
feval = make_gauss(
    x=x + p0[0], 
    y=y + p0[1],
    z=z + p0[2], 
    rel_x0=0.2,
    rel_y0=0.8
)

# Evaluate function along XYZ grid and plot ground truth field
pred = eval_on_grid(feval, p0, x=x, y=y, z=z)
fig = plot_slices(
    x / MM_TO_UM, y / MM_TO_UM, z / MM_TO_UM, pred, 'mm', 'out', None)

# Modify evaluation function to bring transformed coordinates back to
# "normal" coordinate system prior to evaluation
M = MultiTransform()
if theta is not None:
    M.add_rotations(theta)
if vbasis is not None:
    M.add_symmetries(vbasis)
feval = transform_func(feval, M.inverse())

# Scan grid
try:
    scanner = AdaptiveScanner(mp=mp, theta=theta, vbasis=vbasis)
    coords_per_dim, data = scanner.scan(
        x=x, y=y, z=z, feval=scan_proof(feval), 
        plot=True, reset=True, order='XYZ', max_iter=args.maxiter)
    coords_per_dim = {k: v / MM_TO_UM for k, v in coords_per_dim.items()}
except SutterError as e:
    logger.error(e)
    quit()

# Disconnect from manipulator and log completion
if mp is not None:
    mp.disconnect()
logger.info('done.')

# Plot scan results
xyz = [None] * 3
for k, v in coords_per_dim.items():
    xyz['XYZ'.index(k)] = v
fig = plot_slices(*xyz, data, 'mm', 'out', None)

# Plot figures
plt.show()