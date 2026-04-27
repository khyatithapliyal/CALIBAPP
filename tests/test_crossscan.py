# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-05-02 09:58:56
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-02-19 12:37:26

from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
from instrulink import grab_manipulator, SutterError, logger

from usnmexps.constants import *
from usnmexps.transforms import MultiTransform, transform_func
from usnmexps.scanners import CrossSearchScanner
from usnmexps.utils import make_gauss, scan_proof, eval_on_grid, idxmax
from usnmexps.pltutils import plot_slices

# Parse command-line arguments
parser = ArgumentParser()
parser.add_argument(
    '--nx', type=int, default=5, help='# sampled coordinates on X axis')
parser.add_argument(
    '--ny', type=int, default=5, help='# sampled coordinates on Y axis')
parser.add_argument(
    '--nz', type=int, default=5, help='# sampled coordinates on Z axis')
parser.add_argument(
    '--theta', type=float, default=0., help='angle of scanning system (degrees)')
args = parser.parse_args()

# Define XYZ relative coordinates vectors
x = np.linspace(-1, 1, args.nx) * MM_TO_UM  # um
y = np.linspace(-1, 1, args.ny) * MM_TO_UM # um
z = np.linspace(-3, 3, args.nz) * MM_TO_UM # um

# Define scanner vector basis and scanning angle 
vbasis = [1, 1, -1]
theta = {'Y': args.theta * np.pi / 180}

# Grab micro-manipulator, if any
try:
    mp = grab_manipulator()
    p0 = mp.get_position()
except SutterError as e:
    logger.error(e)
    mp = None
    p0 = np.zeros(3)

# Define evaluation function as a gaussian peaking within grid limits
feval = make_gauss(
    x=x + p0[0], 
    y=y + p0[1], 
    z=z + p0[2],
    rel_x0=0.2,
    rel_y0=0.8,
)

# Evaluate function along XYZ grid
pred = eval_on_grid(feval, p0, x=x, y=y, z=z)

# Plot field
fig = plot_slices(
    x / MM_TO_UM, y / MM_TO_UM, z / MM_TO_UM, pred, 'mm', 'out', None)

# Extract max XYZ location and plot it on slice
imax = idxmax(pred)
xyzmax_pred = np.array([x[imax[0]], y[imax[1]], z[imax[2]]]) # + p0
islice_max = np.argmin(np.abs(x - xyzmax_pred[0]))
fig.axes[islice_max].scatter(*xyzmax_pred[1:] / MM_TO_UM, fc='r', ec='none')

# Modify evaluation function to bring transformed coordinates back to
# "normal" coordinate system prior to evaluation
M = MultiTransform()
if theta is not None:
    M.add_rotations(theta)
if vbasis is not None:
    M.add_symmetries(vbasis)
feval = transform_func(feval, M.inverse())

# Transform max on scanner coordinate system
xyzmax_pred = M.apply(xyzmax_pred) + p0

# Cross-scan
try:
    scanner = CrossSearchScanner(mp=mp, theta=theta, vbasis=vbasis)
    scanlocs, scanvals = scanner.scan(
        x=x, y=y, z=z, feval=scan_proof(feval), 
        plot=True, niters=2, order='ZXY')
    logger.info(f'True XYZ max: {xyzmax_pred / MM_TO_UM} mm')
    scanner.ax.scatter(*(xyzmax_pred / MM_TO_UM), c='k')
except SutterError as e:
    logger.error(e)
    quit()

# Disconnect from manipulator and log completion
mp.disconnect()
logger.info('done.')

# Extract max location, and transform it back to original coorindate system
max_xyz = scanlocs[np.argmax(scanvals)] - p0  # um
max_xyz = M.inverse().apply(max_xyz)

# Plot max location on appropriate slice
islice_max = np.argmin(np.abs(x - max_xyz[0]))
fig.axes[islice_max].scatter(*max_xyz[1:] / MM_TO_UM, fc='r', ec='none')

# Plot figures
plt.show()