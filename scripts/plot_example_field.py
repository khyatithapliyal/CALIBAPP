# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-05-01 12:42:15
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-05-11 15:47:02

''' Plot example pressure field '''

from usnmexps.utils import check_conda_env
check_conda_env()

import os
import numpy as np
import matplotlib.pyplot as plt

from instrulink import logger
from usnmexps.pltutils import plot_slices
from usnmexps.config import MAPPING_DATA_FOLDER

# XYZ coordinate vectors
x = np.linspace(-500, 500, 21)
y = x
z = np.linspace(1300, 2800, 31)

# Generate serialized XYZ grid from x, y and z vectors
XYZ = np.array([xx.ravel() for xx in np.meshgrid(x, y, z, indexing='ij')]).T

# Get serialized field
logger.info('loading 3D data')
data = np.loadtxt(os.path.join(MAPPING_DATA_FOLDER, 'example_field.csv'), delimiter=',')
# Reshape 3D field
data = data.reshape((y.size, z.size, x.size))
data = np.swapaxes(data, 1, 2)
# Manually curate x-slices order
xorder = [1, 12, 14, 15, 16, 17, 18, 19, 20, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13]
data = data[xorder, :, :]
# Visualize 3D field
fig = plot_slices(x, y, z, data, 'mm', 'P', None, title='data')

plt.show()
