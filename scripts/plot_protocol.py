# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-11-17 16:09:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-05-11 15:48:03

''' Plot pressure x duty cycle protocol in Ispta space '''

from usnmexps.utils import check_conda_env
check_conda_env()

import os
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from usnmexps.dialog import open_folder_dialog
from usnmexps.constants import *
from usnmexps.pltutils import plot_P_DC_protocol

# Parse command line arguments
parser = ArgumentParser()
parser.add_argument(
    '-s', '--save', default=False, action='store_true', help='Save output figures as PDF')
args = parser.parse_args()

# Load P - DC combinations from protocol file
protocol_fpath = os.path.join(PROTOCOLS_FOLDER, 'protocol_P_DC old.xlsx')
df = pd.read_excel(protocol_fpath, engine='openpyxl')

# Plot protocol
fig = plot_P_DC_protocol(df[P_KEY], df[DC_KEY])

# Save if specified
if args.save:
    outdir = open_folder_dialog()
    if outdir is not None:
        fig.savefig(os.path.join(outdir, 'P_DC_protocol.pdf'))

# Render
plt.show() 