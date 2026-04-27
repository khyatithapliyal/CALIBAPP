# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-02-25 15:26:13
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-05-11 15:46:40

'''
Plot comparative calibration curves (and their ratios) obtained between 2 hydrophones.
'''

from usnmexps.utils import check_conda_env
check_conda_env()

import os
from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from instrulink import logger
from usnmexps.constants import *
from usnmexps.calib_utils import parse_outputs_by_date

# Parse command line arguments
parser = ArgumentParser()
parser.add_argument(
    '-s', '--save', default=False, action='store_true', help='Save output figures as PDF')
args = parser.parse_args()

# Get calibration data
fname = 'transducer_6-8-22#1_glass2layers_2.030MHz.xlsx'
fpath = os.path.join(TRANSDUCERS_FOLDER, fname)
logger.info(f'loading calibration data from {fname}...')
data = pd.read_excel(fpath, engine='openpyxl').set_index(VIN_KEY)

# Parse calibration curves by date
pouts_by_date = parse_outputs_by_date(data.columns)

# Create ratio dataframe
npercond = 5
ratios = pd.DataFrame(index=data.index)
calib_comps = []
calib_PA = []

# For each date
logger.info('classifying data by date and hydrophone')
for cdate, cols in pouts_by_date.items():
    logger.info(cdate)
    # Extract and average calibration curves from both hydrophones
    calib1 = data.loc[:, cols[:npercond]].mean(axis=1)
    calib2 = data.loc[:, cols[npercond:]].mean(axis=1)
    # Merge and rename according to maximum value
    calib_comp = pd.concat([calib1, calib2], axis=1)
    imax = calib_comp.max(axis=0).argmax()
    calib_comp.rename(columns={imax: 'PA', 1 - imax: 'Onda'}, inplace=True)
    calib_comp = calib_comp[['PA', 'Onda']]
    calib_PA.append(calib_comp['PA'].rename(cdate))
    # Compute and store ratio
    ratios[cdate] = calib_comp['PA'] / calib_comp['Onda']
    # Add comparative curves with date
    calib_comps.append(calib_comp.add_prefix(f'{cdate} - '))
calib_comps = pd.concat(calib_comps, axis=1)

# Plot mean +/- std of PA calibration curves
calib_PA = pd.concat(calib_PA, axis=1)
calib_PA.index = calib_PA.index / 1000
fig, ax = plt.subplots()
sns.despine(ax=ax)
mu, sem = calib_PA.mean(axis=1), calib_PA.sem(axis=1)
ax.set_xlabel('input voltage (Vpp)')
ax.set_ylabel('peak-to-peak pressure amplitude (MPa)')
ax.plot(mu.index, mu, c='C0')
ax.fill_between(sem.index, mu - sem, mu + sem, fc='C0', alpha=0.5)
ax.set_ylim(0, ax.get_ylim()[1])
ax.set_xlim(0, ax.get_xlim()[1])

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot comparative calibration curves per date
ax = axes[0]
ax.set_title('comparative calibration curves')
ax.set_ylabel('peak pressure amplitude (MPa)')
sns.lineplot(
    ax=ax, data=calib_comps, palette='tab20', dashes=False)
ax.set_xlim(0, ax.get_xlim()[1])
ax.set_ylim(0., ax.get_ylim()[1])

# Plot ratios vs. input
ax = axes[1]
ax.set_title('pressure ratios vs. input')
ax.set_ylabel('pressure ratio')
ratios.loc[ratios.index < 60., :] = np.nan
ratios.plot(ax=ax)
med_ratios = ratios.median(axis=0)
for med_ratio in med_ratios:
    ax.axhline(med_ratio, c='k', ls='--')
ax.set_xlim(0, ax.get_xlim()[1])
ax.set_ylim(0., ax.get_ylim()[1])

# Plot median ratios and mean +/- sem amongst valid points
med_ratios = med_ratios.rename('value').to_frame()
med_ratios['date'] = med_ratios.index
ax = axes[2]
ax.set_title('median pressure ratios')
ax.set_ylabel('pressure ratio')
sns.barplot(
    data=med_ratios, x='date', y='value', ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)
medratio_mean = med_ratios['value'].mean()
medratio_sem = med_ratios['value'].sem()
cv = medratio_sem / medratio_mean
ax.axhline(medratio_mean, ls='--', c='k')
ax.axhspan(
    medratio_mean - medratio_sem, medratio_mean + medratio_sem,
    fc='silver', ec='k', alpha=0.5, zorder=0)
ax.text(
    0.15, 0.9,
    f'average ratio:\n{medratio_mean:.2f} +/- {medratio_sem:.2f} (i.e. +/- {cv * 1e2:.0f}%)',
    transform=ax.transAxes)

# Despine all figures axes
for ax in axes:
    sns.despine(ax=ax)
fig.tight_layout()

plt.show()