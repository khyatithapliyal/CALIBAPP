# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-09-09 11:07:46
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-06-13 13:26:39

''' Load mapping results from file and plot resulting acoustic field '''

from usnmexps.utils import check_conda_env
check_conda_env()

import matplotlib.pyplot as plt
from argparse import ArgumentParser

from instrulink import logger
from usnmexps.config import MAPPING_DATA_FOLDER
from usnmexps.constants import *
from usnmexps.calib_utils import load_mapping_data
from usnmexps.pltutils import plot_acoustic_field
from usnmexps.dialog import open_file_dialog
from usnmexps.utils import get_last_file, update_last_file

# Parse command line arguments
parser = ArgumentParser()
parser.add_argument(
    '--slice', type=str, default='auto', choices=('x', 'y', 'z', 'auto', 'focus'), help='Slice axis')
parser.add_argument(
    '--cmap', type=str, default='viridis', help='Colormap')
parser.add_argument(
    '--markfocus', type=str, default=None, choices=('center', 'contour'), help='Focus marker (only in focus mode)')
parser.add_argument(
    '-i', '--interactive', default=False, action='store_true', help='Interactive mode')
parser.add_argument(
    '--rmode' , type=str, default='auto', choices=('auto', 'direct', 'GPR', 'interp'), help='Field reconstruction mode (defaults to auto, i.e. inferred from mapping file)')
parser.add_argument(
    '--xyzunit', type=str, choices=('mm', 'um'), default='mm', help='XYZ coordinates unit')
parser.add_argument(
    '--sigma', type=float, default=None, help='Standard deviation (in xyzunit) of gaussian kernel for gaussian filter based denoising of pressure field prior to plotting')
parser.add_argument(
    '-o', '--outmode', type=str, choices=('amp', 'int'), default='amp', help='Output mode (amplitude or intensity)')
parser.add_argument(
    '-n', '--norm', default=False, action='store_true', help='Normalize data')
parser.add_argument(
    '--shading', default='flat', choices=('flat', 'gouraud'), help='Shading method to use')
parser.add_argument(
    '-z', '--zinterp', type=float, default=None, help='z-value at which to interpolate relative pressure (mm)')
parser.add_argument(
    '-s', '--save', default=False, action='store_true', help='Save output figures as PDF')
parser.add_argument(
    '--last', default=False, action='store_true', help='Plot from last opened file')
args = parser.parse_args()

# If specified, fetch last mapping file
if args.last:
    last_mapping_fpath = get_last_file('mapping')
    if last_mapping_fpath is None:
        logger.error('no last mapping file found')
        quit()
    mapping_fpaths = [last_mapping_fpath]

# Otherwise, fetch mapping files from dialog
else:
    mapping_fpaths = open_file_dialog(
        'csv',
        dirname=MAPPING_DATA_FOLDER,
        title='Select mapping results files',
        multiple=True)
    # If no mapping file was selected, quit
    if mapping_fpaths is None:
        quit()

# For each mapping results
for mapping_fpath in mapping_fpaths:
    
    # Check if traces sub-directory exists
    traces_dir = f'{os.path.splitext(mapping_fpath)[0]}_traces'
    if not os.path.isdir(traces_dir):
        traces_dir = None
        if args.interactive:
            logger.warning(
                f'no traces sub-directory found for mapping file "{os.path.basename(mapping_fpath)}" -> disabling interactive mode')
            args.interactive = False
    
    # Load XYZ mapping data
    coords_per_dim, Pmat, fcode = load_mapping_data(mapping_fpath, reconstruction_mode=args.rmode)
    
    # Plot results
    fig = plot_acoustic_field(
        coords_per_dim, Pmat, args.slice,
        xyz_unit=args.xyzunit,
        gaussian_sigma=args.sigma,
        out_mode=args.outmode,
        title=fcode.replace('_', ' '),
        mark_focus=args.markfocus,
        norm=args.norm,
        shading=args.shading, 
        cmap=args.cmap,
        interactive=args.interactive,
        traces_dir=traces_dir,
        zinterp=args.zinterp
    )
    
    # Save figure if needed
    if args.save:
        mapping_dir, mapping_file = os.path.split(mapping_fpath)
        mapping_code = os.path.splitext(mapping_file)[0]
        mapping_code = f'{mapping_code}_{args.outmode}_{args.slice}'
        if args.sigma is not None:
            mapping_code = f'{mapping_code}_gaussianfiltersigma{args.sigma}{args.xyzunit}'
        if args.norm:
            mapping_code = f'{mapping_code}_norm'
        if args.shading != 'flat':
            mapping_code = f'{mapping_code}_{args.shading}shading'
        fig.savefig(os.path.join(mapping_dir, f'{mapping_code}.pdf'))
    
    # Update last mapping file
    update_last_file('mapping', mapping_fpath)

# Plot results
plt.show()