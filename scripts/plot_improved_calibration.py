
'''
Load transducer calibration data with improved plotting features:
- Separate plots for better readability
- Enhanced color schemes with accessibility
- Thicker lines and better markers
- Improved legends and hover functionality
- Better figure sizing and layout
'''

from usnmexps.utils import check_conda_env
check_conda_env()

import matplotlib.pyplot as plt
from argparse import ArgumentParser

from instrulink import logger
from usnmexps.dialog import open_file_dialog
from usnmexps.constants import *
from usnmexps.pltutils import plot_calibration_curves
from usnmexps.utils import get_last_file, update_last_file

# Parse command line arguments
parser = ArgumentParser(description='Plot transducer calibration with enhanced readability')
parser.add_argument(
    '-d', '--details', default=True, action='store_true', help='Plot detailed traces (recommended for readability)')
parser.add_argument(
    '--logx', default=False, action='store_true', help='Turn on log x-scale')
parser.add_argument(
    '--logy', default=False, action='store_true', help='Turn on log y-scale')
parser.add_argument(
    '--last', default=False, action='store_true', help='Plot from last opened file')
parser.add_argument(
    '--yerr', default='sd', choices=['sd', 'se'], help='Error shading type')
parser.add_argument(
    '--yout', default='all', choices=['P', 'cpl', 'all'], help='Output unit(s) - "all" creates separate plots')
args = parser.parse_args()

# Parse output unit(s) - using 'all' by default for separate plots
yout = {
    'P': P_KEY, 
    'cpl': CPLRATIODB_KEY,
    'all': [P_KEY, CPLRATIODB_KEY]  # Creates separate plots for maximum readability
}[args.yout]

# If specified, fetch last calibration file
if args.last:
    last_calib_fpath = get_last_file('transducer')
    if last_calib_fpath is None:
        logger.error('no last transducer calibration file found')
        quit()
    calibration_fpaths = [last_calib_fpath]

# Otherwise, get calibration files list
else:
    calibration_fpaths = open_file_dialog(
        'xlsx',
        dirname=TRANSDUCERS_FOLDER,
        title='Select transducer calibration file', 
        multiple=True)
    if calibration_fpaths is None:
        quit()

print("Enhanced Plotting Features:")
print("- Separate plots for each measurement type")
print("- Expanded color palette for better distinction") 
print("- Thicker lines and larger markers for visibility")
print("- Interactive hover highlighting")
print("- Improved legend positioning and formatting")
print("- Better figure sizing and layout")
print()

# Plot calibration curves with enhanced features
fig = plot_calibration_curves(
    calibration_fpaths,
    details=args.details,  # Using details mode for better visibility
    logx=args.logx,
    logy=args.logy,
    yerr=args.yerr,
    ylabel=yout,
)

print("Plot created with enhanced readability features")
print("Hover over lines to highlight individual traces!")

# Update last calibration file
update_last_file('transducer', calibration_fpaths[-1])

plt.show()