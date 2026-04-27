# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-04-27 08:32:29
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-06-13 11:26:37

''' Calibration utilities '''

from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.interpolate import interp1d, griddata
from argparse import ArgumentParser
from instrulink.logger import logger

from .constants import *
from .config import MAPPING_DATA_FOLDER
from .dialog import open_file_dialog
from .utils import remove_almost_duplicates, get_closest, get_mux_slice, idxmax, today, save_to_excel_file
from .transforms import MultiTransform
from .adaptive_mappers import GPRFieldMapper


def get_calibration_parser(ncycles, add_vpp=True):
    ''' Get a command line parser for calibration operations '''
    parser = ArgumentParser()
    parser.add_argument(
        '--wchtrig', type=int, default=STIM_CH_TRIG, choices=(1, 2), help='Generator trigger channel index')
    parser.add_argument(
        '--wchsig', type=int, default=STIM_CH_SIGNAL, choices=(1, 2), help='Generator signal channel index')
    parser.add_argument(
        '--scope', type=str, default=None, help='Oscilloscope model')
    parser.add_argument(
        '--schtrig', type=int, default=ACQ_CH_TRIG, choices=(1, 2, 3, 4), help='Oscilloscope trigger channel index')
    parser.add_argument(
        '--schsig', type=int, default=ACQ_CH_SIGNAL, choices=(1, 2, 3, 4), help='Oscilloscope signal channel index')
    parser.add_argument(
        '--sch-cpl-fwd', type=int, default=-1, choices=(-1, 1, 2, 3, 4), help='Oscilloscope forward coupling channel index')
    parser.add_argument(
        '--sch-cpl-rev', type=int, default=-1, choices=(-1, 1, 2, 3, 4), help='Oscilloscope reverse coupling channel index')
    parser.add_argument(
        '--gtype', default='trig', choices=('mod', 'trig'), 
        help='Gating type, i.e. "mod" for modulation or "trig" for trigger')    
    parser.add_argument(
        '--tmode', default='loop', choices=('loop', 'prog'),
        help='Generator trigger mode, i.e. "loop" for continous or "prog" for programmatic trigger')
    parser.add_argument(
        '--beep', default=False, action='store_true', help='Beep on trigger (valid for "single" trigger mode only')
    parser.add_argument(
        '--ncycles', type=int, default=ncycles, help='Number of cycles per burst')
    if add_vpp:
        parser.add_argument(
            '--vpp', type=float, default=REF_VPP_CALIB, help='Signal amplitude (in Vpp)')
    parser.add_argument(
        '--PRF', type=float, default=DEFAULT_PRF, help='Pulse repetition frequency (Hz)')
    parser.add_argument(
        '--acqinterval', type=float, default=OSC_ACQ_INTERVAL, help='Inter-acquisition interval (s)')
    parser.add_argument(
        '--acqnpoints', type=int, default=OSC_ACQ_NPOINTS, help='# points per oscilloscope acquisition')
    parser.add_argument(
        '--acqnsweeps', type=int, default=OSC_ACQ_NSWEEPS, help='# sweeps per oscilloscope acquisition')
    parser.add_argument(
        '--npercond', type=int, default=NPERCOND_CALIB, help='Number of acquisitions per condition')
    parser.add_argument(
        '--vdetect', default=False, action='store_true',
        help='Automatically detect the vertical scale of the oscilloscope signal channel')
    parser.add_argument(
        '--vfix', default=False, action='store_true',
        help='Keep the vertical scale of the oscilloscope signal channel constant')
    parser.add_argument(
        '-k-', '--keepfigs', default=False, action='store_true', help='Keep figures open at the end of the calibration')
    return parser


def parse_calibrator_settings(args):
    ''' 
    Parse calibrator settings from command command line arguments
    
    :param args: command line arguments
    :return: 3-tuple with positional and keyword arguments required for calibrator initialization, 
        and positional arguments required for procedure execution
    '''
    # Assemble calibrator initialization positional arguments
    init_args = [
        args.wchsig, 
        args.schsig, 
    ]

    # Gather scope channels and check their validity
    scope_secondary_channels = {
        'trig': args.schtrig, 
        'cpl_fwd': args.sch_cpl_fwd, 
        'cpl_rev': args.sch_cpl_rev
    }
    scope_secondary_channels = {k: None if v == -1 else v for k, v in scope_secondary_channels.items()}
    defined_scope_secondary_channels = {k: v for k, v in scope_secondary_channels.items() if v is not None}
    if len(defined_scope_secondary_channels) != len(set(defined_scope_secondary_channels.values())):
        raise ValueError(f'non-unique secondary channels: {defined_scope_secondary_channels}')
    for k, v in defined_scope_secondary_channels.items():
        if k != 'trig' and v == args.sch:
            raise ValueError(f'scope signal channel cannot be used as {k} channel')
    
    # Assemble calibrator initialization keyword arguments
    init_kwargs = {
        'wch_trig': None if args.wchtrig == -1 else args.wchtrig,
        **{f'sch_{k}': v for k, v in scope_secondary_channels.items()},
        'gate_type': args.gtype,
        'trigger_mode': args.tmode,
        'beep_on_trigger': args.beep,
        'detect_vscale': args.vdetect,
        'ncycles': args.ncycles, 
        'PRF': args.PRF, 
        'acqnpoints': args.acqnpoints,
        'acqnsweeps': args.acqnsweeps
    }

    # Assemble execution arguments
    exec_args = []
    if hasattr(args, 'vpp'):
        exec_args.insert(0, args.vpp)

    # Return initialization calibrator arguments and keyword arguments
    return init_args, init_kwargs, exec_args


def add_scan_args(parser, delta, default_order='XYZ', res=None, niters=None, mmode=None):
    ''' Add scanning arguments to parser '''
    # Search extent arguments
    if not isinstance(delta, dict):
        delta = {k: delta for k in 'XYZ'}
    parser.add_argument(
        '--Dx', type=float, default=delta['X'] / MM_TO_UM, help='Search extent on X axis (mm)')
    parser.add_argument(
        '--Dy', type=float, default=delta['Y'] / MM_TO_UM, help='Search extent on y axis (mm)')
    parser.add_argument(
        '--Dz', type=float, default=delta['Z'] / MM_TO_UM, help='Search extent on Z axis (mm)')

    # Optional search resolution arguments 
    if res is not None:
        if not isinstance(res, dict):
            res = {k: res for k in 'XYZ'}
        parser.add_argument(
            '--dx', type=int, default=res['X'] / MM_TO_UM, help='scanning resolution on X axis (mm)')
        parser.add_argument(
            '--dy', type=int, default=res['Y'] / MM_TO_UM, help='scanning resolution on Y axis (mm)')
        parser.add_argument(
            '--dz', type=int, default=res['Z'] / MM_TO_UM, help='scanning resolution on Z axis (mm)')
        parser.add_argument(
            '--sparse', default=False, action='store_true', help='Sparse scanning mode')

    # Angle and scanning order
    parser.add_argument(
        '--theta', type=float, default=THETAY_SCAN, help='angle of scanning system (degrees)')
    parser.add_argument(
        '-o', '--order', type=str, default=default_order, help='Coordinates scanning order')
    
    # Optional search iterations
    if niters is not None:
        parser.add_argument(
            '--niters', type=int, default=niters, help='# successive cross-search iterations')
    
    # Starting and ending modes
    parser.add_argument(
        '-e', '--endmode', type=str, default='gotomax', 
        choices=SCAN_ENDMODES, help='What to do at the end of the procedure')
    parser.add_argument(
        '--zstart', type=str, default='center', choices=('base', 'center'), help='Initial z position')
    
    # Optional mapping mode
    if mmode is not None:
        parser.add_argument(
            '--mmode', type=str, default=mmode, choices=MAPPING_MODES, help='Mapping mode')


def parse_scan_args(args):
    ''' 
    Parse scanning coordinates 
    
    :param args: command line arguments
    :return: positional and keyword arguments required for scanning procedures
    '''
    # Assemble positional arguments
    delta = {'X': args.Dx * MM_TO_UM, 'Y': args.Dy * MM_TO_UM, 'Z': args.Dz * MM_TO_UM}  # um
    if hasattr(args, 'res'):
        res = {'X': args.dx * MM_TO_UM, 'Y': args.dy * MM_TO_UM, 'Z': args.dz * MM_TO_UM}  # um
        if args.sparse:
            res = {k: RES_SCAN_SPARSE for k in res.keys()}  # um
        nperax = {k: int(np.round(delta[k] / res[k])) + 1 for k in 'XYZ'}
    else:
        nperax = {k: NPERAX_SEARCH for k in 'XYZ'}
    theta = {'Y': args.theta * DEG_TO_RAD}
    scan_args = (delta, nperax, theta)
    
    # Assemble keyword arguments
    scan_kwargs = {
        'order': args.order,
        'niters': args.niters,
        'endmode': args.endmode,
        'zstart': args.zstart
    }
    if hasattr(args, 'mmode'):
        scan_kwargs['mmode'] = args.mmode

    # Return
    return scan_args, scan_kwargs


def get_hydrophone_calibration_file():
    ''' Get full path to hydrophone calibration file. '''
    fpath = open_file_dialog(
        'xlsx', dirname=HYDROPHONES_FOLDER, title='Select hydrophone calibration file')
    if fpath is None:
        raise ValueError('no file was selected')
    return fpath


def get_hydrophone_calibration_data(fpath):
    ''' Load hydrophone calibration data from file. '''
    fname = os.path.basename(fpath)
    logger.info(f'loading hydrophone calibration data from {fname}...')
    return pd.read_excel(fpath, engine='openpyxl')


def get_hydrophone_conversion_constant(data, f):
    '''
    Get hydrophone voltage-to-pressure conversion constant at a specific frequency
    
    :param data: hydrophone calibration dataframe
    :param f: frequency (MHz)
    :return: conversion factor (in V/Pa)
    '''
    fbounds = (data[F_KEY].min(), data[F_KEY].max())
    if f < fbounds[0] or f > fbounds[1]:
        logger.warning(f'frequency {f:.3f} MHz is out of bounds: {fbounds}')
    Mlbounds = (data[CONV_KEY].iloc[0], data[CONV_KEY].iloc[-1])
    finterp = interp1d(
        data[F_KEY], data[CONV_KEY], 
        fill_value=Mlbounds,
        bounds_error=False
    )
    Ml = finterp(f)
    logger.info(f'hydrophone conversion constant at {f:.3f} MHz: Ml = {Ml:.3e} V/Pa')
    return Ml


def check_input_voltages(Vpps_in):
    ''' Check input voltage vector '''
    if np.any(np.sort(Vpps_in) != Vpps_in):
        raise ValueError('input voltage vector should be monotonically increasing')
    if Vpps_in.min() != 0.:
        raise ValueError('the first input voltage value should be zero')


def get_transducer_calibration_file():
    ''' Get the full path to a transducer calibration Excel file '''
    fpath = open_file_dialog(
        'xlsx', dirname=TRANSDUCERS_FOLDER, title='Select transducer calibration file')
    if fpath is None:
        raise ValueError('no file was selected')
    return fpath


def get_transducer_mapping_file():
    ''' Get the full path to a transducer mapping CSV file '''
    fpath = open_file_dialog(
        'csv', dirname=MAPPING_DATA_FOLDER, title='Select transducer calibration file')
    if fpath is None:
        raise ValueError('no file was selected')
    return fpath


def get_amplifier_calibration_file():
    ''' Get the full path to an amplifier calibration Excel file '''
    fpath = open_file_dialog(
        'xlsx', dirname=AMPLIFIERS_FOLDER, title='Select amplifier calibration file')
    if fpath is None:
        raise ValueError('no file was selected')
    return fpath


def parse_transducer_calibration_file(fpath):
    '''
    Parse transducer calibration file

    :param fpath: full path to a transducer calibration Excel file
    :return: dictionary of parsed transducer calibration parameters
    '''
    fname = os.path.basename(fpath)
    mo = TRANSDUCER_CALIBRATION_FILE_PATTERN.match(fname)
    if not mo:
        raise ValueError(f'could not parse transducer calibration file name: {fname}')
    return {
        'transducer_id': mo[1],
        'conditions': mo[2],
        'freq (MHz)': float(mo[3])
    }


def get_focal_distance(transducer_id, date=None):
    '''
    Get focal distance for a specific transducer ID (and date)
    
    :param transducer_id: transducer ID
    :param date: date (in YYYY.MM.DD format) for which to get the focal distance. 
        If None, the last measured focal distance is returned.
    :return: focal distance (in um)
    '''
    # If file does not exist, raise error
    if not os.path.isfile(FOCAL_DISTANCES_FPATH):
        raise ValueError(f'focal distances file not found: {FOCAL_DISTANCES_FPATH}')
    
    # Load focal distances data
    df = pd.read_excel(FOCAL_DISTANCES_FPATH, engine='openpyxl')
    df = df.set_index('date')
    
    # If transducer ID is not in the data, raise error
    if transducer_id not in df.columns:
        raise ValueError(f'transducer ID "{transducer_id}" not found in focal distances file')
    
    # Extract saved focal distances for the transducer ID
    s = df[transducer_id].dropna()
    
    # If date is not specified, extract the last focal distance
    if date is None:
        d = s.iloc[-1]
    
    else:
        # If date not in the data, raise error
        if date not in s.index:
            raise ValueError(f'date "{date}" not found in focal distances file')
    
        # Extract focal distance for the specified date
        d = s[date]
    
    logger.info(f'found focal distance for transducer ID "{transducer_id}" on {date}: d = {d:.3f} um')
    return d


def save_focal_distance(transducer_id, d):
    '''
    Save focal distance for a specific transducer ID

    :param transducer_id: transducer ID
    :param d: focal distance (in um)
    '''
    logger.info(f'saving focal distance for transducer ID "{transducer_id}" to {FOCAL_DISTANCES_FPATH}...')
    # Load/create focal distances data table
    if os.path.isfile(FOCAL_DISTANCES_FPATH):
        df = pd.read_excel(FOCAL_DISTANCES_FPATH, engine='openpyxl')
        df = df.set_index('date')
    else:
        df = pd.DataFrame()
        df.index.name = 'date'
    
    # If transducer ID is not in the data, add it
    if transducer_id not in df.columns:
        df[transducer_id] = np.nan
    
    # If today's date is not in the data, add it
    date = today()
    if date not in df.index: 
        df.loc[date] = np.nan
    
    # If focal distance for the transducer ID and date already exists, log warning
    if not np.isnan(df.loc[date, transducer_id]):
        logger.warning(f'overwriting existing focal distance for transducer ID "{transducer_id}" on {date}')
    
    # Set focal distance for the transducer ID and date
    df.loc[date, transducer_id] = np.round(d, 0)  # um

    # Save updated data to file
    save_to_excel_file(df.reset_index(), FOCAL_DISTANCES_FPATH)


def parse_amplifier_calibration_file(fpath):
    ''' 
    Parse amplifier calibration file
    
    :param fpath: full path to an amplifier calibration Excel file
    :return: dictionary of parsed amplifier calibration parameters
    '''
    fname = os.path.basename(fpath)
    mo = AMPLIFIER_CALIBRATION_FILE_PATTERN.match(fname)
    if not mo:
        raise ValueError(f'could not parse amplifier calibration file name: {fname}')
    return {
        'amplifier_id': mo[1],
        'gain (dB)': float(mo[2]),
        'freq (MHz)': float(mo[3])
    }


def extract_frequency(fpath):
    ''' 
    Extract US frequency from calibration file name
    
    :param fpath: full path to a calibration Excel file
    :return: extracted US frequency (in MHz)
    '''
    fname = os.path.basename(fpath)
    if fname.startswith('amplifier'):
        params = parse_amplifier_calibration_file(fpath)
    elif re.match(TRANSDUCER_ID_PATTERN, fname):
        params = parse_transducer_calibration_file(fpath)
    else:
        raise ValueError(f'unknown calibration file type: {fname}')
    f = params['freq (MHz)']
    logger.info(f'extracted US frequency: f = {f:.3f} MHz')
    return f


def extract_hydrophone_model(fpath):
    ''' Extract hydrophone model from hydrophone calibration file name'''
    fname = os.path.basename(fpath)
    mo = HYDROPHONE_CALIBRATION_FILE_PATTERN.match(fname)
    if not mo:
        raise ValueError('Could not extract hydrophone model from hydrophone file name')
    model = mo[2]
    logger.info(f'extracted hydrophone model: {model}')
    return model


def extract_gain(fpath):
    ''' 
    Extract gain from amplifier calibration file name
    
    :param fpath: full path to an amplifier calibration Excel file
    :return: extracted gain (in dB)
    '''
    params = parse_amplifier_calibration_file(fpath)
    gain = params['gain (dB)']
    logger.info(f'extracted gain: G = {gain:.2f} dB')
    return gain    


def find_new_entry_index(entries, rgxp):
    ''' 
    Find index of new entry of specific type in iterable
    
    :param entries: list of existing entries
    :param rgxp: regexp pattern to match
    :return: index of new entry
    '''
    imax = 0
    for entry in entries:
        mo = rgxp.match(entry)
        if mo is not None:
            i = int(mo[1])
            imax = max(i, imax)
    return imax + 1


def find_out_column_name(cols, key='Pout', unit='MPa'):
    '''
    Find a column name for a new calibration entry based on existing column names
    
    :param cols: list of existing column names in the calibration table
    :param key: output variable name (default = 'Pout')
    :param unit: output variable unit (default = 'MPa')
    :return: column name for new entry
    '''
    # Get regexp for output column with today's date
    rgxp = re.compile(f'{key} {today()} #({INT_REGEXP}) \({unit}\)')
    # Find index of new calibration entry for the day
    ientry = find_new_entry_index(cols, rgxp)
    # Return new column name
    return f'{key} {today()} #{ientry} ({unit})'


def is_file_writeable(fpath):
    ''' Check file availability by trying to rename it '''
    try:
        os.rename(fpath, fpath)
        return True
    except OSError as e:
        logger.warning(f'Access-error on {os.path.basename(fpath)} file')
        return False


def check_file_availability(fpath):
    ''' Check file availability by trying to rename it '''
    while not is_file_writeable(fpath):
        logger.error(f'close file and press enter to continue')
        input()


def convert_date_format(s):
    ''' 
    Find and replace date format from MM.DD.YY to YYYY.MM.DD in a string
    
    :param s: string that might contain date string
    :return: string in which the date string has been converted to new format
    '''
    def repl(match):
        mm, dd, yy = match.groups()
        yyyy = f'20{yy}'  # Assuming 20xx for the year
        return f'{yyyy}.{mm}.{dd}'
    return re.sub(r"(\d{2})\.(\d{2})\.(\d{2})", repl, s)


def convert_calibration_file(fpath):
    ''' Convert date format in column names of a dataframe '''
    # Load dataframe
    data = pd.read_excel(fpath, engine='openpyxl')
    # Convert column names
    data.columns = [convert_date_format(col) for col in data.columns]
    # Save to same file
    save_to_excel_file(data, fpath)
    logger.info(f'converted date format in {os.path.basename(fpath)}')


def parse_outputs_by_date(cols, rgxp=CALIBRATION_POUT_PATTERN):
    '''
    Arange calibration output columns by date
    
    :param cols: list of column names
    :param rgxp: regexp of calibration output columns
    :return: dictionary of column names by date
    '''
    # Initialize empty dates dictionary
    dates = {}
    # For each column
    for col in cols:
        # Parse column name
        mo = re.match(rgxp, col)
        # Raise error if invalid match
        if mo is None:
            raise ValueError(f'"{col}" column does not match expected pattern')
        # Extract date
        *date, acq = mo.groups()
        date = '.'.join(date)
        # Add column to appropriate date entry
        if date not in dates:
            dates[date] = [col]
        else:
            dates[date].append(col)
    # Return dates dictionary
    return dates


def pratio_to_gain(x):
    ''' Power ratio to gain or loss (in dB) '''
    return 10 * np.log10(x)

def gain_to_pratio(x):
    ''' Gain or loss (in dB) to power ratio '''
    return np.power(10, x / 10)

def vratio_to_gain(x):
    ''' Voltage ratio to gain or loss (in dB) '''
    return 2 * pratio_to_gain(x)

def gain_to_vratio(x):
    ''' Gain or loss (in dB) to voltage ratio '''
    return gain_to_pratio(x / 2)


def extract_amplifier_gain(fpath):
    '''
    Parse amplifier theoretical gain from calibration file path(s)
    
    :param fpath: full path to an amplifier calibration Excel file(s)
    :return: extracted gain (in dB)
    '''
    # If several file paths are provided
    if isinstance(fpath, (list, tuple, np.ndarray)):
        # Set reference gain and uniformity flag
        refampgain = None

        # Loop through calibration files
        for fp in fpath:
            # Call function recursively
            gain = extract_amplifier_gain(fp)

            # Update reference gain if not yet set
            if refampgain is None:
                refampgain = gain
            # Otherwise, raise error if different gain was found
            elif gain != refampgain:
                raise ValueError(f'found different amplifier gains: {refampgain} and {gain}')
        
        # Return reference gain
        return refampgain

    # Extract file name from path
    fname = os.path.basename(fpath)

    # Parse file name
    params = parse_amplifier_calibration_file(fname)

    # If parsing was successful
    logger.info(f'extracting gain from amplifier calibration file: {fname}')

    # Extract amplifier code
    amplifier_code = params['amplifier_id']

    # Match amplifier code to theoretical gain
    gain = next((v for k, v in AMPGAINS.items() if amplifier_code.startswith(k)), None)  # dB
    
    # Log warning if no match was found
    if gain is None:
        logger.warning(f'could not match amplifier code "{amplifier_code}" to theoretical gain')

    # Return extracted gain
    return gain


def compute_amplifier_gain_over_time(data):
    '''
    Compute amplifier gain over time

    :param data: dataframe containing amplifier calibration curves
    :return: 2-column dataframe with mean and standard error of amplifier gain over time
    '''
    # Extract input column
    Vpps_in = data[VIN_KEY] * MV_TO_V  # Vpp
    # Get output keys
    validation_pattern = CALIBRATION_VOUT_PATTERN
    output_cols = [k for k in data.columns if re.match(validation_pattern, k)]
    if not output_cols:
        logger.error('no valid output columns found')
        return
    data = data[output_cols]

    # Average output data by date
    outkeys_by_date = parse_outputs_by_date(data.columns, rgxp=validation_pattern)

    avgdata_bydate = pd.DataFrame({
        date: data[cols].mean(axis=1) for date, cols in outkeys_by_date.items()})

    # Compute equivalent gain vectors
    gain_by_date = pd.DataFrame({
        date: vratio_to_gain(vouts / Vpps_in) for date, vouts in avgdata_bydate.items()})
    
    # Compute mean gain and its standard error per date
    gain_by_date = gain_by_date.agg(['mean', 'sem'], axis=0).transpose()

    # Convert index to datetime
    gain_by_date.index = [datetime.strptime(d, DATE_FORMAT).date() for d in gain_by_date.index]
    
    # Return
    return gain_by_date


def parse_mapping_file(fpath):
    ''' Parse parameters from mapping file '''    
    fname = os.path.basename(fpath)
    mo = re.match(MAPPING_FILE_PATTERN, fname)
    if mo is None:
        raise ValueError(f'could not parse mapping parameters from "{fname}" (pattern: {MAPPING_FILE_PATTERN})')
    transducer_id, map_conds, freq, mode, opt_hydrophone, year, month, day, theta, dx, dy, dz, nx, ny, nz = mo.groups()
    if mode == '':
        mode = 'brute'
    if len(opt_hydrophone) > 0:
        opt_hydrophone = opt_hydrophone[:-1]
    else:
        opt_hydrophone = None
    params = {
        'transducer_id': transducer_id,
        'conditions': map_conds,
        'hydrophone': opt_hydrophone,
        'freq (MHz)': float(freq),
        'mode': mode,
        'date': f'{year}.{month}.{day}',
        'tilt (deg)': float(theta),
        'delta': dict(zip('XYZ', (float(dx) * MM_TO_UM, float(dy) * MM_TO_UM, float(dz) * MM_TO_UM))),  # mm
        'nperax': dict(zip('XYZ', (int(nx), int(ny), int(nz))))
    }
    return params


def load_mapping_data(fpath, reconstruction_mode='auto'):
    '''
    Load mapping data from file

    :param fpath: full path to mapping data file
    :param reconstruction_mode: reconstruction mode (optional). One of:
        - "direct": reconstructed as is (for strutured grid data)
        - "GPR": recontructed using Gaussian process regression
        - "interp": interpolated using griddata
        - "auto" (default): recontruction method parsed from input mapping mode
    :return: 3-tuple with:
        - dictionary of coordinates per dimension (um)
        - multi-dimensional array of pressure values along the grid 
        - mapping name (extracted from file name)
    '''
    # Extract file code
    fname = os.path.basename(fpath)
    fcode = os.path.splitext(fname)[0]
    
    # Parse mapping params from file
    map_params = parse_mapping_file(fname)
    theta_xz = map_params['tilt (deg)'] * np.pi / 180  # rad
    nperax = map_params['nperax']  # number of points per axis
    delta = map_params['delta']  # um
    mmode = map_params['mode']  # mapping mode

    # If reconstruction mode is not specified, set it depending on mapping mode
    if reconstruction_mode == 'auto':
        reconstruction_mode = {
            'brute': 'direct',
            'adaptive': 'GPR',
        }[mmode]

    # Check that reconstruction mode is valid 
    if reconstruction_mode not in ('direct', 'GPR', 'interp'):
        raise ValueError(f'unknown reconstruction mode "{reconstruction_mode}"')
    
    # Determine if mapping data is structured in XYZ
    is_structured = mmode == 'brute'

    # If file contains unstructured mapping data, check that requested reconstruction mode is valid
    if reconstruction_mode == 'direct' and not is_structured:
        raise ValueError(f'cannot recontruct field with "{reconstruction_mode}" method: input file contains unstructured mapping data')
    
    # If file contains brute-force data and reconstruction mode is not direct, 
    # increase resolution to 100 um for reconstruction grid
    if is_structured and reconstruction_mode != 'direct':
        res = 100  # um
        nperax = {k: max(nperax[k], int(np.round(v / res)) + 1) for k, v in delta.items()}

    # Load data from file
    logger.info(f'loading mapping data from "{fname}"')
    df = pd.read_csv(fpath)
    logger.info(f'found {len(df)} points in mapping data')
    coords_cols = [f'{k} (um)' for k in 'XYZ']
    coords = df[coords_cols].values
    P = df[P_KEY]

    # Transform coordinates back to reference XYZ
    M1 = MultiTransform()
    if theta_xz != 0.:
        M1.add_rotations({'Y': theta_xz})
    M1.add_symmetries(VBASIS)
    coords = M1.inverse().apply(coords, verbose=True)

    # Offset coordinates to align field on origin
    xyzbounds = np.stack([coords.min(axis=0), coords.max(axis=0)]).T
    org = np.array([np.mean(xyzbounds[0]), np.mean(xyzbounds[1]), xyzbounds[2][0]])
    M2 = MultiTransform()
    M2.add_translation(-org)
    coords = M2.apply(coords, verbose=True)

    # Brute-force reconstruction mode
    if reconstruction_mode == 'direct':
        # Reconstruct coordinate vectors per dimension
        logger.info(f'identifying "true uniques" coordinates in each dimension from {coords.shape[0]} XYZ coordinates')
        coords_per_dim = {k: remove_almost_duplicates(v) for k, v in zip('XYZ', coords.T)}

        # Check that coordinates reconstruction is valid
        dims = ([x.size for x in coords_per_dim.values()])
        if tuple(dims) != tuple(nperax.values()):
            raise ValueError(
                f'reconstruction error: dimensions of extracted coordinates {dims} do not match those on file {nperax}')
        if np.prod(dims) != len(df):
            raise ValueError(
                f'reconstruction error: grid dimensions {dims} do not match output length ({len(df)})')
        logger.info(f'reconstructed {dims} grid dimensions')
        
        # "Clean up" full coordinates vector
        coords = np.array([
            [get_closest(coords_per_dim[k], x) for k, x in zip('XYZ', p)] for p in coords])
        
        # Reconstruct 3D pressure field matrix
        Pmat = np.empty(dims)
        logger.info(f'reconstructing {dims} pressure field matrix')
        # For each serialized pressure value
        for vref, p in zip(coords, P):
            # Get 3D XYZ index from serialized index
            idx = [
                np.where(v == vref[j])[0][0]
                for j, v in enumerate(coords_per_dim.values())
            ]
            ix, iy, iz = idx
            Pmat[ix, iy, iz] = p
    
    # Non-direct reconstruction mode
    else:
        # Define bounds per dimension (um)
        bounds_per_dim = {
            'X': np.array([-delta['X'] / 2, delta['X'] / 2]),
            'Y': np.array([-delta['Y'] / 2, delta['Y'] / 2]),
            'Z': np.array([0, delta['Z']])
        }

        # Define coordinates per dimension
        coords_per_dim = {k: np.linspace(*v, nperax[k]) for k, v in bounds_per_dim.items()}  # um

        # Construct evaluation grid
        Xgrid = np.meshgrid(*coords_per_dim.values(), indexing='ij')  # um

        # GPR reconstruction mode
        if reconstruction_mode == 'GPR':
            
            # Check for existence of model file
            projkey = 'XYZ'
            model_fpath = os.path.splitext(fpath)[0] + f'_GPRmodel{projkey}.pkl'

            # If model file exists, make sure it is more recent than the data file
            if os.path.isfile(model_fpath):
                model_mtime = datetime.fromtimestamp(os.path.getmtime(model_fpath))
                data_mtime = datetime.fromtimestamp(os.path.getmtime(fpath))
                if model_mtime < data_mtime:
                    logger.warning(f'model file "{model_fpath}" is older than data file "{fpath}" -> retraining model')
                    os.remove(model_fpath)
                else:
                    logger.info(f'model file "{model_fpath}" is up to date')
            
            # If model file does not exist
            if not os.path.isfile(model_fpath):

                logger.info('training GPR model...')

                # Create mapper object
                mapper = GPRFieldMapper(
                    None,
                    {k: v / MM_TO_UM for k, v in bounds_per_dim.items()},  # mm
                    unit='mm',
                    projkey=projkey,
                )

                # Assign log training data and fit
                xtrain = dict(zip('XYZ', coords.T / MM_TO_UM))  # mm
                mapper.df_train = pd.DataFrame({
                    **xtrain,  # mm
                    'value': P
                })
                logger.info('fitting GPR model...')
                mapper.fit()

                # Save to file
                mapper.to_pickle(model_fpath)
            
            # Otherwise, load model from file
            else:
                logger.info('loading GPR model...')
                mapper = GPRFieldMapper.from_pickle(model_fpath)
            
            # Reconstruct pressure field over regular grid        
            logger.info(f'{mapper}: reconstructing pressure field over regular grid')
            Xgrid_mm = tuple(x / MM_TO_UM for x in Xgrid)  # mm
            Pmat = mapper.predict(Xgrid_mm)

        # Unstructured interpolation reconstruction mode
        elif reconstruction_mode == 'interp':
            logger.info('reconstructing pressure field over regular grid via interpolation')
            Pmat = griddata(coords, P, Xgrid, method='nearest', fill_value=0)
        
    # Return outputs
    return coords_per_dim, Pmat, fcode


def compute_working_distance(h_headbar, nrots, h_per_rot=0.35):
    '''
    Compute transducer working distance given a headbar thickness and number of rotations

    :param h_headbar: headbar thickness (mm)
    :param nrots: number of rotations
    :param h_per_rot: thickness gain per rotation (mm), default is 0.35 mm
    :return: working distance (mm)
    '''
    return h_headbar + nrots * h_per_rot


def extract_relative_amp(mapping_fpath, target, dimkey='Z', offset=0, plot=False, fs=12):
    ''' 
    Extract relative acoustic pressure amplitude at specified location
    along specified dimension, across focus.
    
    :param mapping_fpath: mapping file path
    :param target: target location along specified dimension (in mm)
    :param dimkey: dimension key (default: 'Z')
    :param offset: offset to apply to coordinates vector before interpolation, in mm (default: 0)
    :param plot: whether to plot the extracted profile (default: True)
    :param fs: font size (default: 12)
    :return: relative acoustic pressure amplitude w.r.t. peak pressure at target location
    '''
    # Parse mapping file name
    params = parse_mapping_file(mapping_fpath)

    # Load mapping data
    coords_per_dim, Pmat, _ = load_mapping_data(mapping_fpath)

    # Extract profile at focus along specified dimension
    logger.info(f'extracting pressure profile along {dimkey} dimension at focus')
    coords = coords_per_dim[dimkey] * 1e-3 + offset
    Pvec = Pmat[get_mux_slice(idxmax(Pmat), dimkey)]

    # Interpolate pressure at specific value
    logger.info(f'interpolating pressure profile at {dimkey} = {target:.2f} mm')
    Ptarget = np.interp(target, coords, Pvec)

    # Compute relative pressure amplitude w.r.t. peak pressure
    rel_Ptarget = Ptarget / Pvec.max()

    # Log result
    logger.info(f'pressure at target = {Ptarget:.2f} MPa ({rel_Ptarget * 1e2:.0f}% of max)')

    # Plot profile and interpolated value if requested
    if plot:
        fig, ax = plt.subplots()
        sns.despine(fig=fig)
        ax.set_xlabel(f'{dimkey} (mm)', fontsize=fs)
        ax.set_ylabel('Pressure (MPa)', fontsize=fs)
        ax.plot(coords, Pvec)
        ax.scatter(target, Ptarget, marker='o', color='k')
        ax.axvline(target, ls='--', color='k')
        ax.axhline(Ptarget, ls='--', color='k')
        ax.text(
            target + 0.1, Pvec.min(), f'{target:.2f} mm', 
            fontsize=fs, ha='left', va='bottom')
        ax.text(
            coords.min(), Ptarget + 0.02 * (Pvec.max() - Pvec.min()), 
            f'{Ptarget:.2f} MPa ({rel_Ptarget * 1e2:.1f}%)', fontsize=fs, ha='left', va='bottom')
        ax.set_title('transducer_' + params['transducer_id'], fontsize=fs + 2)

    # Return relative pressure amplitude
    return rel_Ptarget


def compare_calibration_and_mapping_params(calib_fpath, mapping_fpath):
    '''
    Parse parameters from transducer calibration and mapping files, compare them, 
    and raise errors if inconsistencies are found.

    :param calib_fpath: full path to a transducer calibration Excel file
    :param mapping_fpath: full path to a mapping results CSV file
    :return: Series of common parameters (if no error was raised)
    '''
    calib_params = parse_transducer_calibration_file(calib_fpath)
    mapping_params = parse_mapping_file(mapping_fpath)

    # Assemble calibration and mapping parameters into a single dataframe
    params = pd.DataFrame({
        'calib': calib_params,
        'mapping': mapping_params
    })

    # Remove rows with missing calibration parameters
    params = params[~params['calib'].isna()]

    # Check for missing mapping parameters
    is_missing = params[params['mapping'].isna()].index.values
    if len(is_missing) > 0:
        raise ValueError('missing mapping parameters:', is_missing)

    # Check for differing parameters between calibration and mapping files
    isdiff = params['calib'] != params['mapping']
    diff_params = params[isdiff]
    if len(diff_params) > 0:
        raise ValueError(
            f'found differing parameters between transducer calibration and mapping files:\n{diff_params}')

    # Return
    return params['calib'].rename('params')