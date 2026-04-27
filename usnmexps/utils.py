# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-03-18 14:42:17
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-06-13 11:26:31

import os
from functools import wraps
import numpy as np
import pandas as pd
import openpyxl
import json
from scipy.signal import butter, filtfilt
from scipy.spatial import distance_matrix
import networkx as nx
from instrulink.logger import logger
from datetime import datetime

from .constants import *
from .si_utils import si_format
from .config import CHECK_ENV

class EnvironmentError(Exception):
    ''' Custom exception class for anaconda environment '''
    pass


class CalibrationError(Exception):
    ''' Custom exception class for calibration proecdures '''
    pass


def check_conda_env():
    ''' Check that the correct anaconda environment is activated '''
    if CHECK_ENV:
        env = os.environ['CONDA_DEFAULT_ENV']
        if env != ENV_NAME:
            raise EnvironmentError(
                f'Wrong conda environment: {env}. Use "conda activate {ENV_NAME}"')
    else:
        logger.warning('not checking anaconda environment')
    

# Path to states log file
STATESLOGFPATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stateslog.json')


def today():
    ''' Return formatted today's date '''
    return datetime.now().strftime(DATE_FORMAT)


def rmse(x1, x2, axis=None):
    ''' Compute the root mean square error between two arrays '''
    return np.sqrt(((x1 - x2) ** 2).mean(axis=axis))


def rescale(x, xmin, xmax):
    '''
    Rescale a vector linearly to a new range
    
    :param x: vector
    :param xmin: new minimum value
    :param xmax: new maximum value
    :return: rescaled vector
    '''
    # Rescale to (0,1)
    dx = x.max() - x.min()
    xnorm = (x - x.min()) / dx
    # Rescale to (xmin, xmax)
    return xnorm * (xmax - xmin) + xmin


def apply_rolling_window(x, w, func=None, warn_oversize=True):
    '''
    Generate a rolling window over an array an apply a specific function to the result.
    Defaults to a moving average.
    
    :param x: input array
    :param w: window size (number of array samples used to apply the function)
    :param func (optional): function to apply to the rolling window result
    :return: output array of equal size to the input array, with the rolling window and function applied.
    '''
    # If input is a pandas DataFrame, apply function to each column
    if isinstance(x, pd.DataFrame):
        return x.apply(lambda col: apply_rolling_window(col, w, func=func, warn_oversize=warn_oversize))
    
    # If more than 1 dimension -> reshape to 2D, apply on each row, and reshape back to original shape
    if x.ndim > 1:
        dims = x.shape
        x = x.reshape(-1, dims[-1])
        x = np.array([apply_rolling_window(xx, w, func=func, warn_oversize=False) for xx in x])
        return x.reshape(*dims)
    # Check that window size is valid
    if w % 2 == 0:
        raise ValueError('window size must be an odd number')
    if w > x.size and warn_oversize:
        logger.warning(f'window size ({w}) is larger than array length ({x.size})')
    # If function not provided, apply mean by default
    if func is None:
        func = lambda x: x.mean()
    # Pad input array on both sides
    x = np.pad(x, w // 2, mode='symmetric')
    # Generate rolling window over array
    roll = pd.Series(x).rolling(w, center=True)
    # Apply function over rolling window object, drop NaNs and extract output array 
    return func(roll).dropna().values

def bounds(x, **kwargs):
    '''Get the bounds of an array ''' 
    return np.array([x.min(**kwargs), x.max(**kwargs)])

def nan_like(x):
    ''' Create array of identicial shape as input, filled with nan values '''
    xnan = np.zeros_like(x)
    xnan[:] = np.nan
    return xnan

def gauss1D(x, x0, sigmax):
    ''' 1D gaussian '''
    return np.exp(-((x - x0)**2 / sigmax**2))

def gauss2D(x, y, x0, y0, sigmax, sigmay):
    ''' 2D gaussian '''
    return np.exp(-((x - x0)**2 / sigmax**2 + (y - y0)**2 / sigmay**2))

def gauss3D(x, y, z, x0, y0, z0, sigmax, sigmay, sigmaz):
    ''' 3D gaussian '''
    return np.exp(-((x - x0)**2 / sigmax**2 + (y - y0)**2 / sigmay**2 + (z - z0)**2 / sigmaz**2))


def get_gauss_params(x, rel_x0, rel_sigma):
    ''' Get absolute gaussian parameters given relative parameters and coordinates vector '''
    xmin, xmax = bounds(x)
    dx = xmax - xmin
    x0 = xmin + rel_x0 * dx
    if dx == 0:
        sigma = 1
    else:
        sigma = rel_sigma * dx
    return x0, sigma


def make_gauss(x=None, y=None, z=None, rel_x0=0.5, rel_y0=0.5, rel_z0=0.5, rel_sigmax=0.3, rel_sigmay=0.3, rel_sigmaz=0.3, A=1):
    '''
    Generate a gaussian of appropriate dimensionality peaking within the defined
    set of coordinates vectors
    
    '''
    params = []
    if x is not None:
        params.append(get_gauss_params(x, rel_x0, rel_sigmax))
    if y is not None:
        params.append(get_gauss_params(y, rel_y0, rel_sigmay))
    if z is not None:
        params.append(get_gauss_params(z, rel_z0, rel_sigmaz))
    ndims = len(params)
    if ndims == 0:
        raise ValueError('at least 1 vector must be provided')
    orgs, sigmas = zip(*params)
    if ndims == 1:
        return lambda xx, yy, zz: A * gauss1D(xx, *orgs, *sigmas)
    elif ndims == 2:
        return lambda xx, yy, zz: A * gauss2D(xx, yy, *orgs, *sigmas)
    elif ndims == 3:
        return lambda xx, yy, zz: A * gauss3D(xx, yy, zz, *orgs, *sigmas)


def eval_on_grid(func, p0, x=None, y=None, z=None):
    ''' Evaluate function along XYZ grid '''
    # Gather provided evaluation vectors
    vecs = []
    if x is not None:
        vecs.append(x + p0[0])
    if y is not None:
        vecs.append(y + p0[1])
    if z is not None:
        vecs.append(z + p0[2])
    # Compute grid dimensions
    grid_dims = [item.size for item in vecs]
    # Generate meshgrid from these vectors
    mesh = np.meshgrid(*vecs, indexing='ij')
    # Serialize grid
    smesh = np.array([item.ravel() for item in mesh])
    # Add missing dimensions (if any) to recover 3D
    placeholder = np.zeros(smesh.shape[1])
    idx = None
    if x is None:
        idx = 0
    elif y is None:
        idx = 1
    elif z is None:
        idx = 2
    if idx is not None:
        smesh = np.insert(smesh, [idx], placeholder, axis=0)
    # Evaluate function on serialized meshgrid
    out = func(*smesh)
    # Reshape output to match grid dimensions 
    out = out.reshape(tuple(grid_dims))
    # Return
    return out


def scan_proof(func):
    '''
    Wrapper to make an evaluation function "scan-proof", i.e. making sure
    that it returns a "continue" flag along with its nominal output. 
    '''
    @wraps(func)
    def scan_proof_func(_, *args, **kwargs):
        return func(*args, **kwargs), True
    return scan_proof_func


def redraw(fig):
    ''' Redraw matplotlib figure '''
    fig.canvas.draw()
    fig.canvas.flush_events()


def is_within(x, bounds):
    return np.logical_and(x >= bounds[0], x <= bounds[1])


def project_onto_plane(p, n, org):
    '''
    Project a point onto a plane
    
    :param p: point
    :param n: normal vector to the plane
    :param org: plane origin point
    '''
    # Compute vector from origin to point of interest
    v = p - org
    nnorm = n / np.linalg.norm(n)
    # Compute dot product of that vector with the unit normal vector n, i.e.
    # scalar distance from point to plane along the normal
    dist = np.dot(v, nnorm)
    # Multiply the unit normal vector by the distance, and subtract that vector
    # from original point
    proj = p - dist * nnorm
    # Return projection
    return proj


def linvec(xstart, xend, dx):
    '''
    Generate linearly distributed vector of values between a start and end values, adapting
    step size to ensure end value is included in output. 

    :param xstart: start value
    :param xend: end value
    :param dx: step size
    :return: linearly distributed vector of values
    '''
    # Compute required number of samples   
    n = int((xend - xstart) / dx) + 1
    # Generate linearly distributed vector
    return np.linspace(xstart, xend, n)


def refine_vec(x, ref_factor):
    ''' 
    Refine a vector by a specific factor
    
    :param x: 1D vector
    :param ref_factor: refinement factor
    :return: rescaled vector (centered around zero)
    '''
    # Compute vector variation range and divide by refinement factor
    dx = np.ptp(x) / ref_factor
    # Return rescaled vector of identical size, centered around 0
    return np.linspace(0, dx, x.size) - dx / 2


def remove_almost_duplicates(arr, tol=1e-8):
    ''' 
    Remove "almost-duplicate" values from a 1D array, i.e. values that are
    within a specified small tolerance of each other.

    :param arr: 1D array of values
    :param tol: tolerance for considering values as "almost-duplicate"
    :return: array with almost-duplicate values removed
    '''
    # Sort array to make nearby values adjacent
    arr = np.sort(arr)
    # Initialize "true uniques" list with the first array value
    true_uniques = [arr[0]]
    # Iterate through the rest of the array
    for val in arr[1:]:
        # If value outside tolerance from last "true unique", add it 
        # to list of "true uniques"
        if val - true_uniques[-1] > tol:
            true_uniques.append(val)
    # Return sorted "true uniques" as array
    return np.array(true_uniques)


def get_closest(v, x):
    ''' Get the element of v closest to x '''
    return v[np.argmin(np.abs(v - x))]


def filter_signal(y, fs, fc, order=2, verbose=True):
    '''
    Apply zero-phase filter to signal
    
    :param y: signal array
    :param fs: sampling frequency (Hz)
    :param fc: tuple of cutoff frequencies (Hz)
    :param order: filter order
    '''
    if isinstance(y, pd.Series):
        yf = filter_signal(y.values, fs, fc, order=order)
        return pd.Series(data=yf, index=y.index)
    fc = np.asarray(fc)
    # Determine Butterworth type and cutoff
    btype = 'band'
    if fc[0] == 0.:
        btype = 'low'
        fc = fc[1]
    elif fc[1] == np.inf:
        btype = 'high'    
        fc = fc[0]
    logfunc = {True: logger.info, False: logger.debug}[verbose]
    logfunc(f'{btype}-pass filtering signal (cutoff = {si_format(fc, 3)}Hz)')
    # Determine Nyquist frequency
    nyq = fs / 2
    # Calculate Butterworth filter coefficients
    b, a = butter(order, fc / nyq, btype=btype)
    # Filter signal forward and backward (to ensure zero-phase) and return
    return filtfilt(b, a, y)


def pressure_to_intensity(p, rho=1046.0, c=1546.3):
    '''
    Return the spatial peak, pulse average acoustic intensity (ISPPA)
    associated with the specified pressure amplitude.
    
    Default values of dennsity and speed of sound are taken from the
    IT'IS foundation database for brain tissue. 
    
    :param p: pressure amplitude (Pa)
    :param rho: medium density (kg/m3)
    :param c: speed of sound in medium (m/s)
    :return: spatial peak, pulse average acoustic intensity (W/m2)
    '''
    if rho <= 0:
        raise ValueError(f'Invalid medium density: {rho} kg/m3 (must be strictly positive)')
    if c <= 0:
        raise ValueError(f'Invalid medium speed of sound: {c} m/s (must be strictly positive)')
    return p**2 / (2 * rho * c)


def intensity_to_pressure(I, rho=1046.0, c=1546.3):
    '''
    Return the pressure amplitude (in Pa) associated with the specified
    spatial peak, pulse average acoustic intensity (ISPPA).
    
    Default values of dennsity and speed of sound are taken from the
    IT'IS foundation database for brain tissue. 
    
    :param I: Isppa (W/m2)
    :param rho: medium density (kg/m3)
    :param c: speed of sound in medium (m/s)
    :return: pressure amplitude (Pa)
    '''
    if np.any(np.asarray(I)) < 0:
        raise ValueError(f'Invalid intensity value: {I} W/m2 (must be positive)')
    if rho <= 0:
        raise ValueError(f'Invalid medium density: {rho} kg/m3 (must be strictly positive)')
    if c <= 0:
        raise ValueError(f'Invalid medium speed of sound: {c} m/s (must be strictly positive)')
    return np.sqrt(I * 2 * rho * c)


def get_states_log():
    ''' Return states log file content as nested dictionary '''
    # If log file does not exist, return empty dictionary
    if not os.path.isfile(STATESLOGFPATH):
        logger.warning('could not find states log file')
        return {}

    # Load log file content as dictionary
    with open(STATESLOGFPATH, 'r') as f:
        d = json.load(f)

    # Return dictionary
    return d


def update_states_log(d):
    ''' Update states log file content with provided dictionary '''
    # Save dictionary to log file
    with open(STATESLOGFPATH, 'w') as f:
        json.dump(d, f, indent=4)


def get_last_files_dict():
    ''' Return dictionary of last open data files from states log file '''
    # Get log file dictionary
    d = get_states_log()

    # If log file does not contain "last_files" key, return empty dictionary
    if 'last_files' not in d.keys():
        logger.warning('log file does not contain "last_files" key')
        return {}

    # Extract and return "last_files" key
    return d['last_files']
    

def get_last_file(kind):
    '''
    Fetch the last open data file of a specific type
    
    :param kind: type of data file ("mapping", "calibration")
    :return: path to last open data file of specified type
    '''
    # Get dictionary of last open data files
    d = get_last_files_dict()

    # If empty dictionary, return None
    if not d:
        return None

    # If log file does not contain last file info of that type, return None
    if kind not in d.keys():
        logger.info(f'log file does not contain last {kind} file info')
        return None

    # Extract full path to last data file
    fpath = d[kind]

    # If data file does not exist, return None
    if not os.path.isfile(fpath):
        logger.warning(f'could not find last {kind} file at address: "{fpath}"')
        return None
    
    # Return path to last data file
    return fpath


def update_last_file(kind, fpath):
    '''
    Update states log file with last open data file of a specific type
    
    :param kind: type of data file ("mapping", "calibration")
    :param fpath: path to last open data file of specified type
    '''
    # Get states log file content
    d = get_states_log()

    # Extract "last_files" key from log file content
    if 'last_files' not in d.keys():
        d['last_files'] = {}
    
    # Update "last_files" dictionary with new file path
    d['last_files'][kind] = fpath

    # Save dictionary to log file
    update_states_log(d)


def get_last_position():
    ''' Return last position from states log file '''
    # Get log file dictionary
    d = get_states_log()

    # If log file does not contain "last_position" key, return None
    if 'last_position' not in d.keys():
        logger.warning('log file does not contain "last_position" key')
        return None
    
    # Return last position as numpy array
    return np.array(d['last_position'])


def update_last_position(position):
    ''' Update states log file with last position '''
    # Convert position to list if necessary
    if isinstance(position, np.ndarray):
        position = position.tolist()

    # Get states log file content
    d = get_states_log()

    # Update "last_position" key with new position
    d['last_position'] = position

    # Save dictionary to log file
    update_states_log(d)


def idxmax(data, n=1):
    '''
    Find index of maximum value(s) in multi-dimensional array
    
    :param data: multi-dimensional array
    :param n: number of maximum values to return (default: 1)
    :return: index of maximum value(s), with same number of dimensions as input array
    '''
    imax = np.argpartition(data.ravel(), -n)[-n:][::-1]
    idx = np.unravel_index(imax, data.shape)
    if n == 1:
        return tuple(int(i) for i in idx)
    else:
        return idx


def get_mux_slice(mux, iax):
    '''
    Construct a slice object that can be used to access the 1D content of a 
    multimensional array along a specific axis, at a specific location along the other axes.
    
    :param mux: projection index vector (matching matrix dimenions)
    :param iax: index (or "XYZ" identifier string) of axis from which to extract content
    '''
    # Determine data dimensions
    ndims = len(mux)

    # If projection axis is specified as string, convert to index
    if isinstance(iax, str):
        dimsstr = 'XYZ'[:ndims]
        iax = iax.upper()
        if iax not in dimsstr:
            raise ValueError(f'invalid projection axis: {iax}')
        iax = dimsstr.index(iax)
    
    # Check that projection axis is valid
    if not isinstance(iax, int):
        raise TypeError(f'invalid projection axis type: {type(iax)}')
    if iax >= ndims:
        raise ValueError(f'projection axis ({iax}) outside of data dimensions ({ndims})')
    
    # Adapt projection index to specified axis
    mux = list(mux)
    mux[iax] = slice(None)

    # Return as tuple 
    return tuple(mux)


def cartesian_to_cylindrical(XYZ):
    '''
    Convert cartesian coordinates to cylindrical coordinates
    
    :param XYZ: vector/dictionary of (X, Y, Z) coordinates
    :return: vector/dictionary of (R, Z, Θ) coordinates
    '''
    is_dict = isinstance(XYZ, dict)
    X, Y, Z = [XYZ[k] for k in 'XYZ'] if is_dict else XYZ.T
    R = np.sqrt(X**2 + Y**2)
    Θ = np.arctan2(Y, X)
    out = np.column_stack((R, Z, Θ))
    if is_dict:
        out = {k: v for k, v in zip('RZΘ', out.T)}
        if isinstance(X, float):
            out = {k: v[0] for k, v in out.items()}
        return out
    else:
        if XYZ.ndim == 1:
            return out[0]
        return out


def cylindrical_to_cartesian(RZΘ):
    '''
    Convert cylindrical coordinates to cartesian coordinates
    
    :param RZΘ: vector/dictionary of (R, Z, Θ)
    :return: vector/dictionary of (X, Y, Z) coordinates
    '''
    is_dict = isinstance(RZΘ, dict)
    R, Z, Θ = [RZΘ[k] for k in 'RZΘ'] if is_dict else RZΘ.T
    X = R * np.cos(Θ)
    Y = R * np.sin(Θ)
    out = np.column_stack((X, Y, Z))
    if is_dict:
        out = {k: v for k, v in zip('XYZ', out.T)}
        if isinstance(R, float):
            out = {k: v[0] for k, v in out.items()}
        return out
    else:
        if RZΘ.ndim == 1:
            return out[0]
        return out


def find_shortest_path(points, cyclic=False):
    '''
    Approximates the shortest path visiting all points
    
    :param points: an (n,3) array of XYZ coordinates.
    :param cyclic: whether to implement a cyclic path (i.e., a path 
        returning to the starting point at the end) or not
    :return: index array indicating the order of points to visit in the shortest path, 
        and the total path length 
    '''
    # Number of points
    n = len(points)
    
    # Compute pairwise Euclidean distances
    dist_matrix = distance_matrix(points, points)

    # Create a complete weighted graph
    G = nx.complete_graph(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                G[i][j]['weight'] = dist_matrix[i, j]

    # If cyclic path request, approximate TSP solution using
    # Minimum Spanning Tree + Eulerian Circuit
    if cyclic:
        xyzpath = nx.approximation.traveling_salesman_problem(G, cycle=True)
    # Othwerise, approximate TSP solution using Minimum Spanning Tree + DFS
    # starting from first node
    else:
        mst = nx.minimum_spanning_tree(G)

        # Extract an approximate Hamiltonian path using Depth-First Search (DFS)
        start_node = 0  # Can be changed if needed
        xyzpath = list(nx.dfs_preorder_nodes(mst, source=start_node))

    # Compute total path length
    nstop = n if cyclic else n - 1
    xyzdist = sum(dist_matrix[xyzpath[i], xyzpath[i + 1]] for i in range(nstop))
    logger.debug(f'total path length = {xyzdist:.2f}')
    
    # Return points order in the path, and total path length
    return np.array(xyzpath), xyzdist


def flatten_dict(d):
    '''
    Recursively flattens a nested dictionary.

    :param d: the dictionary to flatten.
    :return: flattened dictionary.
    '''
    items = {}
    for k, v in d.items():
        if isinstance(v, dict):
            items.update(flatten_dict(v))
        else:
            items[k] = v
    return items


def save_to_excel_file(df, fpath, sheet_name='Sheet1'):
    '''
    Save dataframe to excel file and fit columns width to content
    
    :param df: dataframe to save
    :param fpath: path to excel file
    :param sheet_name: name of the sheet
    '''
    # Save DataFrame to Excel with openpyxl as the engine
    with pd.ExcelWriter(fpath, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        worksheet = writer.sheets[sheet_name]  # Access the sheet
    
        # Auto-adjust column widths
        for col_num, col_name in enumerate(df.columns, start=1):
            col_letter = openpyxl.utils.get_column_letter(col_num)
            width = max(df[col_name].astype(str).map(len).max(), len(col_name)) - 1
            worksheet.column_dimensions[col_letter].width = width


def split_name_and_unit(s):
    mo = re.match(NAME_AND_UNIT_RGXP, s)
    if mo is None:
        raise ValueError(f'input string "{s}" does not match "name (unit)" pattern')
    return mo.groups()