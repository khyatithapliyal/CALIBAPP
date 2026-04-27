# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-03-07 17:11:50
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-06-12 15:33:46

import numpy as np
import re
import os

from .config import CALIBRATION_DATA_FOLDER

# Environment
ENV_NAME = 'usnmexps'

# Unknown key
UNKNOWN = '???'

# Date format
DATE_FORMAT = '%Y.%m.%d'  # yyyy.mm.dd

# Regular expressions
INT_REGEXP = r'\d+'
FLOAT_REGEXP = r'[+-]?\d*[.]?\d+'
DATE_REGEXP = r'(\d{4}).(0[1-9]|1[0-2]).(0[1-9]|[12][0-9]|3[01])'
DENSITY_RGXP = f'nX({INT_REGEXP})Y({INT_REGEXP})Z({INT_REGEXP})'
VOLUME_RGXP = f'DX({FLOAT_REGEXP})Y({FLOAT_REGEXP})Z({FLOAT_REGEXP})mm'
TILT_RGXP = f'θY({FLOAT_REGEXP})deg'
FREQ_RGXP = f'({FLOAT_REGEXP})MHz'
NAME_AND_UNIT_RGXP = r'([a-zA-Z]+) \((.+)\)'

EXCEL_FILTER = 'Excel Files (*.xlsx)'
CSV_FILTER = 'CSV Files (*.csv)'
HYDROPHONE_CALIBRATION_FILE_PATTERN = re.compile('(.+)_(.+)-(.+).xlsx')
TRANSDUCER_ID_PATTERN = r'[A-z0-9-#]+'
TRANSDUCER_CALIBRATION_FILE_PATTERN = re.compile(f'({TRANSDUCER_ID_PATTERN})_([A-z0-9]+)_{FREQ_RGXP}.xlsx')
AMPLIFIER_CALIBRATION_FILE_PATTERN = re.compile(r'(.+)_(\-[0-9.]+)dB_([0-9.]+)MHz.xlsx')
MAPPING_FILE_PATTERN = f'^({TRANSDUCER_ID_PATTERN})_([A-z0-9]+)_{FREQ_RGXP}_([A-z]*)map_([A-z0-9\-]*){DATE_REGEXP}_{TILT_RGXP}_{VOLUME_RGXP}_{DENSITY_RGXP}.*.csv$'
CALIBRATION_OUTPUT_PREFIX_PATTERN = f'{DATE_REGEXP} #({INT_REGEXP})'
CALIBRATION_POUT_PATTERN = f'^Pout {CALIBRATION_OUTPUT_PREFIX_PATTERN} \(MPa\)$'
CALIBRATION_VOUT_PATTERN = f'^Vout {CALIBRATION_OUTPUT_PREFIX_PATTERN} \(Vpp\)$'
CALIBRATION_CPL_PATTERN = f'^CPL(FWD|REV) {CALIBRATION_OUTPUT_PREFIX_PATTERN} \(Vpp\)$'

# Data files and folders
HYDROPHONES_FOLDER = os.path.join(CALIBRATION_DATA_FOLDER, 'hydrophones')
TRANSDUCERS_FOLDER = os.path.join(CALIBRATION_DATA_FOLDER, 'transducers')
AMPLIFIERS_FOLDER = os.path.join(CALIBRATION_DATA_FOLDER, 'amplifiers')
FOCAL_DISTANCES_FPATH = os.path.join(TRANSDUCERS_FOLDER, 'focal_distances.xlsx')
PROTOCOL_TEMPLATE = 'protocol_P_DC.xlsx'

# Column names in Excel files
F_KEY = 'Freq (MHz)'
P_KEY = 'P (MPa)'
VOUT_KEY = 'Vpp (V)'
CPLRATIO_KEY = 'V(REV) / V(FWD)'
CPLRATIODB_KEY = 'LogV(REV/FWD) (dB)'
GAIN_KEY = 'Gain (dB)'
DC_KEY = 'DC (%)'
PRF_KEY = 'PRF (Hz)'
CONV_KEY = 'Sen (V/Pa)'
DUR_KEY = 'duration (ms)'
TIME_US = 'time (us)'
VIN_KEY = 'Vin (mVpp)'
NCYCLES_PER_PULSE_KEY = '# cycles / pulse'
NCYCLES_PER_BURST_KEY = '# cycles / burst'
SET_KEY = 'set (Y/N)?'
CODE_KEY = 'code'

# Units conversion
MHZ_TO_HZ = 1e6
KHZ_TO_HZ = 1e3
PA_TO_MPA = 1e-6
S_TO_MS = 1e3
S_TO_US = 1e6
MS_TO_S = 1e-3
MV_TO_V = 1e-3
M_TO_MM = 1e3
MM_TO_UM = 1e3
M2_TO_CM2 = 1e4
DEG_TO_RAD = np.pi / 180

# Physical constants
SPEED_OF_SOUND_WATER = 1500.  # Speed of sound in water (m/s)

# Numerical constants
STIM_CH_TRIG = 1  # stimulator channel number for trigger
STIM_CH_SIGNAL = 2  # stimulator channel number for signal of interest
ACQ_CH_TRIG = 1  # scope channel number for acquisition trigger
ACQ_CH_SIGNAL = 2  # scope channel number for signal of interest
ACQ_CH_CPLFWD = 3  # scope channel number for forward coupling signal
ACQ_CH_CPLREV = 4  # scope channel number for reverse coupling signal
CALIB_CONDS = ('free field', 'glass 1 layer', 'glass 2 layers')  # possible calibration conditions
MAPPING_MODES = ('brute', 'adaptive')  # possible mapping modes
DEFAULT_CARRIER_FREQ = 2.1  # Default carrier frequency (MHz)
FSWEEP_FREQS = np.linspace(1.9, 2.2, 21)  # frequencies used for frequency sweep range (MHz)
MIN_NSAMPLES_PER_CYCLE = 20  # minimum number of samples per cycle to get an accurate waveform depiction
THETAY_SCAN = 15.  # XZ incident angle of scanning system (degrees)
VBASIS = [1, 1, -1]  # Vector base of scanning system
DELTA_SCAN = {'X': 3000., 'Y': 3000, 'Z': 8000}  # scanning extent per axis (um)
DELTA_FOCUS = {'X': 2000, 'Y': 2000, 'Z': 3000}  # focus search extent per axis (um)
DEFAULT_MOVEAWAY_DIST = 200.  # default distance to move away from reference plane (um)
DEFAULT_MOVEUPDOWN_DIST = 1000.  # default distance to move up/down (um)
RES_SCAN = 100.  # scanning resolution (um)
RES_SCAN_SPARSE = 500.  # sparse scanning resolution (um) 
NPERAX_SEARCH = 15  # number of sampled coordinates per axis for focus search
NITERS_SEARCH = 3  # number of cross-search iterations to find focus
MAP_SCAN_ORDER = 'XYZ'  # order of scanned dimensions in grid mapping procedure
FOCUS_CROSS_SCAN_ORDER = 'ZYX'  # order of scanned dimensions in cross-scan procedure used to find focus 
SCAN_ENDMODES = (  # possible end modes for scanning procedures
    'gotoref',  # go back to reference (i.e. pre-scan) location
    'gotomax',  # go to location of maximum value
    'gotomaxproj'  # go to location of maximum value projected onto plane of pre-scan location
)  
NCYCLES_SCAN = 4  # number of cycles per burst for scanning procedures (focus search, mapping)
NCYCLES_CALIB = 50  # number of cycles per burst for transducer calibration procedures (I/O curve, frequency sweeps)
REF_VPP_CALIB = 0.2  # reference peak-to-peak input voltage for calibrations (V)
INPUT_VPPS_CALIB = np.arange(0, 3.01, .05)  # input driving voltages (Vpp)
NPERCOND_CALIB = 2  # number of acquisitions to average from for each condition for calibration curve
MAX_CV_CALIB = 0.05  # max allowed coefficient of variation for across acquisitions of the same condition
DEFAULT_PRF = 100.  # Default pulse repetition frequency (Hz)
OSC_ACQ_INTERVAL = .2  # Interval between oscilloscope acquisitions (s)
OSC_ACQ_NPOINTS = 12000  # Number of points per oscilloscope acquisition
OSC_ACQ_NSWEEPS = 1  # Number of sweeps per oscilloscope acquisition
NAVG_PENV = 101  # size of moving average applied to pressure envelope
NTRIM_PENV = 500  # number of samples to trim off of pressure envelope edges
REL_MAX_START = 0.5  # relative amplitude w.r.t. peak at which to consider the waveform start
MIN_LAT = 1e-6  # minimal waveform signal letency (s)
MAX_LAT = 10e-6  # maximal waveform signal letency (s)
MAX_CPL_DEV_DB = 2  # maximal allowed deviation in coupling ratio (dB) to consider transducer efficiency unchanged
LARGE_SCAN_THR_NPOINTS = 2000  # threshold number of points above which a scan is considered "large"
LARGE_SCAN_REFRESH_EVERY = 100  # scan graph refresh rate for large number of points

EXP_DUR = 200.  # burst duration for experiments (ms)
EXP_SR = 30.02  # microscope sampling rate for experiments (Hz)
EXP_NFRAMES = 601  # number of acquisition frames for experiments
EXP_STIMDELAY = 2.8  # initial stimulus delay for each acquisition during experiments (s)
EXP_NACQS = 16  # number of acquisitions per condition for experiments
EXP_VIDEO_DUR = 10. # duration of video acquisition per trial for experiments (s)

NBYTES_INT16 = 2  # number of bytes taken by an 16-bit integer (int16) on disk
BYTES_TO_GBYTES = 2**-30  # bytes to gigabytes conversion constant

# Theoretical amplifier gains (dB) 
AMPGAINS = {
    'null': 0,  # "null" (i.e. no) amplifier    
    'MC-LZY-22': 40., # LZY-22 amplifier
}