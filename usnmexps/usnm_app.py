# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2025-02-12 23:50:49
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-03-19 15:22:13

# External packages
import numpy as np
import pandas as pd
import shutil
import time
from pyvisa.errors import VisaIOError
from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox
from PyQt6.QtGui import QColor
from instrulink import logger, grab_generator

# Internal modules
from .constants import *
from .virtual_instrument import VirtualInstrument
from .virtual_bench import *
from .comm_utils import connect_to_matlab
from .input_widgets import *
from .config import PROTOCOLS_FOLDER
from .utils import get_last_file


from instrulink import grab_oscilloscope  # Add this import
from .efficiency_monitor import EfficiencyMonitor, get_coupling_reference_from_calibration
from .calibrators import Calibrator


class USNMApp(VirtualBenchApp):
    ''' USNM experiments app '''

    PROTOCOL_KEYS = [
        P_KEY,
        DC_KEY,
        PRF_KEY,
        DUR_KEY,
    ]

    SET_COLOR = QColor(100, 255, 100, 100)  # Light green (RGBA with transparency)
    SKIP_COLOR = QColor(255, 255, 0, 100)  # Light yellow (RGBA with transparency

    # Actions dictionary
    ACTIONS_DICT = {
        'shuffle': 'shuffle',
        'next': 'next',
        'reset': 'reset',
        'acquire': 'acquire',
        'skip': 'skip',
        'clear': 'clear',
        'burst_check': 'burst check',
    }

    def __init__(self, **kwargs):
        '''
        Initialize app
        
        :param testmode: whether to run the app in test mode (no instrument needed)
        '''
        # Instruments list
        instruments = [
            VirtualInstrument('wg', 'waveform generator', lambda: grab_generator(key='rigol')),
            VirtualInstrument('si', 'Matlab ScanImage', connect_to_matlab),
            VirtualInstrument('osc', 'oscilloscope', lambda: grab_oscilloscope(key='bk')),
        ]

        # Initialize default attributes
        self.fMHz = None
        self.interp_data = None

        # Initialize parent class
        super().__init__(instruments, **kwargs)
        # Efficiency monitoring
        self.efficiency_monitor = EfficiencyMonitor()

        # Set default input files
        for key in ['transducer', 'protocol']:
            self.update_input_file(key, get_last_file(key))
    
    # ------------------- LAYOUT -------------------
    
    def set_layout(self):
        ''' Set layout '''
        body = QHBoxLayout()

        # Left column: instruments, acquisition parameters and log
        leftcol = QVBoxLayout()
        leftcol.addWidget(self.instruments_panel())
        self.acq_params_dict, acq_params_table = USNM_ACQ_PARAMS.render()
        leftcol.addWidget(acq_params_table)
        leftcol.addWidget(self.log_panel())        
        body.addLayout(leftcol)

        # Right column: input files, protocol table & action buttons
        rightcol = QVBoxLayout()
        rightcol.addWidget(self.input_files_panel())
        rightcol.addWidget(self.protocol_panel())
        rightcol.addWidget(self.actions_panel(self.ACTIONS_DICT))
        
        body.addLayout(rightcol)
        self.setLayout(body)
    
    def input_files_panel(self):
        ''' Input files selection panel '''
        return super().input_files_panel({
            'transducer': (EXCEL_FILTER, TRANSDUCERS_FOLDER, False),
            'protocol': (EXCEL_FILTER, PROTOCOLS_FOLDER, False)})
    

    def protocol_panel(self):
        ''' Return protocol panel '''
        group = QGroupBox('Protocol')
        self.protocol_table = self.df_to_table(None)
        self.protocol_table.setMinimumWidth(670)
        self.protocol_table.setMinimumHeight(400)
        layout = QVBoxLayout()
        layout.addWidget(self.protocol_table)
        group.setLayout(layout)
        return group

    # ------------------- CALLBACKS -------------------
    
    def register_callbacks(self):
        ''' Register callbacks '''
        # Inputs change
        for k, w in self.acq_params_dict.items():
            if isinstance(w, QLineEdit):
                w.textChanged.connect(self.on_acqinput_update)
            else:
                w.valueChanged.connect(self.on_acqinput_update)
    
        # Input files change
        self.fwidget['transducer'].textChanged.connect(self.on_transducer_file_change)
        self.fwidget['protocol'].textChanged.connect(self.on_protocol_file_change)

        # Actions clicks
        self.connect_actions()
    
    @property
    def protocol_data(self):
        ''' Get protocol data '''
        df = self.fdata['protocol']
        if df is None:
            return None
        return df[[*self.PROTOCOL_KEYS, CODE_KEY]]
    
    def on_transducer_file_change(self, fname):
        ''' Callback when transducer calibration file path has changed '''
        # If no file selected, return
        if self.fpaths['transducer'] is None:
            return

        try:
            # Parse transducer file
            params = parse_transducer_calibration_file(fname)
            
            # Update fundamental frequency
            self.fMHz = params['freq (MHz)']
            
            logger.info(f'loaded transducer file: {fname}')

            # Get output keys
            outkeys = list(filter(
                lambda k: re.match(CALIBRATION_POUT_PATTERN, k), self.fdata['transducer'].columns)) 

            # Extract mean output pressure profile from last transducer calibration date 
            outs_by_date = parse_outputs_by_date(outkeys)
            last_date = list(outs_by_date.keys())[-1]
            outs = outs_by_date[last_date]
            Pkeys = [k for k in outs if 'MPa' in k]
            Pouts = self.fdata['transducer'][Pkeys]
            Pout = Pouts.mean(axis=1).rename(P_KEY)

            # Store pressure-Vpp interpolation data 
            self.interp_data = pd.concat([self.fdata['transducer'][VIN_KEY], Pout], axis=1).dropna()

            # If protocol data is available, attempt to interpolate input voltages
            # at each pressure level found in the protocol
            if self.protocol_data is not None:
                self.interpVpp(self.protocol_data[P_KEY])

        except ValueError as e:
            logger.error(e)
            self.clear_file_input('transducer')
            self.fMHz = None

        # Setup efficiency monitoring
        self.setup_efficiency_monitoring_from_transducer()

        # Update action buttons status
        self.update_actions()
    
    def interpVpp(self, P):
        '''
        Interpolate input voltage at given pressure

        :param P: pressure value (MPa)
        :return: interpolated input voltage (mVpp)
        '''
        if self.interp_data is None:
            raise ValueError('transducer calibration data not loaded')
        return np.interp(P, self.interp_data[P_KEY], self.interp_data[VIN_KEY])
    
    def setup_efficiency_monitoring_from_transducer(self):
        """
        Load efficiency references from transducer calibration file.
        Call this after transducer file is loaded.
        """
        if self.fdata['transducer'] is None:
            logger.warning('Cannot setup efficiency monitoring: no transducer data')
            return False
        
        try:
            # Get current voltage setting
            if self.protocol_data is None or self.row_idx is None:
                # Use middle of range for reference
                vpp = 0.5
            else:
                row = self.protocol_data.loc[self.row_idx]
                vpp = self.interpVpp(row[P_KEY]) * MV_TO_V
            
            # Get references from calibration
            coupling_db, fwd_v, _ = get_coupling_reference_from_calibration(
                self.fdata['transducer'], vpp)
            
            self.efficiency_monitor.set_references(
                coupling_db=coupling_db,
                fwd_v=fwd_v
            )
            return True
            
        except Exception as e:
            logger.warning(f'Could not setup efficiency monitoring: {e}')
            return False

    def on_protocol_file_change(self):
        ''' Callback upon protocol file change '''
        if self.fpaths['protocol'] is None:
            return
        if self.fdata['protocol'] is None:
            return
        
        # Check that all required columns are present
        try:
            self.fdata['protocol'] = self.fdata['protocol'][self.PROTOCOL_KEYS]
        except KeyError as e:
            raise KeyError(f'protocol file missing required columns: {e}')
        
        # Add file codes column
        self.update_filecodes()

        # Add "order" column to store acquisition order
        self.fdata['protocol']['order'] = np.nan

        # Reset order counter and row index
        self.order_counter = 0
        self.row_idx = 0

        # Update protocol tablef
        if self.protocol_table is not None:
            self.protocol_table.clear()
        if self.protocol_data is not None:
            self.update_protocol_table()

        # If transducer data is available, attempt to interpolate input voltages
        # at each pressure level found in the protocol
        if self.interp_data is not None:
            self.interpVpp(self.protocol_data[P_KEY])
        
        self.update_actions()
    
    def update_actions(self):
        ''' Update action buttons status '''
        # Assemble boolean flags for each required condition
        has_generator = self.instruments['wg'].is_connected()
        has_scanimage = self.instruments['si'].is_connected()
        has_oscilloscope = self.instruments['osc'].is_connected()
        has_transducer_data = self.fMHz is not None
        has_protocol_data = self.protocol_data is not None
        is_protocol_started, is_protocol_completed = False, False 
        if has_protocol_data:
            irow = self.get_next_entry()
            if irow is not None:
                is_protocol_started = irow > 0
                is_protocol_completed = irow == len(self.protocol_data)

        # Initialize dictionary of enabled states for each action, conditioned on available protocol data
        is_enabled = {k: has_protocol_data for k in self.ACTIONS_DICT.keys()}

        # Refine enabled states based on protocol state
        for k in ['clear', 'reset']:
            is_enabled[k] = is_enabled[k] and is_protocol_started
        for k in ['skip', 'next']:
            is_enabled[k] = is_enabled[k] and not is_protocol_completed
        
        # Refine enabled states based on available instruments
        for k in ['reset', 'next', 'acquire']:
            is_enabled[k] = is_enabled[k] and has_generator and has_transducer_data
        
        # Refine enabled states based on ScanImage status
        is_enabled['acquire'] = is_enabled['acquire'] and has_scanimage

        # Burst check: needs generator + oscilloscope + transducer frequency, NOT protocol/ScanImage
        is_enabled['burst_check'] = has_generator and has_oscilloscope and has_transducer_data

        # Update action buttons states
        for k, v in is_enabled.items():
            self.actions_btns[k].setEnabled(v)
    
    def on_acqinput_update(self):
        ''' Callback for acquisition input update '''
        self.update_filecodes()
        self.update_scanimage_acqparams()
        if self.fdata['protocol'] is not None:
            self.update_protocol_table()
            self.update_scanimage_basename(self.protocol_data.loc[self.row_idx, CODE_KEY])
    
    def update_protocol_table(self):
        ''' Update protocol table '''
        self.df_to_table(self.protocol_data, table=self.protocol_table)
        for irow, row in self.fdata['protocol'].iterrows():
            order = row['order']
            if not np.isnan(order):
                color = self.SKIP_COLOR if order == -1 else self.SET_COLOR
                for icol in range(self.protocol_table.columnCount()):
                    self.protocol_table.item(irow, icol).setBackground(color)

    def update_scanimage_basename(self, code):
        ''' Update acquisition file basename in MATLAB ScanImage GUI '''
        logger.info(f'setting basename "{code}" in ScanImage...')
        if not self.testmode and self.instruments['si'].is_connected():
            res = self.instruments['si'].obj.set_basename(code)
            if res != code:
                raise ValueError(f'could not set "{code}" basename in ScanImage')
    
    def update_scanimage_acqparams(self):
        ''' Update acquisition parameters in MATLAB ScanImage GUI '''
        logger.info('setting acquisition parameters in ScanImage...')
        if not self.testmode and self.instruments['si'].is_connected():
            ntot = self.instruments['si'].obj.set_acqparams(self.nperacq, self.nacqs, self.stimdelay)    
            if ntot != self.nperacq * self.nacqs: 
                raise ValueError(f'could not set "{ntot}" frames total in ScanImage')
        
    def get_filecode(self, idx, row):
        '''
        Get the file code corresponding to a specific combination of stimulation
        and acquisition parameters. Used for file saving
        
        :param idx: row index
        :param row: row data
        :return: formatted filecode string
        '''
        return (
            f'{self.session_id}_{self.nperacq}frames_{row[PRF_KEY]}Hz_{row[DUR_KEY]}ms'
            f'_{self.sr:.2f}Hz_{row[P_KEY]:.2f}MPa_{int(row[DC_KEY]):02d}DC-run{idx:02d}')

    def update_filecodes(self):
        ''' Update filecodes in protocol table, if available '''
        if self.fdata['protocol'] is not None:
            self.fdata['protocol'][CODE_KEY] = [
                self.get_filecode(*x) for x in self.fdata['protocol'].iterrows()]
                
    def get_next_entry(self):
        ''' Get the next stimulus in the protocol data that has not been set yet '''
        if self.fdata['protocol'] is None:
            return None
        # Get first index whose "order" column is NaN
        return next(
            (irow for irow, row in self.fdata['protocol'].iterrows() if np.isnan(row['order'])), None)

    def mark_as_set(self, idx):
        ''' Mark a protocol entry as set '''
        self.set_entry_order(idx, self.order_counter)
    
    def mark_as_skipped(self, idx):
        ''' Mark a protocol entry as skipped '''
        self.set_entry_order(idx, -1)

    def set_entry_order(self, idx, value):
        '''
        Update protocol entry status
        
        :param idx: protocol row index
        :param order: acquisition order index
        '''
        # Assign value to appropriate protocol dataframe row in order column
        self.fdata['protocol'].loc[idx, 'order'] = value
        # If value >= 0 (i.e. not skipped but set), increment order counter
        if not np.isnan(value) and value >= 0:
            self.order_counter += 1
        self.update_protocol_table()
    
    def set_stimulus(self, idx):
        '''
        Set the stimulus corresponding to a particular protocol entry on the waveform generator

        :param idx: protocol row index
        '''
        # Get row content and extract code
        row = self.protocol_data.loc[idx].copy()
        code = row.pop(CODE_KEY)
        
        # Add frequency to row 
        row[F_KEY] = self.fMHz
        
        # Extract stimulus parameters and log
        rowstr = []
        for k, v in row.items():
            symbol, unit = k.split(' ')
            if len(unit) > 0:
                unit = unit.replace('(', '').replace(')', '')
            rowstr.append(f'{symbol} = {v} {unit}')
        rowstr = ', '.join(rowstr)
        logger.info(f'setting new stimulus: {rowstr}')
        
        # Initialize is_set flag to False
        is_set_flag = False
        if self.testmode:
            # No problem in test mode -> toggle flag to True
            is_set_flag = True
        else:
            # Set stimulus on function generator (only if object exists within class)
            if self.instruments['wg'].obj is not None:
                try:
                    self.instruments['wg'].obj.set_gated_sine_burst(
                        row[F_KEY] * MHZ_TO_HZ,  # Hz
                        self.interpVpp(row[P_KEY]) * MV_TO_V,  # Vpp
                        row[DUR_KEY] * MS_TO_S,  # s
                        row[PRF_KEY],  # Hz
                        row[DC_KEY])  # %
                    is_set_flag = True
                except VisaIOError as e:
                    logger.error(f'could not set stimulus: {e}')
        
        # If is_set flag shows Set was properly initiated
        if is_set_flag:
            # Mark as set
            self.mark_as_set(idx)
            # Change acquisition file basename & parameters in MATLAB GUI
            self.update_scanimage_basename(code) 

    @property
    def to_lock(self):
        ''' GUI widgets that must be enabled/disabled during actions '''
        return [self.acq_params_dict, self.protocol_table, self.fselect]

    def on_shuffle(self, *args, **kwargs):
        ''' Callback for protocol shuffling '''
        logger.info(f'shuffling protocol')
        self.fdata['protocol'] = self.fdata['protocol'].sample(frac=1).reset_index(drop=True)
        self.update_protocol_table()

        # Start efficiency monitoring session
        self.efficiency_monitor.start_session('experiment')
        self.setup_efficiency_monitoring_from_transducer()

    @lock_actions(background=True)
    def on_next(self, *args, **kwargs):
        ''' Callback for switching to next protocol entry '''
        self.row_idx = self.get_next_entry()
        if self.row_idx is None:
            logger.warning('protocol completed')
        else:
            self.set_stimulus(self.row_idx)

    @lock_actions(background=True)
    def on_reset(self, *args, **kwargs):
        ''' Callback for resetting current protocol entry '''
        if self.row_idx is None:
            logger.warning('protocol completed / no protocol')
        else:
            self.set_stimulus(self.row_idx)

    @lock_actions(background=True)
    def on_acquire(self, *args, **kwargs):
        ''' Callback for starting acquisition '''
        logger.info('starting ScanImage 2P acquisition...')

        if self.testmode:
            time.sleep(1.)
        else:
            # Start ScanImage acquisition (this triggers the ultrasound burst)
            self.instruments['si'].obj.start_SI_acquisition()

            # Wait for acquisition to complete
            while self.is_scanimage_acquiring():
                time.sleep(0.1)

    @lock_actions(background=True)
    def on_burst_check(self, *args, **kwargs):
        '''
        Run a standalone burst efficiency check using calibration-style acquisition.
        
        Creates a temporary Calibrator, fires gated sine bursts with the current
        transducer frequency, and measures FWD/REV coupling amplitudes from the
        Hilbert envelope of the last 10 carrier cycles (steady state).
        
        Uses reference Vpp from the current protocol entry if available, otherwise
        falls back to REF_VPP_CALIB (0.2 V).
        '''
        logger.info('Starting burst efficiency check...')

        # Determine Vpp: use current protocol entry if available, else default
        Vpp = REF_VPP_CALIB
        if self.protocol_data is not None and self.row_idx is not None:
            try:
                row = self.protocol_data.loc[self.row_idx]
                Vpp = self.interpVpp(row[P_KEY]) * MV_TO_V
                logger.info(f'Using Vpp={Vpp*1000:.1f} mV from current protocol entry')
            except Exception:
                logger.info(f'Using default Vpp={Vpp*1000:.1f} mV')
        else:
            logger.info(f'No protocol entry selected, using default Vpp={Vpp*1000:.1f} mV')

        try:
            # Create temporary Calibrator for the burst check
            checker = Calibrator.create_for_burst_check(
                wg=self.instruments['wg'].obj,
                scope=self.instruments['osc'].obj,
                f=self.fMHz * MHZ_TO_HZ,
                testmode=self.testmode,
            )

            # Run burst efficiency check with warm-up and plotting
            yamps = checker.run_burst_efficiency_check(
                Vpp,
                BD=0.2,         # 200ms burst
                PRF=100,        # 100Hz PRF
                DC=50,          # 50% duty cycle
                n_bursts=5,
                n_warmup=2,
                plot=True,
            )

            # Log results and feed into efficiency monitor
            if checker.CPL_FWD_KEY in yamps and checker.CPL_REV_KEY in yamps:
                fwd_v = yamps[checker.CPL_FWD_KEY]
                rev_v = yamps[checker.CPL_REV_KEY]
                self.efficiency_monitor.check_pulse(fwd_v, rev_v)
                logger.info('Burst efficiency check complete.')

        except Exception as e:
            logger.error(f'Burst efficiency check failed: {e}')

    def on_skip(self, *args, **kwargs):
        ''' Callback for skipping current protocol entry '''
        self.row_idx = self.get_next_entry()
        if self.row_idx is None:
            logger.warning('protocol completed')
        else:
            self.mark_as_skipped(self.row_idx)
    
    def enable_actions(self, *keys):
        super().enable_actions(*keys)
        self.update_actions()

    def on_clear(self, *args, **kwargs):
        ''' Callback for clearing protocol '''
        logger.info('clearing protocol')

        # End efficiency monitoring session and log summary
        if self.efficiency_monitor.measurements:
            summary = self.efficiency_monitor.end_session()
            logger.info(f'Efficiency summary: {summary}')

        self.fdata['protocol']['order'] = np.nan
        self.row_idx = None
        self.update_protocol_table()
        self.update_actions()
    
    def get_scanimage_framerate(self):
        ''' Extract acquisition frame rate from MATLAB ScanImage '''
        if self.testmode or not self.instruments['si'].iscon:
            return EXP_SR
        elif not hasattr(self, 'sifr'):
            logger.info('extracting ScanImage frame rate...')
            self.sifr = self.instruments['si'].obj.get_SI_framerate()
        return self.sifr 
    
    def is_scanimage_acquiring(self):
        ''' Return whether ScanImage is acquiring or not '''
        if self.testmode or not self.instruments['si'].iscon:
            return False
        return self.instruments['si'].obj.is_SI_acquiring()

    # ------------------- PROPERTIES -------------------

    @property
    def session_id(self):
        return self.acq_params_dict['session name'].text()
    
    @property
    def sr(self):
        return self.acq_params_dict['sampling rate (Hz)'].value()  # Hz

    @property
    def nperacq(self):
        return self.acq_params_dict['# frames / acq'].value()
    
    @property
    def nacqs(self):
        return self.acq_params_dict['# acqs / cond'].value()
    
    @property
    def stimdelay(self):
        return self.acq_params_dict['stim delay (s)'].value()  # s

    def get_required_memory(self):
        ''' Compute required disk space (in GB) for next acquisition '''
        if self.instruments['si'].iscon:
            npix_per_frame = int(self.instruments['si'].obj.get_SI_npix_per_frame())
            nchannels = int(self.instruments['si'].obj.get_SI_nchannels_save())
            req_bytes = self.nacqs * nchannels * self.nframes * npix_per_frame * NBYTES_INT16
            return req_bytes * BYTES_TO_GBYTES
        else:
            return 0

    def get_avail_memory(self):
        ''' Get available disk space (in GB) on disk of ScanImage output directory '''
        if self.instruments['si'].iscon:
            acq_outdir = self.instruments['si'].obj.get_SI_outdir()
            if not isinstance(acq_outdir, str):
                return None
            *_, avail_bytes = shutil.disk_usage(acq_outdir)
            return avail_bytes * BYTES_TO_GBYTES
        else:
            return 1
    
    def check_memory(self):
        '''
        Check that available disk space is large enough to store acquisition data
        
        :return: whether there is enough avilable disk space
        '''
        if self.testmode:
            return True
        # Get required and available disk space
        req_GB = self.get_required_memory()
        avail_GB = self.get_avail_memory()
        if avail_GB is None:
            logger.error('no output directory chosen in ScanImage')
            return False
        is_sufficient = req_GB < avail_GB
        # Log warning if available space is insufficient
        if not is_sufficient:
            logger.error(
                f'insufficient disk space ({avail_GB:.2f} GB) for next acquisition (requires {req_GB:.2f} GB)')
        return is_sufficient    
