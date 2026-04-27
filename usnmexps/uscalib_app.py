# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2025-02-11 11:16:43
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-06-13 14:40:23

# External packages
from PyQt6.QtWidgets import QLineEdit, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QRadioButton, QCheckBox
from instrulink import logger, grab_generator, grab_oscilloscope, grab_manipulator
from pyvisa.errors import VisaIOError

# Internal modules
from .constants import *
from .calib_utils import *
from .input_widgets import *
from .virtual_bench import *
from .virtual_instrument import VirtualInstrument
from .calibrators import *
from .utils import get_last_file, get_last_position, update_last_position, save_to_excel_file
from .pltutils import plot_calibration_data, plot_acoustic_field, plot_focal_distances
from .scanners import Scanner

class CreateCalibrationFileDialog(CustomDialog):
    ''' Generic dialog window for creating calibration files '''

    PREFIX = None
    
    @property
    def ID_label(self):
        ''' Return label for ID input field '''
        return f'{self.PREFIX} ID' if self.PREFIX is not None else 'ID'

    def add_ID_widget(self):
        ''' Add input field for ID '''
        row = QHBoxLayout()
        row.addWidget(QLabel(self.ID_label))
        self.IDwidget = QLineEdit()
        row.addWidget(self.IDwidget)
        self.box.addLayout(row)
    
    def add_freq_widget(self):
        ''' Add input field for frequency '''
        row = QHBoxLayout()
        row.addWidget(QLabel(F_KEY))
        self.fwidget = FloatInput(DEFAULT_CARRIER_FREQ, min=0, step=0.001).render()
        row.addWidget(self.fwidget)
        self.box.addLayout(row)
    
    def add_widgets(self):
        ''' Add widgets to dialog '''
        self.add_ID_widget()
        self.add_freq_widget()

    def get_values(self):
        ''' Retrieve input values '''
        # Get ID
        ID = self.IDwidget.text()
        if ID == '':
            raise ValueError(f'{self.ID_label} must be specified')
        
        # Get transducer frequency
        f = self.fwidget.value()  # MHz

        # Return dictionary of input values
        return {
            F_KEY: f,
            'ID': ID,
        }


class CreateTransducerFileDialog(CreateCalibrationFileDialog):
    ''' Dialog window for creating transducer file '''

    PREFIX = 'transducer'

    def add_widgets(self):
        ''' Add widgets to dialog '''
        # Add parent widgets
        super().add_widgets()

        # Radio buttons for calibration conditions
        row = QGroupBox('calibration condition')
        radio_layout = QHBoxLayout()
        self.radios = {}
        for opt in CALIB_CONDS:
            self.radios[opt] = QRadioButton(opt)
            radio_layout.addWidget(self.radios[opt])
        self.radios[CALIB_CONDS[0]].setChecked(True)  # Default selection
        row.setLayout(radio_layout)
        self.box.addWidget(row)
    
    def get_values(self):
        ''' Retrieve input values '''
        # Get parent inputs
        d = super().get_values()

        # Get calibration condition
        cond = next((k for k, radio in self.radios.items() if radio.isChecked()), None)
        if cond is None:
            raise ValueError('calibration condition must be specified')
        d['condition'] = cond

        # Return dictionary of input values 
        return d
    

class CreateAmplifierFileDialog(CreateCalibrationFileDialog):
    ''' Dialog window for creating amplifier file '''

    PREFIX = 'amplifier'

    def add_widgets(self):
        ''' Add widgets to dialog '''
        # Add parent widgets
        super().add_widgets()

        # Input field for attenuator loss
        row = QHBoxLayout()
        row.addWidget(QLabel('attenuator loss (dB)'))
        self.attwidget = FloatInput(-20, max=0, step=1).render()
        row.addWidget(self.attwidget)
        self.box.addLayout(row)

    def get_values(self):
        ''' Retrieve input values '''
        # Get parent inputs
        d = super().get_values()

        # Get attenuator loss
        d['att loss'] = self.attwidget.value()  # dB

        # Return dictionary of input values
        return d
    

class SelectMappingOptionsDialog(CustomDialog):
    ''' Dialog window for selecting mapping options '''
    
    def add_widgets(self):
        ''' Add widgets to dialog '''
        # Radio buttons for mapping mode
        row = QGroupBox('mapping mode')
        radio_layout = QHBoxLayout()
        self.mmode_radios = {}
        for opt in MAPPING_MODES:
            self.mmode_radios[opt] = QRadioButton(opt)
            radio_layout.addWidget(self.mmode_radios[opt])
        self.mmode_radios[MAPPING_MODES[0]].setChecked(True)  # Default selection
        row.setLayout(radio_layout)
        self.box.addWidget(row)
        
        # Check box for saving waveforms
        self.save_waveforms = QCheckBox('save detailed waveforms')
        self.box.addWidget(self.save_waveforms)

        # Radio buttons for end mode
        row = QGroupBox('end mode')
        radio_layout = QHBoxLayout()
        self.endmode_radios = {}
        for opt in SCAN_ENDMODES:
            self.endmode_radios[opt] = QRadioButton(opt)
            radio_layout.addWidget(self.endmode_radios[opt])
        self.endmode_radios[SCAN_ENDMODES[0]].setChecked(True)  # Default selection
        row.setLayout(radio_layout)
        self.box.addWidget(row)
    
    def get_values(self):
        ''' Retrieve selected mapping options '''
        # Get mapping mode
        mmode = next((k for k, radio in self.mmode_radios.items() if radio.isChecked()), None)
        if mmode is None:
            raise ValueError('mapping mode must be specified')
    
        # Get save waveforms option
        save_waveforms = self.save_waveforms.isChecked()

        # Get end mode
        endmode = next((k for k, radio in self.endmode_radios.items() if radio.isChecked()), None)
        if endmode is None:
            raise ValueError('end mode must be specified')

        # Return dictionary of input values
        return {
            'mmode': mmode,
            'wfsave': save_waveforms,
            'endmode': endmode
        }


class SetMovingDistanceDialog(CustomDialog):
    ''' Dialog window for setting micro-manipulator moving distance '''

    def  __init__(self, *args, default_distance=200., **kwargs):
        ''' 
        Constructor
        
        :param default_distance: default moving distance (um)
        '''
        self.default_distance = default_distance
        super().__init__(*args, **kwargs)

    def add_widgets(self):
        ''' Add widgets to dialog '''
        # Input field for moving distance
        row = QHBoxLayout()
        row.addWidget(QLabel('moving distance (mm, positive = away from base plane)'))
        self.distwidget = FloatInput(
            self.default_distance / MM_TO_UM, step=.05).render()
        self.distwidget.setMinimumWidth(80)
        row.addWidget(self.distwidget)
        self.box.addLayout(row)

    def get_values(self):
        ''' Retrieve input values '''
        # Return moving distance
        return self.distwidget.value() * MM_TO_UM  # um



class CalibrationApp(VirtualBenchApp):
    ''' Desktop application for US calibration & mapping '''

    # Actions dictionary
    ACTIONS_DICT = {
        'general': {
            'acquire': 'acquire',
            'check_efficiency': 'check efficiency',
            'mark_base': 'mark base',
            'save_focal_dist': 'save focal distance',
        },
        'sweeps': {
            'iosweep_run': 'run I/O sweep',
            'iosweep_plot': 'plot I/O sweep',
            'fsweep_run': 'run f-sweep',
        },
        'scan': {
            'focus': 'find focus',
            'plot_fdist': 'plot focal dists.',
            'field_map': 'map 3D field',
            'field_plot': 'plot 3D field',
        },
        'manipulator': {
            'mp_resetorg': 'zero',
            'mp_move_along_normal': 'move along norm. vector',
            'mp_move_updown': 'move up/down',
            'mp_save_pos': 'save position',
            'mp_goto_saved_pos': 'go to saved position',
        }
    }

    def __init__(self, **kwargs):
        ''' Constructor '''
        # Instruments list
        instruments = [
            VirtualInstrument('wg', 'waveform generator', lambda: grab_generator(key='rigol')),
            VirtualInstrument('sc', 'oscilloscope', lambda: grab_oscilloscope(key='bk')),
            VirtualInstrument('mm', 'micro-manipulator', lambda: grab_manipulator(key='sutter'))
        ]

        # Placeholder for calibrator instance
        self.calibrator = None

        # Set empty placeholder for reference base plane position
        self.ref_base_pos = None

        # Initialize parent class
        super().__init__(instruments, **kwargs)

        # Set default files for transducer and hydrophone
        for key in ['transducer', 'hydrophone']:
            self.update_input_file(key, get_last_file(key))
    
    # ------------------- LAYOUT -------------------
    
    def set_layout(self):
        ''' Set layout '''
        body = QHBoxLayout()

        # Left column: stimulus, acquisition and scan parameters
        leftcol = QVBoxLayout()
        self.stim_params_dict, stim_params_table = CALIB_STIM_PARAMS.render() 
        self.acq_params_dict, acq_params_table = CALIB_ACQ_PARAMS.render()

        # Add acquisition options checkboxes
        l = QHBoxLayout()
        self.acq_opts_dict = {
            'measure signal': QCheckBox('measure physical signal'),
            'measure coupling': QCheckBox('measure FWD/REV coupling'),
        }
        self.acq_opts_dict['measure signal'].setChecked(True)
        for w in self.acq_opts_dict.values():
            l.addWidget(w)
        acq_opts = QGroupBox('Acquisition options')
        acq_opts.setLayout(l)
        
        self.scan_params_dict, scan_params_table = CALIB_SCAN_PARAMS.render()
        leftcol.addWidget(stim_params_table)
        leftcol.addWidget(acq_params_table)
        leftcol.addWidget(acq_opts)
        leftcol.addWidget(scan_params_table)
        leftcol.addWidget(self.actions_panel(self.ACTIONS_DICT))
        body.addLayout(leftcol)

        # Right column: instruments, input files, action buttons & log
        rightcol = QVBoxLayout()
        rightcol.addWidget(self.instruments_panel())
        rightcol.addWidget(self.input_files_panel())
        rightcol.addWidget(self.graphs_panel())
        rightcol.addWidget(self.log_panel())
        
        body.addLayout(rightcol)
        self.setLayout(body)
    
    def input_files_panel(self):
        ''' Input files selection panel '''
        return super().input_files_panel({
            'transducer': (EXCEL_FILTER, TRANSDUCERS_FOLDER, True),
            'amplifier': (EXCEL_FILTER, AMPLIFIERS_FOLDER, True),
            'hydrophone': (EXCEL_FILTER, HYDROPHONES_FOLDER, False),
        })
    
    def graphs_panel(self):
        group = QGroupBox('Graphs')
        layout = QVBoxLayout()
        self.canvas = ResizableFigureCanvas(parent=self)
        layout.addWidget(self.canvas)
        group.setLayout(layout)
        return group
    
    # ------------------- CALLBACKS -------------------
    
    def register_callbacks(self):
        ''' Register callbacks '''
        # Input frequency change
        self.stim_params_dict[F_KEY].valueChanged.connect(self.on_freq_change)

        # Input files change
        self.fwidget['transducer'].textChanged.connect(self.on_transducer_file_change)
        self.fwidget['hydrophone'].textChanged.connect(self.on_hydrophone_file_change)
        self.fwidget['amplifier'].textChanged.connect(self.on_amplifier_file_change)

        # Update action buttons upon changes in channel generator/scope channel selection
        for w in self.stim_params_dict['channels [trig.] [sig.]'] + self.acq_params_dict['channels [trig.] [sig.] [cplfwd.] [cplrev.]']:
            w.valueChanged.connect(self.update_actions)
        
        # Update action buttons upon changes in acquisition options
        for w in self.acq_opts_dict.values():
            w.stateChanged.connect(self.update_actions)

        # Actions clicks
        self.connect_actions()
    
    def create_input_file(self, key, dirpath):
        ''' Create input calibration file '''
        if key == 'transducer':
            fpath = self.create_transducer_calibration_file(dirpath)
        elif key == 'amplifier':
            fpath = self.create_amplifier_calibration_file(dirpath)
        else:
            raise ValueError(f'unsupported file type: {key}')
        self.update_input_file(key, fpath)
    
    def create_transducer_calibration_file(self, dirpath):
        ''' Create transducer calibration file '''
        # Create dialog for transducer settings specification
        dialog = CreateTransducerFileDialog(self)

        # If cancel was pressed, return
        if not dialog.exec():
            return
        
        # If OK was pressed, extract settings
        try:
            transducer_settings = dialog.get_values()
        
        # If error occurred, log it and return
        except ValueError as e:
            logger.error(e)
            return
        
        # Construct calibration file path
        calib_cond = transducer_settings['condition'].replace(' ', '')
        fname = f'{transducer_settings["ID"]}_{calib_cond}_{transducer_settings[F_KEY]:.3f}MHz.xlsx'
        fpath = os.path.join(dirpath, fname)

        # Abort if file already exists
        if os.path.isfile(fpath):
            logger.error(f'{fpath} already exists -> aborting')
            return

        # Save file
        logger.info(f'creating transducer calibration file: {fname}')
        df = pd.DataFrame({VIN_KEY: INPUT_VPPS_CALIB / MV_TO_V})
        save_to_excel_file(df, fpath)
        
        # Return file path
        return fpath
    
    def create_amplifier_calibration_file(self, dirpath):
        ''' Create amplifier calibration file '''
        # Create dialog for amplifier settings specification
        dialog = CreateAmplifierFileDialog(self)

        # If cancel was pressed, return
        if not dialog.exec():
            return
        
        # If OK was pressed, extract settings
        try:
            amplifier_settings = dialog.get_values()
        
        # If error occurred, log it and return
        except ValueError as e:
            logger.error(e)
            return

        # Construct calibration file path
        fname = f'{amplifier_settings["ID"]}_{amplifier_settings["att loss"]:.2f}dB_{amplifier_settings[F_KEY]:.2f}MHz.xlsx'
        fpath = os.path.join(dirpath, fname)

        # Abort if file already exists
        if os.path.isfile(fpath):
            logger.error(f'{fpath} already exists -> aborting')
            return

        # Input Vpp range
        Vpps_in = np.linspace(INPUT_VPPS_CALIB[0], INPUT_VPPS_CALIB[-1], 11)
        check_input_voltages(Vpps_in)

        # Save file
        logger.info(f'creating amplifier calibration file: {fname}')
        df = pd.DataFrame({VIN_KEY: Vpps_in / MV_TO_V})
        save_to_excel_file(df, fpath)
        
        # Return file path
        return fpath
    
    def update_actions(self):
        ''' Update action buttons status '''
        hascorrectseqs = self.has_correct_channel_sequences
        for k in ['acquire', 'iosweep_run', 'fsweep_run']:
            self.actions_btns[k].setEnabled(self.can_acquire and hascorrectseqs)
        for k in ['focus', 'field_map']:
            self.actions_btns[k].setEnabled(self.can_scan and hascorrectseqs)
        self.actions_btns['check_efficiency'].setEnabled(
            self.can_acquire and hascorrectseqs and self.has_coupling_data and self.acq_measure_coupling)
        self.actions_btns['iosweep_plot'].setEnabled(self.can_plot_iocurve)
        self.actions_btns['field_plot'].setEnabled(True)
        for k in ['mp_resetorg', 'mp_move_along_normal', 'mp_move_updown', 'mp_save_pos', 'mark_base']:
            self.actions_btns[k].setEnabled(self.has_scan_instr)
        self.actions_btns['save_focal_dist'].setEnabled(
            self.has_scan_instr and self.ref_base_pos is not None and self.fpaths['transducer'] is not None)
        self.actions_btns['mp_goto_saved_pos'].setEnabled(self.has_scan_instr and self.has_last_pos)
        self.actions_btns['plot_fdist'].setEnabled(True)
    
    def on_freq_change(self):
        ''' Callback when fundamental frequency has changed '''
        #  If hydrophone data loaded, update conversion factor
        if self.fpaths['hydrophone'] is not None:
            fMHz = self.stim_params_dict[F_KEY].value()
            Ml = get_hydrophone_conversion_constant(self.fdata['hydrophone'], fMHz)
            self.acq_params_dict['hydrophone sensitivity (V/MPa)'].setValue(Ml / PA_TO_MPA)
    
    def on_transducer_file_change(self, fname):
        ''' Callback when transducer calibration file path has changed '''
        if self.fpaths['transducer'] is None:
            return

        # Parse transducer file, update fundamental frequency (triggers automatic
        # hydrophone update) and clear amplifier data
        try:
            params = parse_transducer_calibration_file(fname)
            fMHz = params['freq (MHz)']
            logger.info(f'loaded transducer file: {fname}')
            self.clear_file_input('amplifier')
            self.acq_params_dict['probe attenuation (dB)'].setValue(0)
        except ValueError as e:
            logger.error(e)
            self.clear_file_input('transducer')
            fMHz = DEFAULT_CARRIER_FREQ
        self.stim_params_dict[F_KEY].setValue(fMHz)

        # Set empty placeholder for reference base plane position
        self.ref_base_pos = None

        # Update action buttons status
        self.update_actions()
    
    def on_hydrophone_file_change(self, fname):
        ''' Callback when hydrophone calibration file path has changed '''
        if self.fpaths['hydrophone'] is None:
            return

        # Check that correct hydrophone calibration data was loaded and
        # update conversion factor, clear amplifier data
        try:
            if any(x not in self.fdata['hydrophone'].columns for x in [F_KEY, CONV_KEY]):
                raise ValueError('missing columns')
            self.on_freq_change()
            logger.info(f'loaded hydrophone file: {fname}')
            self.clear_file_input('amplifier')
        except ValueError as e:
            logger.error(e)
            self.clear_file_input('hydrophone')
        
        # Update action buttons status
        self.update_actions()
    
    def on_amplifier_file_change(self, fname):
        ''' Callback when amplifier calibration file path has changed '''
        if self.fpaths['amplifier'] is None:
            return

        # Parse amplifier calibration file and update attenuation gain and frequency,
        # clear transducer data
        try:
            self.amp_gain = extract_amplifier_gain(fname)
            if self.amp_gain is not None:
                logger.info(f'amplifier theoretical gain = {self.amp_gain} dB')
            params = parse_amplifier_calibration_file(fname)
            fMHz = params['freq (MHz)']
            att_gain = params['gain (dB)']
            logger.info(f'loaded amplifier file: {fname}')
            self.clear_file_input('transducer')
            self.acq_params_dict['probe attenuation (dB)'].setValue(att_gain)
        except ValueError as e:
            logger.error(e)
            self.clear_file_input('amplifier')
            fMHz = DEFAULT_CARRIER_FREQ
            att_gain = 0.
        self.stim_params_dict[F_KEY].setValue(fMHz)
        self.acq_params_dict['probe attenuation (dB)'].setValue(att_gain)
        
        # Update action buttons status
        self.update_actions()
    
    @property
    def to_lock(self):
        ''' GUI widgets that must be enabled/disabled during actions '''
        return [self.stim_params_dict, self.acq_params_dict, self.scan_params_dict, self.fselect, self.fcreate]
    
    def stop_action(self, key):
        if self.calibrator is not None:
            self.calibrator.isrunning = False
        super().stop_action(key)
    
    def get_calibrator(self, ncycles=None, trigger_mode='loop'):
        ''' 
        Get transducer or amplifier calibrator instance from current settings
        
        :param ncycles: number of cycles per pulese (optional). If None, use current setting.
        :param trigger_mode: calibrator trigger mode (optional). Default is 'loop'.
        :return: calibrator instance
        '''
        # If number of cycles is not specified, use current setting
        if ncycles is None:
            ncycles = self.ncycles

        # Gather common calibrator positional and keyword arguments
        args = [
            self.instruments['wg'].obj,
            self.instruments['sc'].obj,
            self.f0,  
            self.wch_sig,
            self.sch_sig if self.acq_measure_signal else None,
        ]
        kwargs = dict(
            wch_trig=self.wch_trig,
            sch_trig=self.sch_trig,
            sch_cpl_fwd=self.sch_cpl_fwd if self.acq_measure_coupling else None,
            sch_cpl_rev=self.sch_cpl_rev if self.acq_measure_coupling else None,
            ncycles=ncycles,
            PRF=self.PRF,
            acq_npoints=OSC_ACQ_NPOINTS,
            acq_nsweeps=self.acq_nsweeps,
            trigger_mode=trigger_mode,
            canvas=self.canvas,
            testmode=self.testmode,
        )

        # Return appropriate calibrator instance
        if self.acqmode == 'amplifier':
            calibrator = AmplifierCalibrator(
                *args,
                output_conv_constant=gain_to_vratio(self.att_gain),
                **kwargs
            )
        else:
            calibrator = TransducerCalibrator(
                *args, 
                Ml=self.Ml,
                **kwargs
            )
        
        return calibrator
    
    def get_scanner(self):
        ''' Get scanner instance from current scanning parameters '''
        return Scanner(
            mp=self.instruments['mm'].obj,
            vbasis=VBASIS,
            theta=self.theta
        )
    
    def send_stop_signal(self, key):
        ''' Stop background thread '''
        if key == 'sleep':
            super().send_stop_signal(key)
        else:
            self.calibrator.isrunning = False
    
    @lock_actions()
    def on_acquire(self, *args, **kwargs):
        ''' Callback for continuous acquisition '''
        # Run continuous acquisition
        try:
            self.calibrator = self.get_calibrator()
            self.calibrator.run_acquisition(
                self.ref_Vpp,
                acq_interval=self.acqinterval,
            )
            logger.info('done.')

        # Log error, if any
        except (VisaError, VisaIOError, ValueError) as e:
            self.calibrator.disable_generator()
            logger.error(e)
    
    @staticmethod
    def extract_coupling_reference(data):
        ''' 
        Extract forward and reverse coupling vectors from last calibration date
        
        :return: dataframe with forward and reverse coupling columns, indexed by input Vpp
        '''
        # Extract coupling reference from last calibration date 
        logger.info('extracting coupling reference from last calibration...')
        
        # Identify forward and reverse coupling columns
        cplkeys = list(filter(
            lambda k: re.match(CALIBRATION_CPL_PATTERN, k), data.columns))
        fwdkeys = [c for c in cplkeys if 'CPLFWD' in c]
        revkeys = [c for c in cplkeys if 'CPLREV' in c]
        assert len(fwdkeys) == len(revkeys), 'mismatched coupling columns'

        # Extract forward and reverse coupling dataframes
        cpl_data = {
            'CPLFWD (Vpp)': data[fwdkeys],
            'CPLREV (Vpp)': data[revkeys]
        }

        # Select only columns from last calibration date
        filtered_cpl_data = {}
        for k, df in cpl_data.items():
            cplkeys_by_date = parse_outputs_by_date(df.columns, rgxp=CALIBRATION_CPL_PATTERN)
            last_date = list(cplkeys_by_date.keys())[-1]
            cplkeys = cplkeys_by_date[last_date]
            filtered_cpl_data[k] = df[cplkeys]

        # Average across all calibration runs from last calibration date
        ref_cpl_data = pd.concat(
            {k: df.mean(axis=1).rename(k) for k, df in filtered_cpl_data.items()},
            axis=1
        )
        ref_cpl_data.index = (data[VIN_KEY] * MV_TO_V).rename('Vin (Vpp)')

        # Fill in missing values by interpolation
        ref_cpl_data = ref_cpl_data.interpolate(method='index')

        # Return reference coupling data
        return ref_cpl_data

    @lock_actions()
    def on_check_efficiency(self, *args, **kwargs):
        ''' 
        Callback for checking transducer efficiency using burst mode.
        Shows waveform plots for visual verification.
        '''
        # Fetching reference forward and reverse coupling vectors from last calibration
        ref_cpl_data = self.extract_coupling_reference(self.calib_data)

        # Interpolate values at current Vpp
        ref_cpl_at_Vpp = {k: np.interp(
            self.ref_Vpp, ref_cpl_data.index, ref_cpl_data[k]) for k in ref_cpl_data.columns}

        # Compute reference coupling ratio at current Vpp
        ref_cpl_at_Vpp['ratio'] = ref_cpl_at_Vpp['CPLREV (Vpp)'] / ref_cpl_at_Vpp['CPLFWD (Vpp)'] if ref_cpl_at_Vpp['CPLFWD (Vpp)'] != 0 else np.nan
        ref_cpl_at_Vpp['ratio_dB'] = vratio_to_gain(ref_cpl_at_Vpp['ratio'])

        logger.info(f'reference coupling at {self.ref_Vpp:.3f} Vpp:')
        for k, v in ref_cpl_at_Vpp.items():
            logger.info(f'  {k}: {v:.3f}')
        
        # Run efficiency check through the reusable burst runner so timing,
        # triggering, and waveform extraction match the production path.
        try:
            # Instantiate calibrator
            self.calibrator = self.get_calibrator(
                trigger_mode='prog', 
            )

            yamps = self.calibrator.run_burst_efficiency_check(
                self.ref_Vpp,
                BD=EXP_DUR * MS_TO_S,
                PRF=self.PRF,
                DC=50,
                n_bursts=5,
                pulse_to_sample=10,
                acq_interval=self.acqinterval,
                plot=False,
                n_warmup=2,
            )

            if self.calibrator.CPL_FWD_KEY not in yamps or self.calibrator.CPL_REV_KEY not in yamps:
                raise ValueError('coupling channels are required for efficiency check')
            if yamps[self.calibrator.CPL_FWD_KEY] <= 0:
                raise ValueError('measured CPLFWD is zero; cannot compute coupling ratio')

            cplratio = yamps[self.calibrator.CPL_REV_KEY] / yamps[self.calibrator.CPL_FWD_KEY]
            cplratio_dB = vratio_to_gain(cplratio)
            logger.info(
                f'measured coupling at {self.ref_Vpp:.3f} Vpp: '
                f'ratio={cplratio:.3f} ({cplratio_dB:.2f} dB)'
            )

            delta_db = cplratio_dB - ref_cpl_at_Vpp['ratio_dB']
            delta_db_str = f'{delta_db:+.2f} dB'
            logger.info(f'deviation from reference: {delta_db_str}')
            if abs(delta_db) < MAX_CPL_DEV_DB:
                logger.info('transducer efficiency within margin of error')
            else:
                logger.warning('transducer efficiency outside margin of error')

            logger.info('efficiency check done.')
            # yamps = self.calibrator.run_acquisition(
            #     self.ref_Vpp,
            #     acq_interval=self.acqinterval,
            #     max_iter=5
            # )
        
        # Log error, if any
        except (VisaError, VisaIOError, ValueError) as e:
            logger.error(e)
        finally:
            if hasattr(self, 'calibrator') and self.calibrator is not None:
                try:
                    self.calibrator.disable_generator()
                except Exception:
                    pass

        # # Compute coupling ratio and log it
        # cplratio = yamps['cpl_rev'] / yamps['cpl_fwd']
        # cplratio_dB = vratio_to_gain(cplratio)  # dB
        # logger.info(f'measured coupling ratio: {cplratio:.3f} ({cplratio_dB:.2f} dB)')

        # # Compute deviation from reference and log it
        # delta_db = cplratio_dB - ref_cpl_at_Vpp['ratio_dB']
        # delta_dB_str = f'{delta_db:.2f} dB' if delta_db < 0 else f'+{delta_db:.2f} dB'
        # logger.info(f'deviation from calibration reference: {delta_dB_str}')

        # # If coupling ratio is within margin of error, log success
        # if abs(cplratio_dB - ref_cpl_at_Vpp['ratio_dB']) < MAX_CPL_DEV_DB:
        #     logger.info('transducer efficiency within margin of error')
        # # Otherwise, log error
        # else:
        #     logger.error('transducer efficiency outside margin of error!!!')


        # # Run burst efficiency check with plotting
        # try:
        #     self.calibrator = self.get_calibrator(trigger_mode='single')
        #     yamps = self.calibrator.run_burst_efficiency_check(
        #         self.ref_Vpp,
        #         BD=0.2,              # 200ms burst
        #         PRF=100,             # 100Hz PRF
        #         DC=50,               # 50% duty cycle
        #         n_bursts=5,
        #         pulse_to_sample=10,
        #         plot=True            # Show waveform plots
        #     )
        
        # except (VisaError, VisaIOError, ValueError) as e:
        #     self.calibrator.disable_generator()
        #     logger.error(e)
        #     return

        # # Calculate measured coupling ratio
        # if yamps[self.calibrator.CPL_FWD_KEY] > 0.0001:
        #     cplratio = yamps[self.calibrator.CPL_REV_KEY] / yamps[self.calibrator.CPL_FWD_KEY]
        #     cplratio_dB = vratio_to_gain(cplratio)
        # else:
        #     logger.error('FWD amplitude too low - cannot calculate coupling ratio')
        #     return
        
        # logger.info(f'Measured coupling: {cplratio:.3f} ({cplratio_dB:.2f} dB)')

        # # Compare to expected
        # delta_db = cplratio_dB - ref_cplratio_dB
        # delta_dB_str = f'{delta_db:.2f} dB' if delta_db < 0 else f'+{delta_db:.2f} dB'
        # logger.info(f'Deviation from expected: {delta_dB_str}')

        # # Pass/fail
        # if abs(cplratio_dB - ref_cplratio_dB) < MAX_CPL_DEV_DB:
        #     logger.info('transducer efficiency within margin of error')
        # else:
        #     logger.error('transducer efficiency outside margin of error!!!')

        # # Follow-up diagnostic: pulse-by-pulse profile using pulse-indexed acquisitions
        # try:
        #     self.calibrator.profile_burst_pulse_stability(
        #         self.ref_Vpp,
        #         BD=0.2,
        #         PRF=100,
        #         DC=50,
        #         n_bursts=1,
        #         plot=True,
        #         n_warmup=0,
        #     )
        # except (VisaError, VisaIOError, ValueError) as e:
        #     logger.warning(f'Pulse stability profile failed: {e}')

    @lock_actions(background=False)
    def on_mark_base(self, *args, **kwargs):
        '''
        Store current micro-manipulator position as base plane reference
        '''
        # Get current position
        pos = self.instruments['mm'].obj.get_position()  # um
        logger.info(f'storing base plane reference position: {pos} um')
        self.ref_base_pos = pos
        self.update_actions()
    
    def on_save_focal_dist(self, *args, **kwargs):
        '''
        Evaluate distance from base plane reference position
        '''
        # Get current position
        pos = self.instruments['mm'].obj.get_position()  # um
        logger.info(f'current position: {pos} um')

        # Project position onto reference plane using reference position as plane origin
        proj_pos = project_onto_plane(
            pos, self.get_scanner().get_normal_vector(), self.ref_base_pos)
        logger.info(f'projected position on base plane: {proj_pos} um')

        # Compute distance from base plane reference position
        delta = pos - proj_pos  # um
        dist = np.linalg.norm(delta)  # um
        logger.info(f'distance from base plane: {dist:.2f} um')

        # Saving focal distance
        tid = os.path.splitext(os.path.basename(self.fpaths['transducer']))[0]
        save_focal_distance(tid, dist)
    
    def pre_focus(self):
        ''' 
        Pre-focus search routine
        
        :return: whether the focus search should proceed
        '''
        # Ask user about hydrophone position, abort if not properly positioned
        is_positioned = self.yesno_dialog('Is the hydrophone positioned around the acoustic focus?')
        if not is_positioned:
            logger.warning('aborting focus search')
            return False
        return True

    @lock_actions(background=False)
    def on_focus(self, *args, **kwargs):
        ''' Callback for finding the acoustic focus '''
        # Run focus search
        try:
            self.calibrator = self.get_calibrator()#ncycles=NCYCLES_SCAN)
            self.calibrator.find_acoustic_focus(
                self.instruments['mm'].obj, 
                self.ref_Vpp,
                DELTA_FOCUS,
                NPERAX_SEARCH,
                self.theta,
                VBASIS,
                niters=self.niters,
                endmode='gotomax',
                order=FOCUS_CROSS_SCAN_ORDER,
                ask_pos=False,
                zstart='center',
            )
            logger.info('done.')

        # Log error, if any
        except (VisaError, VisaIOError, ValueError, SutterError, KeyboardInterrupt) as e:
            self.calibrator.disable_generator()
            logger.error(e)
    
    @lock_actions()
    def on_fsweep_run(self, *args, **kwargs):
        ''' Callback for running frequency sweep '''
        # Run frequency sweep
        try:
            self.calibrator = self.get_calibrator(ncycles=NCYCLES_CALIB)
            self.calibrator.run_freq_sweep(
                self.ref_Vpp, 
                FSWEEP_FREQS * MHZ_TO_HZ,
                acq_interval=self.acqinterval,
            )
            logger.info('done.')

        # Log error, if any
        except (VisaError, VisaIOError, ValueError) as e:
            self.calibrator.disable_generator()
            logger.error(e)
    
    def pre_iosweep_run(self):
        ''' Pre-I/O sweep routine '''
        # Instantiate calibrator
        self.calibrator = self.get_calibrator(ncycles=NCYCLES_CALIB)
        # Store IO curve input 
        self.Vpps_in = self.calib_data[VIN_KEY] * MV_TO_V  # V
        # Set empty placeholder for IO curve output
        self.IO_outs = None
        return True

    @lock_actions(background=False)
    def on_iosweep_run(self, *args, **kwargs):
        ''' Callback for running I/O sweep '''
        # Run calibration
        try:
            self.IO_outs = self.calibrator.run_io_sweep(
                self.Vpps_in, 
                acq_interval=self.acqinterval,
                ref_Vpp=self.ref_Vpp,
            )
            logger.info('done.')

        # If error, log it and disable generator
        except (VisaError, VisaIOError, ValueError) as e:
            self.calibrator.disable_generator()
            logger.error(e)

    def post_iosweep_run(self):
        ''' Post-I/O sweep routine '''
        # Update actions
        self.update_actions()

        # If calibration curve was not built, return
        if self.IO_outs is None:
            logger.warning('no I/O curve output data')
            return

        # Ask whether to save the calibration curve
        save = self.yesno_dialog('Save calibration curve to file?')
        if not save:
            logger.warning('calibration curve not saved')
            return

        # If amplifier calibration
        if isinstance(self.calibrator, AmplifierCalibrator) and 'sig' in self.IO_outs:
            # Compute voltage ratios and corresponding gains
            vals = self.IO_outs['sig']
            ratios = vals[1:] / self.Vpps_in[1:]
            gains = vratio_to_gain(ratios)

            # Log mean and std
            mu_ratio, sigma_ratio = np.nanmean(ratios), np.nanstd(ratios)
            mu_gain, sigma_gain = np.nanmean(gains), np.nanstd(gains)
            logger.info(f'voltage ratio = {mu_ratio:.3f} +/- {sigma_ratio:.3f}')
            logger.info(f'corresponding gains = {mu_gain:.2f} +/- {sigma_gain:.2f} dB')

            # Compute deviation from theoretical_gain (if any)
            if self.amp_gain is not None:
                gain_error = mu_gain - self.amp_gain
                # Log error warning if too large
                if abs(gain_error) > 2:
                    logger.error(
                        f'average measured gain deviates ({mu_gain:.2f} dB) significantly from theoretical gain ({self.amp_gain:.2f} dB)')

        # Add calibration curve(s) as new column(s) in dataframe
        for k, v in self.IO_outs.items():
            outkey, unit = {
                'sig': ('Pout', 'MPa'), 
                'cpl_fwd': ('CPLFWD', 'Vpp'),
                'cpl_rev': ('CPLREV', 'Vpp'),
                }[k]
            colkey = find_out_column_name(self.calib_data.columns, key=outkey, unit=unit)
            self.calib_data[colkey] = v
            logger.info(f'adding {outkey} calibration curve as column "{colkey}" in {self.calib_fpath}')

        # Check file availability
        check_file_availability(self.calib_fpath)
        
        # Save to file
        save_to_excel_file(self.calib_data, self.calib_fpath)

        # Return
        logger.info('done.')
        return
    
    def pre_field_map(self):
        '''
        Pre-mapping routine
        
        :return: whether the mapping should proceed
        '''
        # Ask for mapping options in dialog window
        dialog = SelectMappingOptionsDialog(self)

        # If cancel was pressed, return
        if not dialog.exec():
            logger.error('mapping aborted')
            return False

        # If OK was pressed, extract mapping options
        try:
            mopts = dialog.get_values()
            self.mapping_mode = mopts['mmode']
            self.mapping_endmode = mopts['endmode']
            wfsave = mopts['wfsave']
        
        # If error occurred, log it and return
        except ValueError as e:
            logger.error(f'{e} -> aborting mapping')
            return False
        
        # Ask for output directory, and abort if none given
        outdir = self.select_folder_dialog(initialdir=MAPPING_DATA_FOLDER)
        if outdir is None:
            logger.error('no output directory chosen -> aborting mapping')
            return False

        # Assemble transducer and hydrophone calibration file codes to form mapping code prefix
        fcodes = {}
        for key in ['transducer', 'hydrophone']:
            fcodes[key] = os.path.splitext(os.path.basename(self.fpaths[key]))[0]

        # Get calibrator
        self.calibrator = self.get_calibrator(ncycles=NCYCLES_SCAN)

        # Assemble full path to mapping output file
        mapping_code = self.calibrator.get_mapping_code(
            fcodes['transducer'], fcodes['hydrophone'], 
            self.delta, self.nperax, self.theta, mode=self.mapping_mode)
    
        # Determine path to output mapping file
        self.mapping_fpath = os.path.join(outdir, f'{mapping_code}.csv')

        # If file already exists, ask user whether to overwrite it
        if os.path.exists(self.mapping_fpath):
            overwrite = self.yesno_dialog(f'"{self.mapping_fpath}" file already exists -> overwrite?')        
            # If overwrite selected, delete file
            if overwrite:
                os.remove(self.mapping_fpath)
            # Otherwise, abort mapping
            else:
                logger.warning('aborting mapping')
                return False
        
        # If so, determine path to directory that will contain traces
        if wfsave:
            self.traces_dir = os.path.join(outdir, f'{mapping_code}_traces')
            if os.path.exists(self.traces_dir):
                logger.error(f'"{self.traces_dir}" directory already exists -> aborting mapping')
                return False
            else:
                os.makedirs(self.traces_dir)
        else:
            self.traces_dir = None
        
        # Return
        return True

    @lock_actions(background=True)
    def on_field_map(self, *args, **kwargs):
        ''' Callback for mapping the acoustic field '''   
        # Assemble mapping keyword arguments 
        map_kwargs = dict(
            out_fpath=self.mapping_fpath, 
            mode=self.mapping_mode,
            endmode=self.mapping_endmode,
            traces_dir=self.traces_dir,
            order=MAP_SCAN_ORDER,
            ask_pos=False,
            zstart='base'
        )
        
        # Map acoustic field
        try:
            self.calibrator.map_acoustic_field(
                None if self.testmode else self.instruments['mm'].obj,
                self.ref_Vpp,
                self.delta,
                self.nperax, 
                self.theta, 
                VBASIS,
                **map_kwargs
            )
            logger.info('done.')
        
        # If error, log it and disable generator
        except (VisaError, VisaIOError, ValueError, SutterError, KeyboardInterrupt) as e:
            self.calibrator.disable_generator()
            logger.error(e)
    
    def post_field_map(self):
        ''' Post-mapping routine '''
        # Plot mapping data
        self.plot_mapping_data()
    
    def plot_mapping_data(self, direct_call=False):
        ''' Plot mapping data ''' 
        # Load XYZ mapping data
        try:
            coords_per_dim, Pmat, fcode = load_mapping_data(self.mapping_fpath)
        except ValueError as e:
            logger.error(e)
            if direct_call:
                self.stop_action('field_plot')
            return
        
        # Plot acoustic field through focus
        fig = plot_acoustic_field(
            coords_per_dim,
            Pmat,
            'focus',
            xyz_unit='mm',
            gaussian_sigma='auto',
            out_mode='amp',
            title=fcode.replace('_', ' '),
            mark_focus=None,
            norm=False,
        )
    
        # Define callback for closing the pop-up canvas
        def on_close():
            self.popup_canvas = None
            if direct_call:
                self.stop_action('field_plot')
        
        # Show in dedicated pop-up canvas
        self.show_popup(
            fig, name='mapping data', on_close=on_close)

    @lock_actions(stoppable=False, unlock_on_finish=False)        
    def on_plot_fdist(self, *args, **kwargs):
        # Plot focal distances
        try:
            fig = plot_focal_distances(cbydate=True)
        except ValueError as e:
            logger.error(e)
            self.stop_action('plot_fdist')
            return

        # Define callback for closing the pop-up canvas
        def on_close():
            self.popup_canvas = None
            self.stop_action('plot_fdist')
        
        # Show in dedicated pop-up canvas
        self.show_popup(
            fig, name='focal distances', on_close=on_close)

    @lock_actions(stoppable=False, unlock_on_finish=False)
    def on_iosweep_plot(self, *args, **kwargs):
        ''' Callback for plotting I/O sweep data '''
        # Check that calibration data is available
        if self.calib_data is None:
            logger.error('no calibration data available')
            self.stop_action('iosweep_plot')
            return

        # Determine plot parameters from acquisition mode (transducer or amplifier)
        if self.acqmode == 'amplifier':
            kwargs = dict(
                ylabel=GAIN_KEY,
                convfunc=lambda Vin, Vout: vratio_to_gain(Vout / Vin),
                groundy=self.amp_gain is not None and self.amp_gain > 0,
                yref=self.amp_gain,
            )
        else:
            kwargs = dict(ylabel='all')

        # Extract file code 
        fcode = os.path.splitext(os.path.basename(self.calib_fpath))[0]
            
        # Plot calibration data from transducer or amplifier
        fig = plot_calibration_data(
            self.calib_data.copy(),
            details=True,
            title=f'{fcode} calibration data',
            **kwargs
        )

        # Define callback for closing the pop-up canvas
        def on_close():
            self.popup_canvas = None
            self.stop_action('iosweep_plot')
        
        # Show in dedicated pop-up canvas
        self.show_popup(
            fig, name='calibration data', on_close=on_close)
    
    @lock_actions(stoppable=False, unlock_on_finish=False)
    def on_field_plot(self, *args, **kwargs):
        # Ask for mapping file
        mapping_fpath = self.select_file_dialog(
            CSV_FILTER, initialdir=MAPPING_DATA_FOLDER, 
            title='Select mapping results files')
    
        # Abort if no file was chosen
        if mapping_fpath is None:
            logger.warning('no mapping file chosen')
            self.stop_action('field_plot')
            return

        # Plot mapping data 
        self.mapping_fpath = mapping_fpath
        self.plot_mapping_data(direct_call=True)
    
    @lock_actions(stoppable=False)
    def on_mp_resetorg(self, *args, **kwargs):
        ''' Callback for resetting the micro-manipulator origin '''
        if self.testmode:
            logger.info('resetting micro-manipulator origin')
        else:
            self.instruments['mm'].obj.set_origin()

    def pre_mp_move_along_normal(self):
        ''' Pre-move-along-normal routine '''
        # Ask user for moving distance in dialog window
        dialog = SetMovingDistanceDialog(self, default_distance=DEFAULT_MOVEAWAY_DIST)

        # If cancel was pressed, return
        if not dialog.exec():
            logger.error('move aborted')
            return False

        # If OK was pressed, store distance and return True
        self.vnorm_move_distance = dialog.get_values()  # um
        return True
    
    @lock_actions(stoppable=False)
    def on_mp_move_along_normal(self, *args, **kwargs):
        '''
        Callback for moving the micro-manipulator along vector normal to scanning base plane
        by a fixed distance.
        '''
        if self.testmode:
            s = 'away from' if self.vnorm_move_distance > 0 else 'towards'
            logger.info(f'moving micro-manipulator {s} base plane by {self.vnorm_move_distance} um')
        else:
            # Move stage along normal vector to reference plane by specified distance
            self.get_scanner().move_along_normal_vector(self.vnorm_move_distance)
    
    def pre_mp_move_updown(self):
        ''' Pre-move up/down routine '''
        # Ask user for moving distance in dialog window
        dialog = SetMovingDistanceDialog(self, default_distance=DEFAULT_MOVEUPDOWN_DIST)

        # If cancel was pressed, return
        if not dialog.exec():
            logger.error('move aborted')
            return False

        # If OK was pressed, store distance and return True
        self.vertical_distance = dialog.get_values()
        return True
    
    @lock_actions(stoppable=False)
    def on_mp_move_updown(self, *args, **kwargs):
        '''
        Callback for moving the micro-manipulator up/down by a fixed distance.
        '''
        # Ask user for moving distance in dialog window
        if self.testmode:
            s = 'up' if self.vertical_distance > 0 else 'down'
            logger.info(f'moving micro-manipulator {s} by {np.abs(self.vertical_distance)} um')
        else:
            # Move stage up/down by specified distance
            try:
                self.get_scanner().move_updown(self.vertical_distance)
            except SutterError as e:
                logger.error(e)
                return

    def on_mp_save_pos(self, *args, **kwargs):
        ''' Callback for saving current micro-manipulator position '''
        # If test mode, log action and return
        if self.testmode:
            logger.info('saving current micro-manipulator position')
            return 
    
        # Extract current position from micro-manipulator
        pos = self.instruments['mm'].obj.get_position()  # in um
        logger.info(f'saving current micro-manipulator position: {pos} um')

        # Save position in memory
        update_last_position(pos)
    
    def on_mp_goto_saved_pos(self, *args, **kwargs):
        ''' Callback for moving micro-manipulator to saved position '''
        # If test mode, log action and return
        if self.testmode:
            logger.info('moving micro-manipulator to saved position')
            return
    
        # Extract saved position
        pos = get_last_position()  # in um
        if pos is None:
            return
        
        # Move micro-manipulator to saved position
        self.instruments['mm'].obj.set_position(pos)

    # ------------------- PROPERTIES, GETTERS, AND STATES -------------------

    @property
    def f0(self):
        ''' Fundamental frequency (Hz) '''
        return self.stim_params_dict[F_KEY].value() * MHZ_TO_HZ
    
    @property
    def wgen_channels(self):
        return [x.value() for x in self.stim_params_dict['channels [trig.] [sig.]']]
    
    @property
    def wch_trig(self):
        ''' Waveform generator trigger channel index '''
        return self.wgen_channels[0]
    
    @property
    def wch_sig(self):
        ''' Waveform generator signal channel index '''
        return self.wgen_channels[1]
    
    @property
    def ref_Vpp(self):
        ''' Reference voltage peak-to-peak (Vpp) '''
        return self.stim_params_dict[VIN_KEY].value() * MV_TO_V
    
    @property
    def ncycles(self):
        ''' Number of cycles per pulse '''
        return self.stim_params_dict[NCYCLES_PER_PULSE_KEY].value()
    
    @property
    def PRF(self):
        ''' Pulse repetition frequency (Hz) '''
        return self.stim_params_dict[PRF_KEY].value()
    
    @property
    def att_gain(self):
        ''' Attenuation probe loss (dB) '''
        return self.acq_params_dict['probe attenuation (dB)'].value()
    
    @property
    def Ml(self):
        ''' Hydrophone sensitivity (V/MPa) '''
        return self.acq_params_dict['hydrophone sensitivity (V/MPa)'].value() * PA_TO_MPA
    
    @property
    def scope_channels(self):
        return [x.value() for x in self.acq_params_dict['channels [trig.] [sig.] [cplfwd.] [cplrev.]']]
    
    @property
    def sch_trig(self):
        ''' Oscilloscope trigger channel index '''
        val = self.scope_channels[0]
        if val == 0:
            return None
        return val
    
    @property
    def sch_sig(self):
        ''' Oscilloscope signal channel index '''
        val = self.scope_channels[1]
        if val == 0:
            return None
        return val
    
    @property
    def sch_cpl_fwd(self):
        ''' Oscilloscope forward coupling channel index '''
        val = self.scope_channels[2]
        if val == 0:
            return None
        return val
    
    @property
    def sch_cpl_rev(self):
        ''' Oscilloscope reverse coupling channel index '''
        val = self.scope_channels[3]
        if val == 0:
            return None
        return val

    @property
    def acq_nsweeps(self):
        ''' Number of sweeps per oscilloscope acquisition '''
        return self.acq_params_dict['# sweeps / acq'].value()

    @property
    def acqinterval(self):
        ''' Inter-acquisition interval (s) '''
        return self.acq_params_dict['inter-acq int. (s)'].value()
    
    @property
    def acq_measure_signal(self):
        ''' Whether to measure physical signal '''
        return self.acq_opts_dict['measure signal'].isChecked()
    
    @property
    def acq_measure_coupling(self):
        ''' Whether to measure FWD/REV coupling '''
        return self.acq_opts_dict['measure coupling'].isChecked()
    
    @property
    def has_acq_type(self):
        ''' Whether at least 1 acquisition type is selected '''
        return self.acq_measure_signal or self.acq_measure_coupling
    
    @property
    def theta(self):
        ''' Dictionary of scanning angles per axis (in radians) '''
        return {'Y': self.scan_params_dict['XZ angle (°)'].value() * DEG_TO_RAD}
    
    @property
    def delta(self):
        ''' Scanning range per axis (um) '''
        return dict(zip('XYZ', [x.value() * MM_TO_UM for x in self.scan_params_dict['XYZ range (mm)']]))
    
    @property
    def resolution(self):
        ''' Scanning resolution (um) '''
        return self.scan_params_dict['resolution (mm)'].value() * MM_TO_UM
    
    @property
    def nperax(self):
        ''' Number of points per axis for fixed scanning procedures '''
        return {k: int(np.round(v / self.resolution)) + 1 for k, v in self.delta.items()}

    @property
    def niters(self):
        ''' Number of focus search iterations '''
        return self.scan_params_dict['# focus iters'].value()
    
    @property
    def calib_data(self):
        return self.fdata[self.acqmode]
    
    @property
    def has_calib_data(self):
        if self.calib_data is None:
            return False
        unit = 'V' if self.acqmode == 'amplifier' else 'MPa'
        return any(s.endswith(f'({unit})') for s in self.calib_data.columns)
    
    @property
    def has_coupling_data(self):
        if self.calib_data is None:
            return False
        return any(s.startswith('CPL') for s in self.calib_data.columns)
    
    @property
    def calib_fpath(self):
        return self.fpaths[self.acqmode]
    
    @property
    def has_acq_instr(self):
        return self.instruments['wg'].iscon and self.instruments['sc'].iscon
    
    @property
    def has_scan_instr(self):
        return self.instruments['mm'].iscon

    @property
    def has_last_pos(self):
        return get_last_position() is not None

    @property
    def acqmode(self):
        if self.fpaths['amplifier'] is not None:
            return 'amplifier'
        else:
            return 'transducer'
    
    def check_channels(self, chseq):
        ''' Check selected (i.e. non-zero) channels are different '''
        active_chseq = [x for x in chseq if x != 0]
        if len(active_chseq) != len(set(active_chseq)):
            raise ValueError(f'active channels sequence is non-unique: {chseq}')
    
    @property
    def has_correct_channel_sequences(self):
        ''' Check that selected channels are different '''
        try:
            self.check_channels(self.wgen_channels)
        except ValueError as e:
            logger.warning(f'waveform generator {e}')
            return False
        try:
            self.check_channels(self.scope_channels)
        except ValueError as e:
            logger.warning(f'oscilloscope {e}')
            return False
        return True
    
    @property
    def can_acquire(self):
        ''' Check whether acquisition is possible '''
        if not self.has_acq_type:
            return False
        if self.acqmode == 'amplifier':
            return self.has_acq_instr and self.fpaths['amplifier'] is not None
        else:
            return self.has_acq_instr and all([self.fpaths[k] is not None for k in ['transducer', 'hydrophone']])

    @property
    def can_scan(self):
        ''' Check whether scanning is possible '''
        if self.acqmode == 'amplifier':
            return False
        return self.can_acquire and self.has_acq_instr and self.has_scan_instr

    @property    
    def can_plot_iocurve(self):
        ''' Check whether plotting calibration curves is possible '''
        return self.calib_data is not None and len(self.calib_data) > 0

