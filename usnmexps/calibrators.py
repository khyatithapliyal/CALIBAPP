# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-05-03 10:18:57
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-04-09 18:53:51
# @Last Modified time: 2025-02-11 18:19:20
# this file is teh integration layer that prepares the hardware context and forwrads it to teh burst efficiency runner 
import time
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert, windows
from scipy.optimize import curve_fit
from tkinter import TclError
from pyvisa.errors import VisaIOError

from instrulink import logger, SutterError, VisaError

from .constants import *
from .dialog import askyesno_dialog
from .si_utils import si_format
from .utils import apply_rolling_window, bounds, filter_signal, nan_like, redraw, is_within, idxmax, today, make_gauss, project_onto_plane, split_name_and_unit
from .calib_utils import check_input_voltages, vratio_to_gain
from .scanners import CrossSearchScanner, GridScanner, AdaptiveScanner
from .efficiency_monitor import EfficiencyMonitor, get_coupling_reference_from_calibration
from .burst_efficiency_runner import BurstEfficiencyRunner
from .check_acquisition_rate import check_scope_acquisition_rate

class Calibrator:

    FBOUNDS = (0.3e6, 5e6)  # carrier frequency bounds (Hz)
    REL_TWINDOW = 1.1  # relative scope temporal window w.r.t. waveform duration
    REL_TDELAY = 7.  # relative scope trigger delay w.r.t. time division
    DEFAULT_VDIV = .01  # default vertical division on scope signal channel (V/div)
    DEFAULT_CPL_VDIV = 0.10  # default vertical division on scope coupling channels (V/div)
    MIN_IO_DELAY = 0.  # minimum intrinsic input to output delay (s)
    RELAX_TIME = 0.  # intrinsic output relaxation time (s)
    LOG_DELIMITER = ','
    STIM_CORRECTION = None  # what to do with stimulus electrical artifact ('remove', 'fit' or None)
    TEST_SR = 100 * MHZ_TO_HZ  # dummy sample rate for testing purposes (Hz)
    SIG_KEY = 'sig'
    CPL_KEY = 'cpl'
    CPL_FWD_KEY = 'cpl_fwd'
    CPL_REV_KEY = 'cpl_rev'
    ACQ_RATE_CHECK_DONE = False

    def __init__(self, wg, scope, f, wch_sig, sch_sig, 
                 wch_trig=None, sch_trig=None, sch_cpl_fwd=None, sch_cpl_rev=None,
                 ncycles=NCYCLES_CALIB, PRF=DEFAULT_PRF, acq_npoints=OSC_ACQ_NPOINTS, acq_nsweeps=OSC_ACQ_NSWEEPS, 
                 gate_type='trig', trigger_mode='loop', beep_on_trigger=False, 
                 sch_sig_vdiv=None, sch_cpl_vdiv=None, detect_vscale=False, output_conv_constant=1., 
                 ykey='output voltage (V)', testmode=False, plot=True,
                 canvas=None):
        '''
        Initialization
        
        :param wg: waveform generator object
        :param scope: oscilloscope object
        :param f: carrier frequency (Hz)
        :param wch_sig: waveform generator signal channel index
        :param sch_sig: oscilloscope signal channel index
        :param wch_trig: waveform generator trigger channel index
        :param sch_trig: oscilloscope trigger channel index
        :param sch_cpl_fwd: oscilloscope forward channel index (for energy tranduction efficiency measurements)
        :param sch_cpl_rev: oscilloscope reverse channel index (for energy tranduction efficiency measurements)
        :param ncycles: number of cycles per burst
        :param PRF: pulse repetition frequency (Hz)
        :param acq_npoints: number of samples per acquisition
        :param acq_nsweeps: number of sweeps per acquisition
        :param gate_type: generator gating type ('trig' for trigger or 'mod' for modulation)
        :param trigger_mode: generator trigger mode ('loop' for looping or 'prog' for programmatic trigger)
        :param beep_on_trigger: whether to beep upon generator trigger (only for single trigger mode)
        :param sch_sig_vdiv (optional): vertical division scale on scope signal channel
        :param sch_cpl_vdiv (optional): vertical division scale on scope coupling (forward and reversed) channels
        :param detect_vscale: whether to run scope auto-scale to detect signal (and coupling) channel(s) amplitude(s)
        :param output_conv_constant: output conversion constant
        :param ykey: output variable unit
        :param testmode: whether to run in test mode (no actual instrument communication)
        :param plot: whether to display waveform(s) in real-time
        :param canvas: canvas object to display plot(s). If None, a new figure window will be created.
        '''
        self.wg = wg
        self.scope = scope
        self.f = f
        self.wch_sig = wch_sig
        self.sch_sig = sch_sig
        self.wch_trig = wch_trig
        self.sch_trig = sch_trig
        self.sch_cpl_fwd = sch_cpl_fwd
        self.sch_cpl_rev = sch_cpl_rev
        if not self.has_coupling_channels and not self.has_signal_channel:
            raise ValueError('at least one scope signal (or coupling) channel must be specified')
        self.ncycles = ncycles
        self.PRF = PRF
        self.acq_npoints = acq_npoints
        self.acq_nsweeps = acq_nsweeps 
        self.gate_type = gate_type
        self.trigger_mode = trigger_mode
        self.beep_on_trigger = beep_on_trigger
        self.output_conv_constant = output_conv_constant
        if sch_sig_vdiv is None:
            sch_sig_vdiv = self.DEFAULT_VDIV
        self.sch_sig_vdiv = sch_sig_vdiv
        if sch_cpl_vdiv is None:
            sch_cpl_vdiv = self.DEFAULT_CPL_VDIV
        self.sch_cpl_vdiv = sch_cpl_vdiv
        self.detect_vscale = detect_vscale
        self.ykey = ykey
        self.fig = None
        self.testmode = testmode
        self.plot = plot
        self.canvas = canvas
        if self.testmode:
            self.tmock, self.ymock = self.generate_mock_waveform()
            self.Amock = 1.

    @property
    def f(self):
        return self._f
    
    @f.setter
    def f(self, value):
        if not is_within(value, self.FBOUNDS):
            s = ', '.join([f'{si_format(f, 3)}Hz' for f in self.FBOUNDS])
            raise ValueError(f'frequency value ({si_format(value, 3)}Hz) is outside of predefined interval ({s})')
        self._f = value
    
    @property
    def ykey(self):
        return self._ykey

    @ykey.setter
    def ykey(self, val):
        self.yname, self.yunit = split_name_and_unit(val)
        self._ykey = val
        
    @property
    def sch_channels(self):
        ''' List of relevant scope channels '''
        l = []
        if self.sch_sig is not None:
            l.append(self.sch_sig)
        if self.sch_trig is not None:
            l.append(self.sch_trig)
        if self.sch_cpl_fwd is not None:
            l.append(self.sch_cpl_fwd)
        if self.sch_cpl_rev is not None:
            l.append(self.sch_cpl_rev)
        return l
    
    @property
    def has_signal_channel(self):
        ''' Check if scope signal channel is specified '''
        return self.sch_sig is not None
    
    @property
    def has_coupling_channels(self):
        ''' Check if scope coupling channels are specified '''
        if self.sch_cpl_fwd is None:
            return False
        if self.sch_cpl_rev is None:
            return False
        return True
        
    def check_sample_rate(self):
        ''' Check that sampling rate allows to grab enough values per cycle '''
        sr = self.scope.get_sample_rate()  # oscilloscope sampling rate (Hz)
        nsamples_per_cycle = sr / self.f  # number of samples per cycle
        if nsamples_per_cycle < MIN_NSAMPLES_PER_CYCLE:  
            raise ValueError(
                f'sampling rate ({si_format(sr, 3)}Hz) is too low w.r.t. frequency value ({si_format(self.f, 3)}Hz). Consider reducing the acquisition time scale.')
        logger.info(f'sample rate = {si_format(sr, 3)}Hz')

    def check_acquisition_rate(self, channels=None, n_trials=20, acq_interval=None):
        ''' Measure scope waveform acquisition time and effective refresh rate. '''
        if channels is None:
            channels = self.sch_channels
        return check_scope_acquisition_rate(
            self.scope,
            channels,
            n_trials=n_trials,
            acq_interval=acq_interval,
        )

    def enable_generator(self):
        ''' Enable genator output on specified channel(s) '''
        self.wg.enable_output_channel(self.wch_sig)
        if self.wch_trig is not None:
            self.wg.enable_output_channel(self.wch_trig)

    def disable_generator(self):
        ''' Disable genator output on specificed channel(s) '''
        self.wg.disable_output_channel(self.wch_sig)
        if self.wch_trig is not None:
            self.wg.disable_output_channel(self.wch_trig)
    
    @property
    def wch_seed(self):
        ''' Seed channel for waveform generator '''
        return self.wch_trig if self.wch_trig is not None else self.wch_sig

    def set_generator(self, Vpp, ncycles, PRF):
        '''
        Set appropriate waveform parameters on the waveform generator
        
        :param Vpp: voltage amplitude (Vpp)
        :param ncycles: number of cycles per burst
        :param PRF: pulse repetition frequency (Hz)
        '''
        # Test mode: return
        if self.testmode:
            return
        
        # Set waveform generator parameters with reference Vpp value
        self.wg.set_looping_sine_burst(
            self.wch_sig, 
            self.f, 
            Vpp=Vpp, 
            ncycles=ncycles, 
            PRF=PRF,
            ich_trig=self.wch_trig,
            gate_type=self.gate_type)
        
        # If specified, set trigger channel to "manual" trigger mode
        if self.trigger_mode == 'prog':
            logger.info(f'switching {self.wch_seed} trigger mode to manual')
            self.wg.set_trigger_source(self.wch_seed, 'MAN')
    
    def get_scope_tdiv(self, dur):
        ''' 
        Get target scope temporal scale according to
        expected waveform duration
        
        :param dur: expected waveform duration
        :return: propsective scope time division (s/div)
        '''
        twindow = self.REL_TWINDOW * (dur + self.RELAX_TIME)  # display time window (s)
        return twindow / self.scope.NHDIVS  # time division (s/div)
    
    def set_scope_time_settings(self, tdiv):
        '''
        Set scope temporal scale and trigger delay
        
        :param tdiv: horizontal scale (s/div)
        '''
        # Set scope temporal scale
        self.scope.set_temporal_scale(tdiv)
        # Set scope trigger delay accordingly
        self.scope.set_relative_trigger_delay(self.REL_TDELAY)
    
    def set_scope_vertical_settings(self, vdiv_sig, vdiv_cpl):
        ''' 
        Set scope vertical scale and trigger settings 
        
        :param vdiv_sig: vertical division for signal channel (V/div)
        :param vdiv_cpl: vertical division for coupling channels (V/div)
        '''
        vscales = {
            self.sch_trig: 'TRIG',
        }
        if self.sch_sig is not None and vdiv_sig is not None:
            vscales[self.sch_sig] = vdiv_sig
        if self.sch_cpl_fwd is not None and vdiv_cpl is not None:
            vscales[self.sch_cpl_fwd] = vdiv_cpl
        if self.sch_cpl_rev is not None and vdiv_cpl is not None:
            vscales[self.sch_cpl_rev] = vdiv_cpl
        self.scope.set_multichannel_vscale(vscales)
    
    def detect_scope_vscale(self, exp_factor=1.):
        '''
        Run scope auto-setup to detect vertical scale on the signal (and coupling) channel(s).
        
        :param exp_factor: expansion factor for vertical division after auto-setup
        :return: new scope vertical division(s)
        '''
        # Run auto-setup
        self.scope.auto_setup()
        # Extract vertical scale from signal channel
        try:
            vdiv_sig, vdiv_cpl = None, None
            if self.has_signal_channel:
                vdiv_sig = self.scope.get_vertical_scale(self.sch_sig)  # v/div
                logger.info(f'detected vertical scale for signal channel = {vdiv_sig:.3f} V/div')
            if self.has_coupling_channels:
                vdiv_cpl_fwd = self.scope.get_vertical_scale(self.sch_cpl_fwd)
                vdiv_cpl_rev = self.scope.get_vertical_scale(self.sch_cpl_rev)
                vdiv_cpl = max(vdiv_cpl_fwd, vdiv_cpl_rev)
                logger.info(f'detected vertical scale for coupling channels = {vdiv_cpl:.3f} V/div')
        except VisaIOError:
                raise VisaError(f'oscilloscope reached {self.scope.timeout / S_TO_MS:.1f} s timeout during auto-setup')
        # If specified, adjust vertical scale to expand/reduce vertical range
        if exp_factor != 1.:
            if vdiv_sig is not None:
                vdiv_sig *= exp_factor
                logger.info(f'expanding signal vertical scale to {vdiv_sig:.3f} V/div')
            if vdiv_cpl is not None:
                vdiv_cpl *= exp_factor
                logger.info(f'expanding coupling vertical scale to {vdiv_cpl:.3f} V/div')
        
        # Return vertical divisions
        return vdiv_sig, vdiv_cpl
    
    def set_scope(self, ncycles, acq_npoints, acq_nsweeps, **kwargs):
        '''
        Set the oscilloscope parameters

        :param ncycles: number of cycles per burst
        :param acq_npoints: number of samples per acquisition
        :param acq_nsweeps: number of sweeps per acquisition
        '''
        if self.testmode:
            return 1
        # Set oscilloscope parameters: channels display & attenuation factors
        logger.info(f'restricting display to channel(s) {self.sch_channels}')
        self.scope.restrict_traces(self.sch_channels)
        for ich in self.sch_channels:
            self.scope.set_probe_attenuation(ich, 1)
        
        # Compute waveform duration and scope time division
        tstim = ncycles / self.f  # (s)
        tdiv = self.get_scope_tdiv(tstim)  # s/div
        
        # Set scope time & vertical settings
        self.set_scope_time_settings(tdiv)
        self.set_scope_vertical_settings(self.sch_sig_vdiv, self.sch_cpl_vdiv)
        
        # If specified
        if self.detect_vscale:
            # Auto-detect vertical scale on relevant (i.e. non-trigger) channels
            vdiv_sig, vdiv_cpl = self.detect_scope_vscale(exp_factor=kwargs.pop('exp_factor', 1))            
            # Update scope vertical settings accordingly 
            self.set_scope_vertical_settings(vdiv_sig, vdiv_cpl)
            # Reset scope time settings
            self.set_scope_time_settings(tdiv)
        
        # Extract scope final vertical scales
        if self.has_signal_channel:
            vdiv_sig = self.scope.get_vertical_scale(self.sch_sig)
            logger.info(f'final vertical scale for reference signal: {si_format(vdiv_sig)}V/div')
        else:
            vdiv_sig = None
        if self.has_coupling_channels:
            vdiv_cpl_fwd = self.scope.get_vertical_scale(self.sch_cpl_fwd)
            vdiv_cpl_rev = self.scope.get_vertical_scale(self.sch_cpl_rev)
            vdiv_cpl = max(vdiv_cpl_fwd, vdiv_cpl_rev)
            logger.info(f'final vertical scale for coupling channels: {si_format(vdiv_cpl)}V/div')
        else:
            vdiv_cpl = None
            
        # Set low-pass filter on scope signal channel, if specified
        if self.REL_FLIMS is not None:
            if self.REL_FLIMS[1] < np.inf:
                if self.has_signal_channel:
                    self.scope.set_filter(
                        self.sch_sig, 'LP', fhigh=self.f * self.REL_FLIMS[1])
                    self.scope.enable_filter(self.sch_sig)
                if self.has_coupling_channels:
                    self.scope.set_filter(
                        self.sch_cpl_fwd, 'LP', fhigh=self.f * self.REL_FLIMS[1])
                    self.scope.enable_filter(self.sch_cpl_fwd)
                    self.scope.set_filter(
                        self.sch_cpl_rev, 'LP', fhigh=self.f * self.REL_FLIMS[1])
                    self.scope.enable_filter(self.sch_cpl_rev)
        
        # Check that sampling rate allows to acquire waveform details
        self.check_sample_rate()

        # Set a specific number of points & sweeps per acquisition
        self.scope.set_waveform_settings(npoints=acq_npoints)
        self.scope.set_nsweeps_per_acquisition(acq_nsweeps)
        self.scope.hide_menu()

        # Run a one-time practical acquisition/refresh rate benchmark.
        if not self.testmode and not Calibrator.ACQ_RATE_CHECK_DONE:
            try:
                self.check_acquisition_rate(
                    channels=self.sch_channels,
                    n_trials=10,
                    acq_interval=OSC_ACQ_INTERVAL,
                )
            except Exception as e:
                logger.warning(f'acquisition-rate check skipped: {e}')
            finally:
                Calibrator.ACQ_RATE_CHECK_DONE = True

        # Return scope vertical division for signal and coupling channels
        return {self.SIG_KEY: vdiv_sig, self.CPL_KEY: vdiv_cpl}

    def set_for_waveform(self, Vpp, **kwargs):
        ''' 
        Set signal generator and oscilloscope parameters to generate and acquire waveform
        
        :param Vpp: voltage amplitude (Vpp)
        :return: nominal burst duration & scope vertical division
        '''
        # Compute waveform duration
        tstim = self.ncycles / self.f  # s

        # Prepare waveform generator and oscilloscope
        self.set_generator(Vpp, self.ncycles, self.PRF)
        vdivs = self.set_scope(self.ncycles, self.acq_npoints, self.acq_nsweeps, **kwargs)

        # Return burst duration and scope vertical divisions
        return tstim, vdivs

    def get_tbounds(self, dur):
        '''
        Get signal temporal boundaries for a prospective waveform duration
        
        :param dur: propesctive waveform duration (s)
        :return: projected signal boundaries (s)
        '''
        tbounds = np.array([0., dur])
        if self.sch_trig is not None:
            tbounds += self.MIN_IO_DELAY
        return tbounds
    
    @staticmethod
    def extract_envelope(y, navg=1, ntrim=0):
        ''' Extract envelope from sinusoidal signal '''
        # Extract envelope as amplitude of Hilbert transform
        yenv = np.abs(hilbert(y))
        # If specified, apply moving average to smooth envelope
        if navg > 1:
            yenv = apply_rolling_window(yenv, navg)
        # If specified, trim off envelope edges
        if ntrim > 0:
            yenv[:ntrim] = np.nan
            yenv[-ntrim:] = np.nan
        # Return processed envelope
        return yenv
    
    def get_sine_burst(self, t, dur, amp=1., delay=0., offset=0., phi=np.pi):
        '''
        Get sine burst with specific parameters matching a given time vector
        
        :param t: time vector (s)
        :param dur: signal duration (s)
        :param amp: signal amplitude
        :param delay: signal delay (s)
        :param offset: signal vertical offset (during burst)
        :param phi: sinusoidal phase (rad)
        :return: signal vector
        '''
        y = np.zeros(t.size)
        iburst = np.where(np.logical_and(t > delay, t < dur + delay))[0]
        y[iburst] = np.sin(2 * np.pi * self.f * (t[iburst] - delay) + phi)
        return y * amp + offset

    def fit_sine_burst(self, t, y, tstim):
        ''' Fit sine burst to drive readout '''
        # Define sine burst function with duration fixed to tstim
        fitfunc = lambda t, A, d, o: self.get_sine_burst(
            t, tstim, amp=A, delay=d, offset=o)
        # Define initial function parameters
        A0 = np.ptp(y) / 2  # amplitude: half of peak-to-peak
        delay0 = t[np.where(np.abs(y) > 0.9 * A0)[0][0]]  # time to first peak
        delay0 -= 0.2 / self.f  # retract 1/4 of cycle
        offset0 = 0.  # offset: 0
        p0 = (A0, delay0, offset0)
        popt, _ = curve_fit(fitfunc, t, y, p0=p0)
        yfit = fitfunc(t, *popt)
        return yfit

    def process_waveform(self, t, y, tstim, verbose=True, stim_correction=None, t_amp_detect=None):
        '''
        Process acquired waveform
        
        :param t: time vector
        :param y: zero mean sinusoidal signal
        :param tstim: stimulus duration (s)
        :param verbose: whether to output details of intermediate processed traces
        :param stim_correction (optional): stimulus artifact correction method
        :param t_amp_detect (optional): time at which to detect waveform amplitude. If None, the maximum of the envelope is used.
        :return: 2-tuple with:
            - dictionary of processed traces (including envelope)
            - envelope amplitude
        '''
        # Store raw trace
        ytraces = {'raw': y}
    
        # Apply filtering if specified
        if self.REL_FLIMS is not None:
            if self.testmode:
                sr = self.TEST_SR
            else:
                sr = self.scope.get_sample_rate()  # oscilloscope sampling rate (Hz)
            fc = self.f * np.asarray(self.REL_FLIMS)
            ytraces['filt'] = filter_signal(y, sr, fc, verbose=False)
            y = ytraces['filt']
    
        # Apply stimulus artifact correction, if specified
        if stim_correction is not None:
            # Method 1: fit and subtract sine burst
            if stim_correction == 'fit':
                yfit = self.fit_sine_burst(t, y, tstim)
                ytraces['corr'] = y - yfit
                y = ytraces['corr']
            # Method 2: set artifact window to zero
            elif stim_correction == 'remove':
                iartifact = np.where(is_within(t, (0., tstim + 1e-6)))[0]
                ytraces['corr'] = y.copy()
                ytraces['corr'][iartifact] = 0.
                y = ytraces['corr']
            else:
                raise ValueError(
                    f'invalid stim artifact correction: "{stim_correction}"')

        # Optional: restrict traces to processed waveform
        if not verbose:
            ytraces = {'waveform': y}

        # Extract envelope from reference signal
        ytraces['env'] = self.extract_envelope(y, navg=NAVG_PENV, ntrim=NTRIM_PENV)
        
        # Compute amplitude from envelope, either at specified time or at peak
        if t_amp_detect is not None:
            tbounds = bounds(t)
            if not is_within(t_amp_detect, tbounds):
                logger.warning(f't_amp_detect ({si_format(t_amp_detect, 2)}) is outside of time vector bounds ({si_format(bounds(t), 2)}) -> clipping to closest bound')
                tdelta = tbounds[1] - tbounds[0]
                tmid = tbounds[0] + tdelta / 2
                if t_amp_detect < t.min():
                    t_amp_detect = tmid - 0.45 * tdelta
                elif t_amp_detect > t.max():
                    t_amp_detect = tmid + 0.45 * tdelta
            yamp = np.interp(t_amp_detect, t, ytraces['env'])
        else:
            yamp = np.nanmax(ytraces['env'])
        
        # Return traces & amplitude
        return ytraces, yamp
    
    def generate_mock_waveform(self, amp=1.):
        ''' 
        Generate a mock waveform for testing purposes
        
        :param amp: amplitude of the mock waveform
        :return: time and waveform vectors
        '''
        # Construct time vector
        t = np.arange(3000) / self.TEST_SR  # s
        # Construct basic sinusoid
        y = np.sin(2 * np.pi * self.f * t) * amp
        # Construct envelope
        yenv = np.hstack([
            np.zeros(500),
            windows.tukey(2000),
            np.zeros(500)]
        ) * amp
        # Return time and waveform vectors
        return t, y * yenv

    def acquire_and_process(self, tstim, trace_out_fpath=None, **kwargs):
        '''
        Acquire and process a waveform

        :param tstim: stimulus duration (s)
        :param trace_out_fpath: path to output file where to save trace(s) (optional)
        :return: waveform envelope amplitude and coupling channel amplitudes (if applicable)
        '''
        # If "prog" trigger mode: beep, trigger generator and arm scope
        if self.trigger_mode == 'prog' and not self.testmode:
            if self.beep_on_trigger:
                self.wg.beep()
            logger.info('triggering generator and arming oscilloscope')
            self.wg.trigger(verbose=False)
            self.scope.arm_acquisition()

        # Test mode: generate mock sinusoid
        if self.testmode:
            t, yout = self.tmock, {self.SIG_KEY: self.ymock * self.Amock}
        
        # Normal mode: acquire waveform(s)
        else:
            yout = {}
            if self.has_signal_channel:
                t, yout[self.SIG_KEY] = self.scope.get_waveform_data(self.sch_sig)
            else:
                t = None
            if self.has_coupling_channels:
                t1, yout[self.CPL_FWD_KEY] = self.scope.get_waveform_data(self.sch_cpl_fwd)
                _, yout[self.CPL_REV_KEY] = self.scope.get_waveform_data(self.sch_cpl_rev)
                if t is None:
                    t = t1
        
        # Rescale output signal according to conversion constant
        if self.has_signal_channel:
            yout[self.SIG_KEY] = yout[self.SIG_KEY] / self.output_conv_constant
        
        # Subtract mean(s)
        yout = {k: v - v.mean() for k, v in yout.items()}

        # Initiailze output amplitude(s) and waveform(s) data dictionaries
        yamps, wf_data = {}, {}
        
        # Process signal channel waveform, if available
        if self.has_signal_channel:
            ytraces, yamps[self.SIG_KEY] = self.process_waveform(
                t, yout[self.SIG_KEY], tstim, stim_correction=self.STIM_CORRECTION, **kwargs)
            wf_data[self.SIG_KEY] = pd.DataFrame({TIME_US: t * S_TO_US, **ytraces})
        else:
            yamps[self.SIG_KEY], wf_data[self.SIG_KEY] = None, None
        
        # Process coupling channels waveforms, if available
        if self.has_coupling_channels:
            for k in [self.CPL_FWD_KEY, self.CPL_REV_KEY]:
                ytraces, yamps[k] = self.process_waveform(
                    t, yout[k], tstim, t_amp_detect=0.9 * tstim, **kwargs)
                wf_data[k] = pd.DataFrame({TIME_US: t * S_TO_US, **ytraces})            
            yamp_cpl_ratio = yamps[self.CPL_REV_KEY] / yamps[self.CPL_FWD_KEY]
            logger.debug(f'CPL amplitudes: FWD = {yamps[self.CPL_FWD_KEY]:.4f} V, REV = {yamps[self.CPL_REV_KEY]:.4f} V, ratio: {yamp_cpl_ratio:.3f}')
        else:
            for k in [self.CPL_FWD_KEY, self.CPL_REV_KEY]:
                yamps[k], wf_data[k] = None, None

        # If plot mode enabled
        if self.plot:
            # For each waveform type, if corresponding axis exists, update its data
            for k, v in wf_data.items():
                if v is not None and k in self.axobjs and self.axobjs[k] is not None:
                    self.update_waveform_plot(k, v.copy(), yamps[k])

        # If output file path provided, save signal traces there if applicable
        if trace_out_fpath is not None and self.has_signal_channel:
            wf_data[self.SIG_KEY].to_csv(trace_out_fpath, index=False)
        
        # Return dictionary of detected envelope amplitudes
        return yamps

    def init_fig(self, tstim, naxes=1, add_axis=True):
        ''' Initialize figure '''
        # Close existing figure, if any
        if self.fig is not None:
            plt.close(self.fig)
        # Initialize axes and line objects dictionaries
        self.axobjs = {}
        self.lineobjs = {}
        # If no canvas: Turn on interactive mode to see curves being populated
        if self.canvas is None:
            plt.ion()
        # Create figure with appropriate size
        self.fig = plt.figure(figsize=(6 * naxes, 5))
        # If canvas exists, assign figure to it and resize it
        if self.canvas is not None:
            self.canvas.figure = self.fig
            self.canvas.resize_figure()
        # Otherwise, set up callback event upon figure close
        else:
            def on_close(evt):
                logger.info('closing figure')
                self.terminate_procedure()
            self.fig.canvas.mpl_connect('close_event', on_close)
        # Initialize signal waveform axis, if specified
        if add_axis:
            self.axobjs[self.SIG_KEY] = self.fig.add_subplot(1, naxes, 1)
            self.init_waveform_plot(self.SIG_KEY, tstim)
    
    def init_waveform_plot(self, axkey, tstim, ykey=None):
        ''' Initialize waveform plot '''
        ax = self.axobjs[axkey]
        ax.set_title(f'{axkey} waveform')
        ax.set_xlabel(TIME_US)
        if ykey is None:
            ykey = self.ykey
        ax.set_ylabel(ykey)
        stimbounds = self.get_tbounds(tstim) * S_TO_US
        ax.axvspan(*stimbounds, ec=None, fc='C0', alpha=0.3, label='stimulus span')
        self.lineobjs[axkey] = None
        if self.canvas is not None:
            self.canvas.draw()

    def init_iocurve_plot(self, axkey, ylabel):
        ''' Initialize I/O curve plot '''
        ax = self.axobjs[axkey]
        ax.set_title(f'{axkey} I/O curve')
        ax.set_xlabel('peak-to-peak input voltage (V)')
        ax.set_ylabel(ylabel)
        self.lineobjs[axkey] = None
        if self.canvas is not None:
            self.canvas.draw()
    
    def init_fsweep_plot(self, axkey, ylabel):
        ''' Initialize frequency sweep plot '''
        ax = self.axobjs[axkey]
        ax.set_title(f'{axkey} f-sweep')
        ax.set_xlabel('frequency (MHz)')
        ax.set_ylabel(ylabel)
        ax.axvline(self.f / MHZ_TO_HZ, color='k', linestyle='--', label='ref. freq.')
        self.lineobjs[axkey] = None
        if self.canvas is not None:
            self.canvas.draw()

    def update_waveform_plot(self, axkey, wf_data, wamp):
        '''
        Update waveform plot
        
        :param ax: matplotlib axis object
        :param wf_data: waveform dataframe
        :param wamp: waveform envelope amplitude
        '''
        ax = self.axobjs[axkey]
        tplt = wf_data.pop(TIME_US)
        wenv = wf_data.pop('env')
        tamp = bounds(tplt)
        yamp = np.array([wamp, wamp])
        
        if self.lineobjs[axkey] is None:
            self.lineobjs[axkey] = {}
            for k, v in wf_data.items():
                self.lineobjs[axkey][k], = ax.plot(tplt, v, label=k)
            self.lineobjs[axkey]['envneg'], = ax.plot(tplt, -wenv, 'k--', label='envelope')
            self.lineobjs[axkey]['envpos'], = ax.plot(tplt, wenv, 'k--')
            self.lineobjs[axkey]['ampneg'], = ax.plot(tamp, -yamp, 'r--', label='amplitude')
            self.lineobjs[axkey]['amppos'], = ax.plot(tamp, yamp, 'r--')
            if len(self.axobjs) == 1:
                self.fig.subplots_adjust(right=.7)
                ax.legend(bbox_to_anchor=(1.05, .5), loc='center left')
        else:
            for k, v in wf_data.items():
                self.lineobjs[axkey][k].set_data(tplt, v)
            self.lineobjs[axkey]['envneg'].set_data(tplt, -wenv)
            self.lineobjs[axkey]['envpos'].set_data(tplt, wenv)
            self.lineobjs[axkey]['ampneg'].set_data(tamp, -yamp)
            self.lineobjs[axkey]['amppos'].set_data(tamp, yamp)
            
            # Update view only if max trace value is larger than y-axis bounds
            # or if time range is significantly different than current x-axis range
            pmax = wf_data.abs().max().max()
            trange_plt = np.diff(ax.get_xlim())[0] / 1.05
            trange_ratio = trange_plt / np.diff(tamp)[0]
            if pmax > ax.get_ylim()[1] or np.abs(np.log10(trange_ratio)) > .1:
                ax.relim()
                ax.autoscale_view()
        if self.canvas is not None:
            self.canvas.draw()
    
    def update_iocurve_plot(self, axkey, x, y):
        ''' Update I/O curve plot '''
        ax = self.axobjs[axkey]
        if self.lineobjs[axkey] is None:
            self.lineobjs[axkey], = ax.plot(x, y, marker='o', label='data')
            ax.legend()
        else:
            self.lineobjs[axkey].set_ydata(y)
            ax.relim()
            ax.autoscale_view()
        if self.canvas is not None:
            self.canvas.draw()
        
    def update_fsweep_plot(self, axkey, f, y):
        ''' Update frequency sweep plot '''
        ax = self.axobjs[axkey]
        if self.lineobjs[axkey] is None:
            self.lineobjs[axkey], = ax.plot(f / MHZ_TO_HZ, y, marker='o', label='data')
            ax.legend()
        else:
            self.lineobjs[axkey].set_ydata(y)
            ax.relim()
            ax.autoscale_view()
        if self.canvas is not None:
            self.canvas.draw()
    
    @property
    def isrunning(self):
        return self._isrunning
    
    @isrunning.setter
    def isrunning(self, value):
        self._isrunning = value
        if self.scanner is not None:
            self.scanner.continue_scan = value
    
    def terminate_procedure(self, keepfig=False):
        ''' Generic function called upon procedure termination '''
        logger.info(f'ending acquisition')
        self.isrunning = False
        # Disable generator, if not in test mode
        if not self.testmode:
            # Turn off generator output
            self.disable_generator()
        # If figure exists
        if self.plot:
            # If no canvas: 
            if self.canvas is None:
                # Turn off interactive mode
                plt.ioff()
                # Show non-interactive figure if specified
                if keepfig:
                    plt.show()
        # If "prog" trigger mode, reset scope trigger mode to "normal"
        if self.trigger_mode == 'prog' and not self.testmode:
            self.scope.set_normal_trigger()
    
    def run_acquisition(self, Vpp, acq_interval=OSC_ACQ_INTERVAL, adjust_scope_vscale=True, 
                        max_iter=None,**kwargs):
        ''' 
        Run continuous or finite acquisition
        
        :param Vpp: voltage amplitude (Vpp)
        :param acq_interval: acquisition interval (s)
        :param adjust_scope_vscale: whether to adjust the oscilloscope vertical scale to detected signal amplitude
        :param max_iter: maximum number of iterations (optional). If None, run indefinitely.
        :return: output amplitude and coupling channel amplitudes from last acquisition
        '''
        self.isrunning = True
        s = 'continous' if max_iter is None else f'{max_iter} iterations'
        logger.info(
            f'starting {s} acquisition at f = {si_format(self.f, 3)}Hz, Vpp = {si_format(Vpp)}V')
        # Set generator & scope for waveform
        tstim, _ = self.set_for_waveform(Vpp, **kwargs)

        # Unlock generator front panel
        if not self.testmode:
            self.wg.unlock_front_panel()
        
        # Initialize figure and axis
        if self.plot:
            naxes = 0
            if self.has_signal_channel:
                naxes += 1
            if self.has_coupling_channels:
                naxes += 2
            self.init_fig(tstim, naxes=naxes, add_axis=self.has_signal_channel)
            iax = 2 if self.has_signal_channel else 1
            if self.has_coupling_channels:
                for k in [self.CPL_FWD_KEY, self.CPL_REV_KEY]:
                    self.axobjs[k] = self.fig.add_subplot(1, naxes, iax)
                    self.init_waveform_plot(k, tstim, ykey='V')
                    iax += 1
                self.fig.subplots_adjust(wspace=.8)

        # Initalize infinite loop
        i = 0
        try:
            while self.isrunning:
                # Wait specific delay
                time.sleep(acq_interval)

                # Test mode: generate random waveform amplitude between .2 and .5
                if self.testmode:
                    self.Amock = np.random.uniform(.2, .5)
                    
                # Acquire and process waveform, and extract output amplitude(s)
                yamps = self.acquire_and_process(tstim)

                # If specified, adjust scope vertical scale(s) to detected signal amplitude(s)
                if not self.testmode and adjust_scope_vscale:
                    if self.has_signal_channel:
                        self.scope.adjust_vertical_scale(
                            self.sch_sig, yamps[self.SIG_KEY] * self.output_conv_constant)
                    if self.has_coupling_channels:
                        max_yamp_cpl = max(yamps[self.CPL_FWD_KEY], yamps[self.CPL_REV_KEY])
                        for ich in [self.sch_cpl_fwd, self.sch_cpl_rev]:
                            self.scope.adjust_vertical_scale(ich, max_yamp_cpl)

                # Redraw figure, if any
                if self.plot:
                    redraw(self.fig)

                # Log
                s = [f'acquisition {i + 1}']
                if self.has_signal_channel:
                    s.append(f'peak-to-peak {self.yname} = {2 * yamps[self.SIG_KEY]:.3f} {self.yunit}')
                if self.has_coupling_channels:
                    cplratio = yamps[self.CPL_REV_KEY] / yamps[self.CPL_FWD_KEY]
                    s.append(f'coupling ratio = {cplratio:.3f} ({vratio_to_gain(cplratio):.1f} dB)')
                logger.info(', '.join(s))

                # Increment iteration counter
                i += 1

                # If specified, stop after a certain number of iterations
                if max_iter is not None and i >= max_iter:
                    self.isrunning = False
        
        # Stop loop upon figure closing
        except TclError:
            logger.warning('interrupted!')
        
        # Terminate procedure
        self.terminate_procedure(keepfig=True)

        # Return detected output amplitude(s)
        return yamps

    def run_burst_efficiency_check(self, Vpp, BD=0.2, PRF=100, DC=50, n_bursts=5, 
                                    pulse_to_sample=10, acq_interval=1.0, plot=True,
                                    n_warmup=2):
        '''
        Run burst efficiency check via the reusable orchestration module.

        Public API is kept in Calibrator for backwards compatibility with app
        call sites, while implementation lives in BurstEfficiencyRunner.
        '''
        return BurstEfficiencyRunner(self).run(
            Vpp,
            BD=BD,
            PRF=PRF,
            DC=DC,
            n_bursts=n_bursts,
            pulse_to_sample=pulse_to_sample,
            acq_interval=acq_interval,
            plot=plot,
            n_warmup=n_warmup,
        )

    def _plot_burst_waveforms_envelope(self, waveform_data, yamps, pulse_to_sample, PRF, DC, t_pulse_mid):
        '''
        Backwards-compatible wrapper around reusable runner plotting.
        '''
        return BurstEfficiencyRunner(self).plot(
            waveform_data,
            yamps,
            pulse_to_sample=pulse_to_sample,
            PRF=PRF,
            DC=DC,
            t_pulse_mid=t_pulse_mid,
        )

    def profile_burst_pulse_stability(self, Vpp, BD=0.2, PRF=100, DC=50, n_bursts=1,
                                      plot=True, n_warmup=0):
        '''
        Run pulse-indexed pulse-by-pulse amplitude profiling via BurstEfficiencyRunner.
        '''
        return BurstEfficiencyRunner(self).profile_pulse_stability(
            Vpp,
            BD=BD,
            PRF=PRF,
            DC=DC,
            n_bursts=n_bursts,
            plot=plot,
            n_warmup=n_warmup,
        )

    def run_sweep(self, X, xkey, xunit, ref_Vpp, input_set_func, 
                  sweep_plot_init_func=None, sweep_plot_update_func=None,
                  acq_interval=OSC_ACQ_INTERVAL, npercond=NPERCOND_CALIB, 
                  adjust_scope_vscale=True,
                  **kwargs):
        '''
        Generic function to run input parameter sweep and extract output amplitudes.
        
        :param X: input parameter vector
        :param ref_Vpp: reference input voltage (Vpp) used to calibrate the oscilloscope
        :param sweep_plot_init_func: function to initialize sweep plot axis (optional, must be provided if plot=True)
        :param sweep_plot_update_func: function to update sweep plot axis (optional, must be provided if plot=True)
        :param acq_interval: acquisition interval (s)
        :param npercond (optional): number of acquisitions for each condition
        :param adjust_scope_vscale: whether to adjust the vertical scale of the oscilloscope
            signal channel for each new input signal amplitude of the generator
        :return: vector of peak-to-peak output amplitudes 
        '''
        # Set running flag
        self.isrunning = True
        
        # Get sweep size
        n = X.size

        # Set generator & scope for waveform
        tstim, vdivs = self.set_for_waveform(ref_Vpp, **kwargs)

        # Get ratio of scope vertical scale(s) over reference Vpp
        scope_ratios = {k: vdiv / ref_Vpp if vdiv is not None else None for k, vdiv in vdivs.items()}

        # Initialize figure and axes
        if self.plot:
            if sweep_plot_init_func is None:
                raise ValueError('sweep_plot_init_func must be provided if plot=True')
            if sweep_plot_update_func is None:
                raise ValueError('sweep_plot_update_func must be provided if plot=True')
            
            # Determine number of axes
            naxes = 0
            # if signal channel exists, add 2 axes (waveform and sweep curve)
            if self.has_signal_channel:
                naxes += 2
            
            # If coupling channels exist
            if self.has_coupling_channels:
                # If signal channel exists, add only 1 axis (CPL sweep curve)
                if self.has_signal_channel:
                    naxes += 1
                # Otherwise, add 3 axes (forward and reverse coupling waveforms, and CPL sweep curve)
                else:
                    naxes += 3

            # Initialize figure
            self.init_fig(tstim, naxes=naxes, add_axis=False)
            iax = 1

            # If signal channel exists, initialize signal waveform and sweep curve plots
            if self.has_signal_channel:
                self.axobjs[self.SIG_KEY] = self.fig.add_subplot(1, naxes, iax)
                self.init_waveform_plot(self.SIG_KEY, tstim)
                iax += 1
                self.axobjs['sig_sweep'] = self.fig.add_subplot(1, naxes, iax)
                sweep_plot_init_func('sig_sweep', f'peak-to-peak {self.ykey}')
                iax += 1
            # Othwerwise, initialize coupling channel waveforms plots
            else:
                for k in [self.CPL_FWD_KEY, self.CPL_REV_KEY]:
                    self.axobjs[k] = self.fig.add_subplot(1, naxes, iax)
                    self.init_waveform_plot(k, tstim, ykey='V')
                    iax += 1
            
            # If coupling channels exist, initialize coupling ratio sweep curve plot
            if self.has_coupling_channels:
                self.axobjs['cpl_ratio'] = self.fig.add_subplot(1, naxes, iax)
                sweep_plot_init_func('cpl_ratio', CPLRATIODB_KEY)
            
            # If more than 2 axes, adjust horizontal space between them
            if naxes > 2:
                self.fig.subplots_adjust(wspace=.8)        
        
        # If specified, adjust acquisition vertical scale(s) to detected signal amplitude
        if adjust_scope_vscale and not self.testmode:
            yamps = self.acquire_and_process(tstim)
            if scope_ratios[self.SIG_KEY] is not None:            
                vdivs[self.SIG_KEY] = self.scope.adjust_vertical_scale(
                    self.sch_sig, yamps[self.SIG_KEY] * self.output_conv_constant)
                scope_ratios[self.SIG_KEY] = vdivs[self.SIG_KEY] / ref_Vpp
            if scope_ratios[self.CPL_KEY] is not None:
                max_yamp_cpl = max(yamps[self.CPL_FWD_KEY], yamps[self.CPL_REV_KEY])
                vdiv_cpl_fwd = self.scope.adjust_vertical_scale(self.sch_cpl_fwd, max_yamp_cpl)
                vdiv_cpl_rev = self.scope.adjust_vertical_scale(self.sch_cpl_rev, max_yamp_cpl)
                vdivs[self.CPL_KEY] = max(vdiv_cpl_fwd, vdiv_cpl_rev)
                scope_ratios[self.CPL_KEY] = vdivs[self.CPL_KEY] / ref_Vpp
        
        # Define vectors of output peak-to-peak signal and coupling amplitudes,
        # if applicable
        outkeys = []
        if self.has_signal_channel:
            outkeys.append(self.SIG_KEY)
        if self.has_coupling_channels:
            outkeys = outkeys + [self.CPL_FWD_KEY, self.CPL_REV_KEY]
        yp2ps = {k: nan_like(X) for k in outkeys}
        
        # Run through input values
        try:
            for i, x in enumerate(X):
                # If procedure was interrupted, exit loop
                if not self.isrunning:
                    break

                # Call input setting function with current input value
                input_set_func(x)

                # If specified, adjust acquisition vertical scale according to 
                # cuurrent input value, assuming linear scaling
                if adjust_scope_vscale and not self.testmode and xkey == 'Vin':
                    if self.has_signal_channel:
                        self.scope.set_vertical_scale(self.sch_sig, x * scope_ratios[self.SIG_KEY], verbose=False)
                    if self.has_coupling_channels:
                        for ich in [self.sch_cpl_fwd, self.sch_cpl_rev]:
                            self.scope.set_vertical_scale(ich, x * scope_ratios[self.CPL_KEY], verbose=False)
                
                # Initialize arrays for applicable output signal amplitudes for each repetition,
                yamp_per_acq = {k: np.zeros(npercond) for k in yp2ps.keys()}

                # For each repetition
                for iacq in range(npercond):                    
                    # Wait specific delay
                    time.sleep(acq_interval)
                    
                    # Acquire and process waveform(s), update graph(s) if necessary,
                    # and extract output(s)
                    yamps = self.acquire_and_process(tstim)
                    for k, yamp in yamps.items():
                        if yamp is not None:
                            yamp_per_acq[k][iacq] = yamp
                        
                    # If specified, adjust scope vertical scale(s) to detected signal amplitude
                    if adjust_scope_vscale and not self.testmode:
                        if scope_ratios[self.SIG_KEY] is not None:            
                            vdivs[self.SIG_KEY] = self.scope.adjust_vertical_scale(
                                self.sch_sig, yamps[self.SIG_KEY] * self.output_conv_constant)
                            scope_ratios[self.SIG_KEY] = vdivs[self.SIG_KEY] / x
                        if scope_ratios[self.CPL_KEY] is not None:
                            max_yamp_cpl = max(yamps[self.CPL_FWD_KEY], yamps[self.CPL_REV_KEY])
                            vdiv_cpl_fwd = self.scope.adjust_vertical_scale(self.sch_cpl_fwd, max_yamp_cpl)
                            vdiv_cpl_rev = self.scope.adjust_vertical_scale(self.sch_cpl_rev, max_yamp_cpl)
                            vdivs[self.CPL_KEY] = max(vdiv_cpl_fwd, vdiv_cpl_rev)
                            scope_ratios[self.CPL_KEY] = vdivs[self.CPL_KEY] / x

                    # Redraw figure, if any
                    if self.plot:
                        redraw(self.fig)
                
                # For each available channel, update outputs list with average 
                # peak-to-peak amplitude value (only if below a certain CV)
                for k, ypa in yamp_per_acq.items():
                    ypa_cv = ypa.std() / ypa.mean()
                    if ypa_cv <= MAX_CV_CALIB:
                        yp2ps[k][i] = 2 * ypa.mean()
                    else:
                        logger.warning(
                            f'{k} amplitude variation coefficient is too high ({ypa_cv * 1e2:.1f} %) -> skipping')
                
                # Log
                s = [f'acquisition {i + 1}/{n} ({xkey} = {si_format(x, 2)}{xunit})']
                if self.has_signal_channel and not np.isnan(yp2ps[self.SIG_KEY][i]):
                    s.append(f'peak-to-peak {self.yname} = {yp2ps[self.SIG_KEY][i]:.3f} {self.yunit}')
                if self.has_coupling_channels and not np.isnan(yp2ps[self.CPL_FWD_KEY][i]) and not np.isnan(yp2ps[self.CPL_REV_KEY][i]):
                    cplratio = yp2ps[self.CPL_REV_KEY][i] / yp2ps[self.CPL_FWD_KEY][i]
                    s.append(f'coupling ratio = {cplratio:.3f} ({vratio_to_gain(cplratio):.1f} dB)')
                logger.info(', '.join(s))
                
                # Update I/O curve, if any
                if self.plot:
                    if 'sig_sweep' in self.axobjs:
                        sweep_plot_update_func('sig_sweep', X, yp2ps[self.SIG_KEY])
                    if 'cpl_ratio' in self.axobjs:
                        sweep_plot_update_func(
                            'cpl_ratio', X, vratio_to_gain(yp2ps[self.CPL_REV_KEY] / yp2ps[self.CPL_FWD_KEY]))

        # If figure was closed during process, terminate procedure and raise error
        except TclError:
            self.terminate_procedure()
            raise ValueError('process interrupted!')

        # Terminate procedure
        self.terminate_procedure()

        # Return output(s)
        return yp2ps
    
    def run_io_sweep(self, Vpps_in, ref_Vpp=REF_VPP_CALIB, **kwargs):
        '''
        Run an input-output sweep for a specific transducer and driving frequency.
        
        :param Vpps_in: vector of input driving voltage amplitudes (Vpp)
        :param ref_Vpp: reference input voltage (Vpp) used to calibrate the oscilloscope
        :return vector of peak-to-peak output amplitudes
        '''
        # Check input voltages
        check_input_voltages(Vpps_in)

        # Log
        logger.info(f'building I/O curve at f = {si_format(self.f, 3)}Hz')
        
        # Define input setter function
        def set_input_Vpp(Vpp):
            if self.testmode:
                self.Amock = Vpp / 10.
            else:
                self.wg.set_waveform_amp(self.wch_sig, Vpp)
        
        # If first element is 0, remove it
        has_zero = Vpps_in[0] == 0.
        if has_zero:
            Vpps_in = Vpps_in[1:]

        # Run sweep and extract output vector(s)
        yout = self.run_sweep(
            Vpps_in,
            'Vin', 
            'Vpp', 
            ref_Vpp, 
            set_input_Vpp, 
            sweep_plot_init_func=self.init_iocurve_plot, 
            sweep_plot_update_func=self.update_iocurve_plot,      
            **kwargs
        )

        # If first element was 0, add it back
        if has_zero:
            yout = {k: np.insert(y, 0, 0.) for k, y in yout.items()}
        
        # Return output vector(s)
        return yout

    def run_freq_sweep(self, Vpp, freqs, **kwargs):
        '''
        Run frequency sweep for a specific transducer and input voltage.
        
        :param Vpp: input driving voltage amplitude (Vpp)
        :param freqs: vector of driving frequencies (Hz)
        :return: vector of output amplitudes 
        '''
        # Log
        fmin, fmax = freqs.min(), freqs.max()
        logger.info(f'running frequency sweep within {si_format([fmin, fmax], 3)}Hz')

        # Define input setter function
        def set_input_freq(f):
            self.f = f
            if self.testmode:
                _, self.ymock = self.generate_mock_waveform()
            else:
                self.wg.set_waveform_freq(self.wch_sig, f)

        # Save current frequency
        fref = self.f
    
        # Run sweep and extract output vector(s)
        yout = self.run_sweep(
            freqs,
            'f', 
            'Hz', 
            Vpp, 
            set_input_freq, 
            sweep_plot_init_func=self.init_fsweep_plot, 
            sweep_plot_update_func=self.update_fsweep_plot,      
            **kwargs
        )

        # Reset frequency to reference value
        self.f = fref

        # Return output vector(s)
        return yout


class AmplifierCalibrator(Calibrator):
    ''' Amplifier calibrator '''

    REL_FLIMS = None
    STIM_CORRECTION = None

    @property
    def DEFAULT_VDIV(self):
        ''' default vertical division on scope signal channel (V/div) '''
        if self.output_conv_constant == 1:
            return 0.1
        else:
            return 2


class TransducerCalibrator(Calibrator):
    
    SCAN_VDIV_EXP_FACTOR = 2.  # expansion factor for vertical division after auto-setup during scanning procedures
    CALIB_VDIV_EXP_FACTOR = 1.  # expansion factor for vertical division after auto-setup during calibration procedures
    DEFAULT_VDIV = 0.01  # default vertical division (V/div)
    STIM_CORRECTION = None  #'remove'
    REL_FLIMS = (0.5, 7)  # multiples of carrier frequency at which to set the low and high cutoff frequencies
    MIN_IO_DELAY = 0. # 3e-6  # Minimum input-output delay (s)
    RELAX_TIME = 10e-6  # intrinsic output relaxation time (s)
    MAX_DIST_MM = 15.  # maximum distance between hydrophone and transducer (mm)
    MAX_PROP_DELAY = MAX_DIST_MM * 1e-3 / SPEED_OF_SOUND_WATER  # maximum input-output delay due to acoustic propagation (s)

    def __init__(self, *args, Ml=None, **kwargs):
        errstr = 'hydrophone-specific conversion constant ("Ml") must be provided'
        if Ml is None:
            raise ValueError(errstr)
        if 'output_conv_constant' in kwargs:
            raise ValueError(f'only the {errstr}')
        self.scanner = None
        super().__init__(
            *args, **kwargs, output_conv_constant=Ml / PA_TO_MPA, ykey=P_KEY)
    
    def get_scope_tdiv(self, dur):
        ''' 
        Get target scope temporal scale according to
        expected waveform duration
        
        :param dur: expected waveform duration
        :return: propsective scope time division (s/div)
        '''
        twindow = self.REL_TWINDOW * (dur + self.RELAX_TIME + self.MAX_PROP_DELAY)  # display time window (s)
        return twindow / self.scope.NHDIVS  # time division (s/div)
    
    def run_io_sweep(self, *args, **kwargs):
        return super().run_io_sweep(
            *args, exp_factor=self.CALIB_VDIV_EXP_FACTOR, **kwargs)
    
    @staticmethod
    def get_scan_vector(start, end, n):
        ''' Wrapper around numpy linspace that also handles the case where n = 1 '''
        if n == 1:
            return np.array([np.mean([start, end])])
        else:
            return np.linspace(start, end, n)
    
    @classmethod
    def get_scan_vectors(cls, delta, nperax, zstart='base'):
        '''
        Get XYZ vectors or relative scan coordinates
        
        :param delta: scanning extent per axis (um)
        :param nperax: dictionary of number of sampled coordinates per axis
        :return: XYZ vectors of relative scan coordinates (um)
        '''
        # Parse delta and nperax
        if isinstance(delta, int):
            dx, dy, dz = [delta] * 3
        else:
            dx, dy, dz = delta['X'], delta['Y'], delta['Z']
        if isinstance(nperax, int):
            nx, ny, nz = [nperax] * 3
        else:
            nx, ny, nz = nperax['X'], nperax['Y'], nperax['Z']
        # Generate vectors
        x = cls.get_scan_vector(-dx / 2, dx / 2, nx)  # X: centered around current X position
        y = cls.get_scan_vector(-dy / 2, dy / 2, ny)  # Y: centered around current Y position
        z = cls.get_scan_vector(0, dz, nz)  # Z: above current Z position
        if zstart == 'center':
            z -= dz / 2  # in center mode: center z around zero
        return x, y, z

    @property
    def scan_data(self):
        if self.scanner is not None:
            return self.scanner.data
        else:
            return None

    def log_header_row(self, log_fpath, row):
        ''' Write header row to log file '''
        with open(log_fpath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=self.LOG_DELIMITER)
            writer.writerow(row)
    
    def log_data_row(self, log_fpath, row):
        ''' Write data row to log file '''
        with open(log_fpath, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=self.LOG_DELIMITER)
            writer.writerow(row)

    def perform_scan(self, mp, Vpp, delta, nperax, theta, vbasis, scanner_class, zstart='base',
                     log_fpath=None, traces_dir=None, ask_pos=True, adjust_scope_vscale=False, **kwargs):
        '''
        Perform XYZ scan

        :param mp: micro-manipulator object
        :param Vpp: input driving voltage amplitude (Vpp)
        :param delta: scanning extent per axis (um)
        :param nperax: number of sampled coordinates per axis
        :param scanner_class: scanner class
        :param theta: dictionary of rotation angles of scanning system (rad)
        :param vbasis: vector base for scanner coordinate system
        :param log_fpath: path to output log file
        :param traces_dir: path to directory where to store output trace files  
        :param ask_pos: whether the use should be prompted to confirm the hyrophone prior to start
        :return: scan result
        '''
        self.isrunning = True
        # Ask whether the hydrophone is well positioned
        if zstart == 'base':
            startloc_str = 'right above the center of the transducer aperture'
        elif zstart == 'center':
            startloc_str = 'around the acoustic focus'
        question = f'Is the hydrophone positioned {startloc_str}?'
        is_positioned = askyesno_dialog(question) if ask_pos else True
        if not is_positioned:
            raise SutterError('Wrong hydrophone position')
        
        # If test mode, set micro-manipulator to None
        if self.testmode:
            mp = None

        # Gather reference position
        if mp is None:
            self.refpos = np.zeros(3)
        else:
            self.refpos = mp.get_position()
        logger.info(f'reference position: {self.refpos} um')

        # Set generator & scope for waveform
        tstim, _ = self.set_for_waveform(Vpp, exp_factor=self.SCAN_VDIV_EXP_FACTOR)
        
        # Initialize scanner
        self.scanner = scanner_class(mp, theta=theta, vbasis=vbasis, canvas=self.canvas)
        
        # Initialize figure and axes
        if self.plot:
            self.init_fig(tstim, naxes=2)
            self.scanner.ax = self.scanner.get_scan_plot(
                'XYZ', fig=self.fig, subplot=122)
            self.axobjs['scan'] = self.scanner.ax

        # Get scan x, y, z vectors
        x, y, z = self.get_scan_vectors(delta, nperax, zstart=zstart)

        # If neccessary, get number of positions and associated zero padding for traces saving
        if traces_dir is not None:
            npos = x.size * y.size * z.size
            npad = int(np.floor(np.log10(npos))) + 1

        # If log file path provided, create file
        if log_fpath is not None:
            self.log_header_row(log_fpath, ['X (um)', 'Y (um)', 'Z (um)', P_KEY])
        
        # If testmode
        if self.testmode:
            # Get gaussian funciton peaking within domain limits
            mygauss = make_gauss(
                x=x, 
                y=y,
                z=z, 
                rel_x0=0.55,
                rel_y0=0.45,
                rel_z0=0.3,
            )

            # Define function that inverse transforms physical coordinates prior to evaluation 
            Minv = self.scanner.get_transform().inverse()
            def amp_mock_feval(*coords):
                coords = Minv.apply(np.array(coords))
                return mygauss(*coords)
        
            # Assign as mock amplitude evaluation function            
            self.amp_mock_feval = amp_mock_feval
        
        # Define evaluation function
        def feval(ipos, xx, yy, zz):
            # Compute relative position
            relpos = np.array([xx, yy, zz]) - self.refpos

            # If trace output directory provided, set trace file path
            if traces_dir is not None:
                traces_fpath = os.path.join(traces_dir, f'traces_pos{ipos:0{npad}}.csv')
            else:
                traces_fpath = None
            
            # Test mode: adjust mock waveform amplitude according to position
            if self.testmode:
                self.Amock = self.amp_mock_feval(*relpos)
            
            # Acquire and process waveform, and extract pressure amplitude value
            yamps = self.acquire_and_process(tstim, trace_out_fpath=traces_fpath)
            yamp = yamps[self.SIG_KEY]

            # If specified, adjust acquisition vertical scale
            if adjust_scope_vscale and not self.testmode:
                self.scope.adjust_vertical_scale(
                    self.sch_sig, yamp * self.output_conv_constant)

            # Compute peak-to-peak pressure value
            yp2p = 2 * yamp

            # Log data row 
            if log_fpath is not None:
                self.log_data_row(log_fpath, [*relpos, yp2p])
            
            # Return peak-to-peak pressure value and running status
            return yp2p, self.isrunning

        # Run scanning procedure
        tsearch = time.perf_counter()
        scanres = self.scanner.scan(
            x=x, y=y, z=z, feval=feval, plot=self.plot, **kwargs)
        tsearch = time.perf_counter() - tsearch
        logger.info(f'scanning completed in {tsearch:.2f} s')

        # Terminate procedure
        self.terminate_procedure()

        # Return scan result 
        return scanres

    def find_acoustic_focus(self, mp, *args, niters, endmode, adjust_scope_vscale=True, **kwargs):
        '''
        Scan the XYZ space to find the acoustic focus

        :param mp: micro-manipulator object
        :param niters: number of recursive cross-search iterations
        :return: acoustic focus XYZ location and pressure value
        '''
        # Run scanning procedure
        logger.info(f'searching for acoustic focus at f = {si_format(self.f, 3)}Hz')
        scanlocs, scanvals = self.perform_scan(
            mp, *args, CrossSearchScanner, niters=niters, endmode=endmode,
            adjust_scope_vscale=adjust_scope_vscale, **kwargs)

        # Extract XYZ focus and its value
        imax = np.argmax(scanvals)
        focus_xyz, focus_MPa = scanlocs[imax], scanvals[imax]
        
        # Compute relative position of focus relative to reference position
        rel_focus_xyz = focus_xyz - self.refpos
        logger.info(
            f'found focus @ rel. XYZ = {rel_focus_xyz} um, Ppp = {focus_MPa:.3f} MPa)')

        # Return outputs
        return scanlocs[-1], scanvals[-1]

    def map_acoustic_field(self, mp, *args, mode='brute', endmode='gotoref', out_fpath=None, traces_dir=None, **kwargs):
        '''
        Scan the XYZ space to find the acoustic focus

        :param mp: micro-manipulator object
        :param mode: mapping mode, one of 'brute' (for brute force) or 'adaptive' (for adaptive scanning using GPR model)
        :param out_fpath: output file path
        :param traces_dir: path to output directory containing traces files
        :return: 2-tuple with:
            - dictionary of coordinates per dimension (um)
            - multi-dimensional array of pressure values along the grid 
        '''
        # Check end mode validity
        if endmode not in SCAN_ENDMODES:
            raise ValueError(f'invalid endmode "{endmode}". Candidates are {SCAN_ENDMODES}')
        
        # Select appropriate scanner class
        if mode == 'brute':
            scanner_class = GridScanner
        elif mode == 'adaptive':
            scanner_class = AdaptiveScanner
        else:
            raise ValueError(f'invalid mapping mode: "{mode}"')

        # Run scanning procedure
        logger.info(f'mapping transducer field at f = {si_format(self.f, 3)}Hz')
        coords_per_dim, Pmat = self.perform_scan(
            mp, *args, scanner_class, log_fpath=out_fpath, traces_dir=traces_dir, **kwargs)
        
        # Extract focus position (both absolute and relative to scanning origin) and value
        focus_idx = idxmax(Pmat)
        focus_xyz = np.array([coords_per_dim[k][i] for k, i in zip('XYZ', focus_idx)])  # um
        focus_MPa = Pmat[focus_idx]  # MPa
        focus_pos = self.scanner.get_transform().apply(focus_xyz) + self.refpos  # um
        logger.info(
            f'found focus @ XYZ = {focus_pos} um, rel. XYZ = {focus_xyz} um, Ppp = {focus_MPa:.3f} MPa)')

        # If requested, move back to original position
        if endmode == 'gotoref':
            logger.info('moving back to original position')
            if mp is not None:
                mp.set_position(self.refpos)
        
        # If specified, go to position of max value
        if endmode == 'gotomax':
            logger.info(f'going to max value position: {self.scanner.pos_str(focus_pos)}')
            if mp is not None:
                mp.set_position(focus_pos)
        
        # If specified, go to max value position projected onto reference plane
        # of initial position
        elif endmode == 'gotomaxproj':
            logger.info(f'projecting {self.scanner.pos_str(focus_pos)} onto reference plane passing by {self.scanner.pos_str(self.refpos)}')
            proj_pos = project_onto_plane(focus_pos, self.scanner.get_normal_vector(), self.refpos)
            logger.info(f'going to projected location: {self.scanner.pos_str(proj_pos)}')
            if mp is not None:
                mp.set_position(proj_pos)

        # Invert matrix z-axis 
        Pmat = Pmat[:, :, ::-1]

        # Return outputs
        return coords_per_dim, Pmat
    
    @staticmethod
    def get_mapping_code(transducer_code, hydrophone_model, delta, nperax, theta, mode='brute'):
        ''' 
        Get mapping code
        
        :param transducer: transducer code
        :param hydrophone_model: hydrophone model
        :param delta: scanning extent per axis (um)
        :param nperax: number of sampled coordinates per axis
        :param theta: dictionary of rotation angles of scanning system (rad)
        '''
        nperax_str = 'n' + ''.join([f'{k}{v}' for k, v in nperax.items()])
        delta_str = 'D' + ''.join([f'{k}{v / MM_TO_UM:.1f}' for k, v in delta.items()]) + 'mm'
        theta_str = 'θ' + ''.join([f'{k}{v / DEG_TO_RAD:.0f}' for k, v in theta.items()]) + 'deg'
        return f'{transducer_code}_{mode}map_{hydrophone_model}_{today()}_{theta_str}_{delta_str}_{nperax_str}'

