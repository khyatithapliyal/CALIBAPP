# -*- coding: utf-8 -*-
# Standalone test script for burst efficiency check
# Run with: python scripts/test_burst_efficiency.py --testmode

import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

from usnmexps.pulse_sampler import midpoint_time, extract_envelope, steady_state_amplitude



# MOCK CLASSES FOR TESTING it simulates the actual instruments

class MockLogger:
    def info(self, msg):
        print(f'[INFO] {msg}')
    def warning(self, msg):
        print(f'[WARNING] {msg}')
    def error(self, msg):
        print(f'[ERROR] {msg}')
    def debug(self, msg):
        pass

logger = MockLogger()


def si_format(value, precision=3):
    """Simple SI formatter"""
    if value >= 1e6:
        return f'{value/1e6:.{precision}f}M'
    elif value >= 1e3:
        return f'{value/1e3:.{precision}f}k'
    elif value < 1:
        return f'{value*1e3:.{precision}f}m'
    else:
        return f'{value:.{precision}f}'


def vratio_to_gain(ratio):
    """Convert voltage ratio to dB"""
    return 20 * np.log10(ratio)


class MockWaveformGenerator:
    def set_gated_sine_burst(self, f, Vpp, BD, PRF, DC, ich_gate=1, ich_carrier=2):
        logger.info(f'WG: Setting gated sine burst: f={si_format(f)}Hz, Vpp={Vpp}V, BD={BD*1000}ms, PRF={PRF}Hz, DC={DC}%')
    
    def set_trigger_source(self, ich, source):
        logger.info(f'WG: Set channel {ich} trigger source to {source}')
    
    def trigger_channel(self, ich):
        logger.info(f'WG: Triggered channel {ich} programmatically')
    
    def trigger(self, verbose=False):
        logger.info('WG: Triggered burst')
    
    def enable_output_channel(self, ch):
        logger.info(f'WG: Enabled channel {ch}')
    
    def disable_output_channel(self, ch):
        logger.info(f'WG: Disabled channel {ch}')


class MockOscilloscope:
    NHDIVS = 10
    
    def __init__(self, f=2.1e6):
        self.f = f
        self.trigger_source = 1
        self.trigger_holdoff = 0.0
    
    def set_temporal_scale(self, tdiv):
        logger.info(f'OSC: Set time scale to {tdiv*1000:.2f} ms/div')
    
    def set_relative_trigger_delay(self, delay):
        pass

    def set_trigger_source(self, ich):
        self.trigger_source = ich
        logger.info(f'OSC: Trigger source set to channel {ich}')

    def set_trigger_slope(self, ich, slope):
        logger.info(f'OSC: Trigger slope for channel {ich} set to {slope}')

    def set_trigger_holdoff(self, ich, holdoff_s):
        self.trigger_source = ich
        self.trigger_holdoff = holdoff_s
        logger.info(f'OSC: Trigger holdoff set to {holdoff_s*1e3:.2f} ms on channel {ich}')

    def reset_trigger_holdoff(self, ich=None):
        self.trigger_holdoff = 0.0
        logger.info('OSC: Trigger holdoff reset')
    
    def restrict_traces(self, channels):
        logger.info(f'OSC: Restricting to channels {channels}')
    
    def set_probe_attenuation(self, ch, att):
        pass
    
    def arm_acquisition(self):
        logger.info('OSC: Armed for acquisition')
    
    def get_waveform_data(self, ch):
        """Generate mock waveform data for a short carrier window."""
        ncycles = 50
        tstim = ncycles / self.f
        n_points = 12000
        t = np.linspace(0, tstim * 1.1, n_points)

        if ch == 3:  # FWD
            y = 0.042 * np.sin(2 * np.pi * self.f * t)
        elif ch == 4:  # REV
            y = 0.024 * np.sin(2 * np.pi * self.f * t) * 0.4
        else:  # Signal
            y = 0.15 * np.sin(2 * np.pi * self.f * t)

        # Add low-frequency drift/noise to make envelope extraction meaningful.
        y += 0.0003 * np.sin(2 * np.pi * 3e3 * t)
        y += 0.0005 * np.random.randn(len(t))

        return t, y


# -----------------------------------------------------------------------------
# BURST EFFICIENCY CHECK (same logic as in calibrators.py)
# -----------------------------------------------------------------------------

class BurstEfficiencyChecker:
    
    CPL_FWD_KEY = 'cpl_fwd'
    CPL_REV_KEY = 'cpl_rev'
    SIG_KEY = 'sig'
    
    def __init__(self, wg, scope, f=2.1e6, wch_trig=1, wch_sig=2, 
                 sch_cpl_fwd=3, sch_cpl_rev=4, sch_sig=None, testmode=False):
        self.wg = wg
        self.scope = scope
        self.f = f
        self.wch_trig = wch_trig
        self.wch_sig = wch_sig
        self.sch_cpl_fwd = sch_cpl_fwd
        self.sch_cpl_rev = sch_cpl_rev
        self.sch_sig = sch_sig
        self.testmode = testmode
        self.isrunning = True
    
    @property
    def has_coupling_channels(self):
        return self.sch_cpl_fwd is not None and self.sch_cpl_rev is not None
    
    @property
    def has_signal_channel(self):
        return self.sch_sig is not None
    
    def run_burst_efficiency_check(self, Vpp, BD=0.2, PRF=100, DC=50, n_bursts=5, 
                                    pulse_to_sample=10, acq_interval=0.5, plot=True):
        '''
        Run efficiency check using bursts.
        '''
        n_pulses = int(BD * PRF)
        pulse_to_sample = int(np.clip(pulse_to_sample, 1, max(n_pulses, 1)))
        t_pulse_mid = midpoint_time(pulse_to_sample, PRF=PRF, DC=DC)
        
        logger.info('='*60)
        logger.info('BURST EFFICIENCY CHECK')
        logger.info('='*60)
        logger.info(f'f = {si_format(self.f)}Hz, Vpp = {Vpp}V')
        logger.info(f'Burst: {BD*1000:.0f}ms, {PRF}Hz PRF, {DC}% DC ({n_pulses} pulses/burst)')
        logger.info(f'Sampling pulse #{pulse_to_sample} midpoint at {t_pulse_mid*1e3:.2f} ms')
        logger.info('-'*60)
        
        yamps = {}
        all_waveforms = []
        
        if not self.testmode:
            # Configure generator for gated burst delivery
            self.wg.set_gated_sine_burst(
                self.f, Vpp, BD, PRF, DC,
                ich_gate=self.wch_trig,
                ich_carrier=self.wch_sig
            )
            # Switch trigger channel to manual mode for programmatic triggering
            self.wg.set_trigger_source(self.wch_trig, 'MAN')
            if self.sch_cpl_fwd is not None and hasattr(self.scope, 'set_trigger_source'):
                self.scope.set_trigger_source(self.wch_trig)
                self.scope.set_trigger_slope(self.wch_trig, 'POS')
            if hasattr(self.scope, 'set_trigger_holdoff'):
                self.scope.set_trigger_holdoff(self.wch_trig, t_pulse_mid)
            self.wg.enable_output_channel(self.wch_trig)
            self.wg.enable_output_channel(self.wch_sig)
        
        try:
            for i in range(n_bursts):
                if not self.isrunning:
                    break
                
                if i > 0:
                    time.sleep(acq_interval)
                
                if self.testmode:
                    # Generate mock data matching calibration-style capture
                    ncycles = 50
                    tstim = ncycles / self.f
                    n_points = 12000
                    t = np.linspace(0, tstim * 1.1, n_points)
                    yout = {}
                    
                    y_fwd = 0.042 * np.sin(2 * np.pi * self.f * t)
                    y_rev = 0.042 * 0.4 * np.sin(2 * np.pi * self.f * t)  # ratio ~0.4
                    
                    yout[self.CPL_FWD_KEY] = y_fwd + 0.0005 * np.random.randn(len(t))
                    yout[self.CPL_REV_KEY] = y_rev + 0.0005 * np.random.randn(len(t))
                else:
                    # Real acquisition: arm scope, trigger burst, wait, read
                    self.scope.arm_acquisition()
                    time.sleep(0.05)  # allow scope to arm
                    self.wg.trigger_channel(self.wch_trig)
                    time.sleep(BD + 0.1)
                    
                    yout = {}
                    t, yout[self.CPL_FWD_KEY] = self.scope.get_waveform_data(self.sch_cpl_fwd)
                    _, yout[self.CPL_REV_KEY] = self.scope.get_waveform_data(self.sch_cpl_rev)
                
                # Store last waveform for plotting
                if i == n_bursts - 1:
                    all_waveforms.append({'t': t, 'yout': yout.copy()})
                
                yamps = {}
                envs = {}
                for key in [self.CPL_FWD_KEY, self.CPL_REV_KEY]:
                    y = yout[key] - np.mean(yout[key])
                    envs[key] = extract_envelope(y)
                    yamps[key] = steady_state_amplitude(t, y)

                if i == n_bursts - 1:
                    all_waveforms[-1]['env'] = envs
                
                # Log
                cpl_ratio = yamps[self.CPL_REV_KEY] / yamps[self.CPL_FWD_KEY]
                cpl_ratio_dB = vratio_to_gain(cpl_ratio)
                logger.info(f'Burst {i+1}: FWD={yamps[self.CPL_FWD_KEY]:.4f}V, '
                           f'REV={yamps[self.CPL_REV_KEY]:.4f}V, '
                           f'ratio={cpl_ratio:.3f} ({cpl_ratio_dB:.1f} dB)')
        
        finally:
            if not self.testmode:
                if hasattr(self.scope, 'reset_trigger_holdoff'):
                    self.scope.reset_trigger_holdoff(self.wch_trig)
                self.wg.disable_output_channel(self.wch_trig)
                self.wg.disable_output_channel(self.wch_sig)
        
        # Plot
        if plot and all_waveforms:
            self._plot_waveforms(all_waveforms[-1], pulse_to_sample, PRF, DC, t_pulse_mid, yamps)
        
        logger.info('='*60)
        return yamps
    
    def _plot_waveforms(self, waveform_data, pulse_to_sample, PRF, DC, t_pulse_mid, yamps):
        '''Plot captured waveforms with envelope overlays.'''
        t = waveform_data['t']
        yout = waveform_data['yout']
        yenvs = waveform_data.get('env', {})
        
        t_us = t * 1e6  # Convert to microseconds
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        fig.suptitle('BURST EFFICIENCY CHECK - Envelope Verification', fontsize=14, fontweight='bold')
        
        # FWD
        ax = axes[0]
        y_fwd = yout[self.CPL_FWD_KEY]
        ax.plot(t_us, y_fwd * 1000, 'b-', linewidth=0.5, alpha=0.7, label='FWD signal')
        y_fwd_env = yenvs.get(self.CPL_FWD_KEY, extract_envelope(y_fwd - np.mean(y_fwd)))
        ax.plot(t_us, y_fwd_env * 1000, 'b-', linewidth=2, alpha=0.8, label='Envelope')
        ax.axhline(yamps[self.CPL_FWD_KEY] * 1000, color='orange', linestyle='--', label='Mean envelope')
        ax.axhline(-yamps[self.CPL_FWD_KEY] * 1000, color='orange', linestyle='--')
        ax.set_ylabel('FWD (mV)', fontsize=12)
        ax.set_title(f'Forward Coupling — Envelope amplitude: {yamps[self.CPL_FWD_KEY]*1000:.2f} mV', fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # REV
        ax = axes[1]
        y_rev = yout[self.CPL_REV_KEY]
        ax.plot(t_us, y_rev * 1000, 'r-', linewidth=0.5, alpha=0.7, label='REV signal')
        y_rev_env = yenvs.get(self.CPL_REV_KEY, extract_envelope(y_rev - np.mean(y_rev)))
        ax.plot(t_us, y_rev_env * 1000, 'r-', linewidth=2, alpha=0.8, label='Envelope')
        ax.axhline(yamps[self.CPL_REV_KEY] * 1000, color='orange', linestyle='--', label='Mean envelope')
        ax.axhline(-yamps[self.CPL_REV_KEY] * 1000, color='orange', linestyle='--')
        ax.set_ylabel('REV (mV)', fontsize=12)
        ax.set_title(f'Reverse Coupling — Envelope amplitude: {yamps[self.CPL_REV_KEY]*1000:.2f} mV', fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time (μs)', fontsize=12)
        
        # Add summary box
        cpl_ratio = yamps[self.CPL_REV_KEY] / yamps[self.CPL_FWD_KEY]
        cpl_ratio_dB = vratio_to_gain(cpl_ratio)
        
        summary_text = (
            f'SUMMARY\n'
            f'─────────────────\n'
            f'Sampled pulse: #{pulse_to_sample} @ {t_pulse_mid*1e3:.2f} ms\n'
            f'PRF/DC: {PRF:.1f} Hz / {DC:.1f}%\n'
            f'FWD Amplitude: {yamps[self.CPL_FWD_KEY]*1000:.2f} mV\n'
            f'REV Amplitude: {yamps[self.CPL_REV_KEY]*1000:.2f} mV\n'
            f'Coupling Ratio: {cpl_ratio:.3f}\n'
            f'Coupling (dB): {cpl_ratio_dB:.1f} dB'
        )
        
        fig.text(0.02, 0.02, summary_text, fontsize=11, fontfamily='monospace',
                 verticalalignment='bottom',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange'))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.18)
        plt.show()




def main():
    parser = argparse.ArgumentParser(description='Test burst efficiency check')
    parser.add_argument('--testmode', action='store_true', 
                        help='Run in test mode (no hardware)')
    parser.add_argument('--vpp', type=float, default=0.3,
                        help='Voltage Vpp (default: 0.3)')
    parser.add_argument('--n_bursts', type=int, default=5,
                        help='Number of bursts (default: 5)')
    args = parser.parse_args()
    
    print('\n' + '='*60)
    print('BURST EFFICIENCY CHECK - TEST SCRIPT')
    print('='*60)
    print(f'Mode: {"TEST (simulated)" if args.testmode else "HARDWARE"}')
    print(f'Vpp: {args.vpp} V')
    print(f'Number of bursts: {args.n_bursts}')
    print('='*60 + '\n')
    
    # Create instruments (mock or real)
    if args.testmode:
        wg = MockWaveformGenerator()
        scope = MockOscilloscope()
    else:
        # TODO: Replace with actual instrument initialization
        # from instrulink import grab_generator, grab_oscilloscope
        # wg = grab_generator()
        # scope = grab_oscilloscope()
        print("ERROR: Hardware mode not configured. Use --testmode for testing.")
        print("To use hardware, uncomment and configure the instrument imports above.")
        return
    
    # Create checker
    checker = BurstEfficiencyChecker(
        wg=wg,
        scope=scope,
        f=2.1e6,           # 2.1 MHz carrier
        wch_trig=1,        # Trigger channel
        wch_sig=2,         # Signal channel
        sch_cpl_fwd=3,     # Oscilloscope FWD channel
        sch_cpl_rev=4,     # Oscilloscope REV channel
        testmode=args.testmode
    )
    
    # Run check
    yamps = checker.run_burst_efficiency_check(
        Vpp=args.vpp,
        BD=0.2,            # 200ms burst
        PRF=100,           # 100Hz PRF
        DC=50,             # 50% duty cycle
        n_bursts=args.n_bursts,
        pulse_to_sample=10,
        plot=True
    )
    
    # Simulate comparison to reference
    print('\n' + '-'*60)
    print('COMPARISON TO REFERENCE (simulated)')
    print('-'*60)
    
    # Mock reference (in real code, this comes from calibration data)
    ref_ratio = 0.57
    ref_ratio_dB = vratio_to_gain(ref_ratio)
    
    measured_ratio = yamps['cpl_rev'] / yamps['cpl_fwd']
    measured_ratio_dB = vratio_to_gain(measured_ratio)
    
    delta_dB = measured_ratio_dB - ref_ratio_dB
    
    print(f'Expected coupling ratio: {ref_ratio:.3f} ({ref_ratio_dB:.1f} dB)')
    print(f'Measured coupling ratio: {measured_ratio:.3f} ({measured_ratio_dB:.1f} dB)')
    print(f'Deviation: {delta_dB:+.2f} dB')
    print()
    
    MAX_CPL_DEV_DB = 3.0  # Threshold
    if abs(delta_dB) < MAX_CPL_DEV_DB:
        print('✓ PASS - Transducer efficiency within margin')
    else:
        print('✗ FAIL - Transducer efficiency outside margin!')
    
    print('='*60 + '\n')


if __name__ == '__main__':
    main()
