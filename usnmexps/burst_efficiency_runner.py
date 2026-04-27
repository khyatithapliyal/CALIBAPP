# -*- coding: utf-8 -*-
# this file is where the triggering  of he waveform generator and oscilloscope and readin scope, computing ratioan dthe final plotting of teh efficinecy happens
import time
import numpy as np
import matplotlib.pyplot as plt

from instrulink import logger

from .pulse_sampler import midpoint_time, extract_envelope, steady_state_amplitude
from .calib_utils import vratio_to_gain


class BurstEfficiencyRunner:
    """Reusable burst efficiency acquisition/plot orchestration."""

    SIGNAL_TAIL_FRACTION = 0.5
    WAVEFORM_READ_RETRIES = 2

    def __init__(self, calibrator):
        self.cal = calibrator

    @staticmethod
    def _is_retryable_waveform_error(err):
        msg = str(err).lower()
        return (
            'waveform parsing error' in msg
            or 'expected number of points (0)' in msg
        )
    # the actual oscilloscope read is then called here and if it fails we retry a few times before giving up, this is because sometimes the scope read can fail due to transient communication issues and we want to be robust to that
    def _acquire_waveforms_once(self, BD):
        self.cal.scope.arm_acquisition()
        time.sleep(0.05)
        self.cal.wg.trigger_channel(self.cal.wch_trig)
        time.sleep(BD + 0.1)

        t = None
        yout = {}
        if self.cal.has_signal_channel:
            # this is where we get teh waveform from the scope for the signal of interest, we also get the coupling channels if they exist
            t, yout[self.cal.SIG_KEY] = self.cal.scope.get_waveform_data(self.cal.sch_sig)
        if self.cal.has_coupling_channels:
            t1, yout[self.cal.CPL_FWD_KEY] = self.cal.scope.get_waveform_data(self.cal.sch_cpl_fwd)
            _, yout[self.cal.CPL_REV_KEY] = self.cal.scope.get_waveform_data(self.cal.sch_cpl_rev)
            if t is None:
                t = t1
        return t, yout

    def _acquire_waveforms_with_retry(self, BD, context_label):
        for iretry in range(self.WAVEFORM_READ_RETRIES + 1):
            try:
                return self._acquire_waveforms_once(BD)
            except Exception as e:
                retryable = self._is_retryable_waveform_error(e)
                if retryable and iretry < self.WAVEFORM_READ_RETRIES:
                    logger.warning(
                        f'{context_label}: transient waveform read failure ({e}) -> retry {iretry + 1}/{self.WAVEFORM_READ_RETRIES}'
                    )
                    time.sleep(0.2)
                    continue
                raise
    # Thsi is where the main runner starts and configuresthe cope and genrator, triggers bursts, reads waveform , converst teh waveform to amplitudes and computes the ratio
    def run(
        self,
        Vpp,
        BD=0.2,
        PRF=100,
        DC=50,
        n_bursts=5,
        pulse_to_sample=10,
        acq_interval=1.0,
        plot=True,
        n_warmup=2,
    ):
        """Run burst efficiency check using midpoint sampling and envelope amplitude."""
        n_pulses = int(BD * PRF)
        pulse_duration = DC / (100 * PRF)
        if n_pulses < 1:
            raise ValueError(f'invalid burst settings: BD={BD}, PRF={PRF} produce no pulses')
        pulse_to_sample = int(np.clip(pulse_to_sample, 1, n_pulses))
        t_pulse_mid = midpoint_time(pulse_to_sample, PRF=PRF, DC=DC)

        logger.info('=' * 60)
        logger.info('BURST EFFICIENCY CHECK')
        logger.info('=' * 60)
        logger.info(f'f = {self.cal.f:.6g}Hz, Vpp = {Vpp:.6g}V')
        logger.info(f'Burst: {BD*1000:.0f}ms, {PRF}Hz PRF, {DC}% DC ({n_pulses} pulses/burst)')
        logger.info(f'Each pulse: {pulse_duration*1000:.1f}ms ON, {(1/PRF - pulse_duration)*1000:.1f}ms OFF')
        logger.info(f'Sampling pulse #{pulse_to_sample} midpoint at t = {t_pulse_mid*1e3:.2f} ms from burst start')
        logger.info('-' * 60)

        self.cal.isrunning = True
        yamps_list = []
        last_waveform = None
        last_yamps = {}

        tstim = self.cal.ncycles / self.cal.f
        tdiv = self.cal.get_scope_tdiv(tstim)
        logger.info('Using midpoint-targeted short-window acquisition:')
        logger.info(f'  ncycles = {self.cal.ncycles}, tstim = {tstim*1e6:.1f} us')
        logger.info(f'  tdiv = {tdiv*1e6:.2f} us/div')

        cpl_vdiv = max(self.cal.sch_cpl_vdiv, Vpp * 0.2)
        logger.info(f'  cpl_vdiv = {cpl_vdiv*1000:.0f} mV/div')
        # The scope is told to open ts capture window at exactly that momentv via holdoff
        holdoff_supported = False
        if not self.cal.testmode:
            self.cal.set_scope_time_settings(tdiv)
            self.cal.set_scope_vertical_settings(self.cal.sch_sig_vdiv, cpl_vdiv)

            if self.cal.sch_trig is not None and hasattr(self.cal.scope, 'set_trigger_source'):
                try:
                    self.cal.scope.set_trigger_source(self.cal.sch_trig)
                    self.cal.scope.set_trigger_slope(self.cal.sch_trig, 'POS')
                except Exception as e:
                    logger.warning(f'Could not explicitly configure scope trigger source: {e}')

            if self.cal.sch_trig is not None and hasattr(self.cal.scope, 'set_trigger_holdoff'):
                try:
                    self.cal.scope.set_trigger_holdoff(self.cal.sch_trig, t_pulse_mid)
                    holdoff_supported = True
                    logger.info('Configured scope trigger holdoff for midpoint sampling')
                except Exception as e:
                    logger.warning(f'Could not set trigger holdoff ({e}) -> using default trigger timing')
            else:
                logger.warning('Scope does not support trigger holdoff API; midpoint targeting disabled')

            self.cal.scope.restrict_traces(self.cal.sch_channels)
            for ich in self.cal.sch_channels:
                self.cal.scope.set_probe_attenuation(ich, 1)

            try:
                self.cal.check_sample_rate()
            except ValueError as e:
                logger.warning(f'Sample rate check: {e}')

            # Ensure waveform parser has a valid expected point count.
            if hasattr(self.cal.scope, 'set_waveform_settings') and hasattr(self.cal, 'acq_npoints'):
                self.cal.scope.set_waveform_settings(npoints=self.cal.acq_npoints)
            if hasattr(self.cal.scope, 'set_nsweeps_per_acquisition') and hasattr(self.cal, 'acq_nsweeps'):
                self.cal.scope.set_nsweeps_per_acquisition(self.cal.acq_nsweeps)

            self.cal.wg.set_gated_sine_burst(
                self.cal.f,
                Vpp,
                BD,
                PRF,
                DC,
                ich_gate=self.cal.wch_trig,
                ich_carrier=self.cal.wch_sig,
            )
            logger.info(f'switching trigger channel {self.cal.wch_trig} to manual trigger mode')
            self.cal.wg.set_trigger_source(self.cal.wch_trig, 'MAN')
            self.cal.enable_generator()

        try:
            if not self.cal.testmode and n_warmup > 0:
                logger.info(f'Firing {n_warmup} warm-up burst(s) (discarded)...')
                for w in range(n_warmup):
                    self.cal.scope.arm_acquisition()
                    time.sleep(0.05)
                    self.cal.wg.trigger_channel(self.cal.wch_trig)
                    time.sleep(BD + 0.1)
                    logger.info(f'  warm-up {w+1}/{n_warmup} done')
                time.sleep(0.5)

            for i in range(n_bursts):
                if not self.cal.isrunning:
                    break
                if i > 0:
                    time.sleep(acq_interval)

                if self.cal.testmode:
                    t, ymock = self.cal.tmock.copy(), self.cal.ymock.copy()
                    yout = {}
                    if self.cal.has_signal_channel:
                        yout[self.cal.SIG_KEY] = ymock * self.cal.Amock
                    if self.cal.has_coupling_channels:
                        yout[self.cal.CPL_FWD_KEY] = ymock * self.cal.Amock * 0.5
                        yout[self.cal.CPL_REV_KEY] = ymock * self.cal.Amock * 0.2
                else:
                    t, yout = self._acquire_waveforms_with_retry(BD, f'burst {i+1}/{n_bursts}')

                if t is None:
                    logger.warning(f'burst {i+1}/{n_bursts}: no waveform captured')
                    continue
                # Log the captured time range and number of points for the first burst
                if i == 0:
                    logger.info(f'Captured: {t[0]*1e6:.1f} to {t[-1]*1e6:.1f} us ({len(t)} points)')
                # envelope extraction 
                yamps = {}
                envs = {}
                for key, y in yout.items():
                    if y is not None and len(y) > 0:
                        y_centered = y - np.mean(y)
                        env = extract_envelope(y_centered)
                        envs[key] = env
                        tail_fraction = 1.0
                        if key == self.cal.SIG_KEY:
                            tail_fraction = self.SIGNAL_TAIL_FRACTION
                        yamps[key] = steady_state_amplitude(
                            t,
                            y_centered,
                            tail_fraction=tail_fraction,
                        )

                last_waveform = {'t': t, 'yout': yout.copy(), 'env': envs.copy()}
                last_yamps = yamps.copy()
                yamps_list.append(yamps.copy())

                if self.cal.has_coupling_channels and self.cal.CPL_FWD_KEY in yamps:
                    fwd_amp = yamps.get(self.cal.CPL_FWD_KEY, 0)
                    rev_amp = yamps.get(self.cal.CPL_REV_KEY, 0)
                    if fwd_amp > 1e-6:
                        cpl_ratio = rev_amp / fwd_amp
                        cpl_ratio_dB = vratio_to_gain(cpl_ratio)
                        logger.info(
                            f'burst {i+1}/{n_bursts}: FWD={fwd_amp*1000:.2f}mV, '
                            f'REV={rev_amp*1000:.2f}mV, '
                            f'ratio={cpl_ratio:.3f} ({cpl_ratio_dB:.1f} dB)'
                        )
                    else:
                        logger.warning(f'burst {i+1}/{n_bursts}: FWD amplitude too low ({fwd_amp*1000:.4f}mV)')

        finally:
            if not self.cal.testmode:
                if holdoff_supported and hasattr(self.cal.scope, 'reset_trigger_holdoff'):
                    try:
                        self.cal.scope.reset_trigger_holdoff(self.cal.sch_trig)
                    except Exception as e:
                        logger.warning(f'Could not reset trigger holdoff: {e}')
                self.cal.disable_generator()

        if len(yamps_list) > 0:
            settled_yamps_list = yamps_list[1:] if len(yamps_list) > 1 else yamps_list
            if len(yamps_list) > 1:
                logger.info('Excluding burst 1 from reported mean due to startup transient')
            avg_yamps = {}
            for key in settled_yamps_list[0].keys():
                values = [y[key] for y in settled_yamps_list if key in y]
                avg_yamps[key] = np.mean(values)

            if self.cal.has_coupling_channels and self.cal.CPL_FWD_KEY in avg_yamps:
                fwd_amp = avg_yamps[self.cal.CPL_FWD_KEY]
                rev_amp = avg_yamps[self.cal.CPL_REV_KEY]
                if fwd_amp > 0:
                    mean_ratio = rev_amp / fwd_amp
                    mean_ratio_dB = vratio_to_gain(mean_ratio)
                    ratios = [
                        y[self.cal.CPL_REV_KEY] / y[self.cal.CPL_FWD_KEY]
                        for y in settled_yamps_list
                        if y.get(self.cal.CPL_FWD_KEY, 0) > 0
                    ]
                    std_ratio = np.std(ratios) if len(ratios) > 1 else 0
                    logger.info('-' * 60)
                    logger.info(f'MEAN: FWD={fwd_amp*1000:.2f}mV, REV={rev_amp*1000:.2f}mV')
                    logger.info(
                        f'MEAN coupling ratio: {mean_ratio:.3f} ({mean_ratio_dB:.1f} dB) ± {std_ratio:.4f}'
                    )
            yamps = avg_yamps
        else:
            yamps = {}

        if plot and last_waveform is not None:
            self.plot(
                last_waveform,
                last_yamps,
                pulse_to_sample=pulse_to_sample,
                PRF=PRF,
                DC=DC,
                t_pulse_mid=t_pulse_mid,
            )

        logger.info('=' * 60)
        return yamps

    def plot(self, waveform_data, yamps, pulse_to_sample, PRF, DC, t_pulse_mid):
        """Plot burst-check waveforms with extracted envelope and amplitude overlays."""
        t = waveform_data['t']
        yout = waveform_data['yout']
        yenvs = waveform_data.get('env', {})
        t_us = t * 1e6

        n_plots = 0
        if self.cal.has_coupling_channels:
            n_plots += 2
        if self.cal.has_signal_channel and self.cal.SIG_KEY in yout:
            n_plots += 1
        if n_plots == 0:
            return

        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots), sharex=True)
        if n_plots == 1:
            axes = [axes]
        fig.suptitle('BURST EFFICIENCY CHECK - Pulse Envelope', fontsize=14, fontweight='bold')

        def plot_one(ax, key, color, ylabel, title_prefix):
            y = yout[key]
            env = yenvs.get(key, None)
            y_mV = y * 1000
            ax.plot(t_us, y_mV, color=color, linewidth=0.5, alpha=0.5, label='raw')
            if env is not None:
                env_mV = env * 1000
                ax.plot(t_us, env_mV, 'k--', linewidth=1.2, label='envelope')
                ax.plot(t_us, -env_mV, 'k--', linewidth=1.0)
            amp_mV = yamps.get(key, np.nan) * 1000
            if np.isfinite(amp_mV):
                ax.axhline(amp_mV, color='orange', linestyle='--', linewidth=1.2, label='mean envelope')
                ax.axhline(-amp_mV, color='orange', linestyle='--', linewidth=1.2)
            ax.set_ylabel(ylabel)
            ax.set_title(f'{title_prefix} - amplitude: {amp_mV:.2f} mV')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')

        ax_idx = 0
        if self.cal.has_coupling_channels and self.cal.CPL_FWD_KEY in yout:
            plot_one(axes[ax_idx], self.cal.CPL_FWD_KEY, 'b', 'FWD (mV)', 'Forward coupling')
            ax_idx += 1
        if self.cal.has_coupling_channels and self.cal.CPL_REV_KEY in yout:
            plot_one(axes[ax_idx], self.cal.CPL_REV_KEY, 'r', 'REV (mV)', 'Reverse coupling')
            ax_idx += 1
        if self.cal.has_signal_channel and self.cal.SIG_KEY in yout:
            plot_one(axes[ax_idx], self.cal.SIG_KEY, 'g', 'Signal (mV)', 'Hydrophone signal')

        axes[-1].set_xlabel('Time (us)')

        if self.cal.has_coupling_channels and self.cal.CPL_FWD_KEY in yamps and self.cal.CPL_REV_KEY in yamps:
            fwd_amp = yamps[self.cal.CPL_FWD_KEY]
            rev_amp = yamps[self.cal.CPL_REV_KEY]
            if fwd_amp > 0:
                cpl_ratio = rev_amp / fwd_amp
                cpl_ratio_dB = vratio_to_gain(cpl_ratio)
                summary_text = (
                    f'RESULTS\n'
                    f'-----------------\n'
                    f'Sampled pulse: #{pulse_to_sample} @ {t_pulse_mid*1e3:.2f} ms\n'
                    f'PRF/DC: {PRF:.1f} Hz / {DC:.1f}%\n'
                    f'FWD: {fwd_amp*1000:.2f} mV\n'
                    f'REV: {rev_amp*1000:.2f} mV\n'
                    f'Ratio: {cpl_ratio:.3f}\n'
                    f'dB: {cpl_ratio_dB:.1f} dB'
                )
                fig.text(
                    0.02,
                    0.02,
                    summary_text,
                    fontsize=10,
                    fontfamily='monospace',
                    verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
                )

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
        plt.show(block=False)
        plt.pause(0.001)
    # thsi where is the runner does pu;se profile, pu;lse 1, pulse 2.... one short window each
    def profile_pulse_stability(self, Vpp, BD=0.2, PRF=100, DC=50, n_bursts=1, plot=True, n_warmup=0):
        """Profile pulse amplitudes using pulse-indexed short-window acquisitions."""
        n_pulses = int(BD * PRF)
        if n_pulses < 1:
            raise ValueError(f'invalid burst settings: BD={BD}, PRF={PRF} produce no pulses')

        logger.info('-' * 60)
        logger.info('Running pulse-by-pulse stability profile as part of efficiency check')
        logger.info('=' * 60)
        logger.info('PULSE STABILITY PROFILE')
        logger.info('=' * 60)
        logger.info(f'f = {self.cal.f:.6g}Hz, Vpp = {Vpp:.6g}V')
        logger.info(f'Burst: {BD*1000:.0f}ms, {PRF}Hz PRF, {DC}% DC ({n_pulses} pulses/burst)')
        logger.info(f'Sampling pulse #1-{n_pulses} using midpoint-triggered short windows')
        logger.info(f'Repeated captures per pulse: {n_bursts}')
        logger.info('-' * 60)

        pulse_nums = np.arange(1, n_pulses + 1)
        channel_keys = []
        if self.cal.has_coupling_channels:
            channel_keys.extend([self.cal.CPL_FWD_KEY, self.cal.CPL_REV_KEY])
        if self.cal.has_signal_channel:
            channel_keys.append(self.cal.SIG_KEY)
        pulse_profiles = {
            key: np.full((n_bursts, n_pulses), np.nan, dtype=float)
            for key in channel_keys
        }

        tstim = self.cal.ncycles / self.cal.f
        tdiv = self.cal.get_scope_tdiv(tstim)
        cpl_vdiv = max(self.cal.sch_cpl_vdiv, Vpp * 0.2)
        holdoff_supported = False

        if not self.cal.testmode:
            self.cal.set_scope_time_settings(tdiv)
            self.cal.set_scope_vertical_settings(self.cal.sch_sig_vdiv, cpl_vdiv)

            if self.cal.sch_trig is not None and hasattr(self.cal.scope, 'set_trigger_source'):
                try:
                    self.cal.scope.set_trigger_source(self.cal.sch_trig)
                    self.cal.scope.set_trigger_slope(self.cal.sch_trig, 'POS')
                except Exception as e:
                    logger.warning(f'Could not explicitly configure scope trigger source: {e}')

            if self.cal.sch_trig is not None and hasattr(self.cal.scope, 'set_trigger_holdoff'):
                try:
                    self.cal.scope.set_trigger_holdoff(self.cal.sch_trig, midpoint_time(1, PRF=PRF, DC=DC))
                    holdoff_supported = True
                except Exception as e:
                    logger.warning(f'Could not set trigger holdoff for pulse indexing: {e}')

            if not holdoff_supported:
                raise ValueError('Pulse profile requires scope trigger holdoff support')

            self.cal.scope.restrict_traces(self.cal.sch_channels)
            for ich in self.cal.sch_channels:
                self.cal.scope.set_probe_attenuation(ich, 1)

            try:
                self.cal.check_sample_rate()
            except ValueError as e:
                logger.warning(f'Sample rate check: {e}')

            # Ensure waveform parser has a valid expected point count.
            if hasattr(self.cal.scope, 'set_waveform_settings') and hasattr(self.cal, 'acq_npoints'):
                self.cal.scope.set_waveform_settings(npoints=self.cal.acq_npoints)
            if hasattr(self.cal.scope, 'set_nsweeps_per_acquisition') and hasattr(self.cal, 'acq_nsweeps'):
                self.cal.scope.set_nsweeps_per_acquisition(self.cal.acq_nsweeps)

            self.cal.wg.set_gated_sine_burst(
                self.cal.f,
                Vpp,
                BD,
                PRF,
                DC,
                ich_gate=self.cal.wch_trig,
                ich_carrier=self.cal.wch_sig,
            )
            self.cal.wg.set_trigger_source(self.cal.wch_trig, 'MAN')
            self.cal.enable_generator()

        captured_once = False
        try:
            if not self.cal.testmode and n_warmup > 0:
                logger.info(f'Firing {n_warmup} warm-up burst(s) before profile...')
                for w in range(n_warmup):
                    self.cal.scope.arm_acquisition()
                    time.sleep(0.05)
                    self.cal.wg.trigger_channel(self.cal.wch_trig)
                    time.sleep(BD + 0.1)
                    logger.info(f'  warm-up {w + 1}/{n_warmup} done')

            for ipulse, pulse_num in enumerate(pulse_nums):
                if not self.cal.isrunning:
                    break

                if not self.cal.testmode:
                    t_pulse_mid = midpoint_time(pulse_num, PRF=PRF, DC=DC)
                    try:
                        self.cal.scope.set_trigger_holdoff(self.cal.sch_trig, t_pulse_mid)
                    except Exception as e:
                        logger.warning(f'Could not set trigger holdoff for pulse {pulse_num}: {e}')
                        continue

                logger.info(f'Acquiring pulse {pulse_num}/{n_pulses}...')

                for iburst in range(n_bursts):
                    if not self.cal.isrunning:
                        break

                    if self.cal.testmode:
                        sr = max(int(250e6), int(np.ceil(200 * self.cal.f)))
                        window_dur = 10 * tdiv
                        t = np.arange(0, window_dur, 1 / sr)
                        amplitude_scale = 1.0 - 0.15 * np.exp(-(pulse_num - 1) / 2)
                        carrier = np.sin(2 * np.pi * self.cal.f * t)
                        yout = {}
                        if self.cal.has_signal_channel:
                            yout[self.cal.SIG_KEY] = 0.026 * amplitude_scale * carrier
                        if self.cal.has_coupling_channels:
                            yout[self.cal.CPL_FWD_KEY] = 0.038 * amplitude_scale * carrier
                            yout[self.cal.CPL_REV_KEY] = 0.016 * amplitude_scale * carrier
                    else:
                        t, yout = self._acquire_waveforms_with_retry(
                            BD,
                            f'profile pulse {pulse_num}/{n_pulses} repetition {iburst + 1}/{n_bursts}',
                        )

                    if t is None:
                        logger.warning(
                            f'profile pulse {pulse_num}/{n_pulses} repetition {iburst + 1}/{n_bursts}: no waveform captured'
                        )
                        continue

                    if not captured_once:
                        logger.info(
                            f'Captured short window: {t[0]*1e6:.1f} to {t[-1]*1e6:.1f} us ({len(t)} points)'
                        )
                        captured_once = True

                    for key, y in yout.items():
                        if key not in pulse_profiles or y is None or len(y) == 0:
                            continue

                        y_centered = y - np.mean(y)
                        env = extract_envelope(y_centered)
                        valid = np.isfinite(env)
                        if not np.any(valid):
                            amp = np.nan
                        else:
                            tail_fraction = self.SIGNAL_TAIL_FRACTION if key == self.cal.SIG_KEY else 1.0
                            amp = steady_state_amplitude(
                                t,
                                y_centered,
                                tail_fraction=tail_fraction,
                            )
                        pulse_profiles[key][iburst, ipulse] = amp

                if self.cal.has_coupling_channels and self.cal.CPL_FWD_KEY in pulse_profiles:
                    fwd_vals = pulse_profiles[self.cal.CPL_FWD_KEY][:, ipulse]
                    rev_vals = pulse_profiles[self.cal.CPL_REV_KEY][:, ipulse]
                    fwd_mean = float(np.nanmean(fwd_vals)) if np.any(np.isfinite(fwd_vals)) else np.nan
                    rev_mean = float(np.nanmean(rev_vals)) if np.any(np.isfinite(rev_vals)) else np.nan
                    if np.isfinite(fwd_mean) and fwd_mean > 0 and np.isfinite(rev_mean):
                        ratio = rev_mean / fwd_mean
                        ratio_db = vratio_to_gain(ratio)
                        logger.info(
                            f'  pulse {pulse_num:02d}: FWD={fwd_mean*1000:.2f}mV, '
                            f'REV={rev_mean*1000:.2f}mV, ratio={ratio:.3f} ({ratio_db:.1f} dB)'
                        )

        finally:
            if not self.cal.testmode:
                if holdoff_supported and hasattr(self.cal.scope, 'reset_trigger_holdoff'):
                    try:
                        self.cal.scope.reset_trigger_holdoff(self.cal.sch_trig)
                    except Exception as e:
                        logger.warning(f'Could not reset trigger holdoff after profile: {e}')
                self.cal.disable_generator()

        profile_results = {}
        for key, burst_array in pulse_profiles.items():
            if not np.any(np.isfinite(burst_array)):
                continue
            mean_profile = np.nanmean(burst_array, axis=0)
            std_profile = np.nanstd(burst_array, axis=0) if burst_array.shape[0] > 1 else np.zeros_like(mean_profile)
            profile_results[key] = {
                'pulse_nums': pulse_nums,
                'bursts': burst_array,
                'mean': mean_profile,
                'std': std_profile,
            }
            self._log_profile_summary(key, mean_profile)

        if plot and profile_results:
            self._plot_pulse_profile(profile_results)

        logger.info('=' * 60)
        return profile_results

    def _log_profile_summary(self, key, mean_profile):
        """Log a concise, readable summary of one pulse profile."""
        label_map = {
            self.cal.CPL_FWD_KEY: 'FWD',
            self.cal.CPL_REV_KEY: 'REV',
            self.cal.SIG_KEY: 'HYD',
        }
        label = label_map.get(key, key.upper())
        valid = np.isfinite(mean_profile)
        if not np.any(valid):
            logger.warning(f'{label} profile: no valid pulse amplitudes extracted')
            return

        pulse1 = mean_profile[0]
        settled = mean_profile[1:] if mean_profile.size > 1 else mean_profile
        settled = settled[np.isfinite(settled)]
        if settled.size == 0:
            settled = mean_profile[valid]

        settled_mean = float(np.nanmean(settled))
        settled_std = float(np.nanstd(settled)) if settled.size > 1 else 0.0
        settled_min = float(np.nanmin(settled))
        settled_max = float(np.nanmax(settled))
        logger.info(
            f'{label} profile summary: pulse1={pulse1*1000:.2f} mV, '
            f'settled(2-{mean_profile.size})={settled_mean*1000:.2f} ± {settled_std*1000:.2f} mV, '
            f'range={settled_min*1000:.2f}-{settled_max*1000:.2f} mV'
        )

    def _plot_pulse_profile(self, profile_results):
        """Plot readable pulse-by-pulse amplitude profiles."""
        n_channels = len(profile_results)
        if n_channels == 0:
            return

        fig, axes = plt.subplots(n_channels, 1, figsize=(10, 4 * n_channels), sharex=True)
        if n_channels == 1:
            axes = [axes]

        first_result = next(iter(profile_results.values()))
        n_pulses = len(first_result['pulse_nums'])
        fig.suptitle(
            f'PULSE STABILITY PROFILE - Amplitude Evolution Across Pulse #1-{n_pulses}',
            fontsize=14,
            fontweight='bold',
        )
        colors = {
            self.cal.CPL_FWD_KEY: 'tab:blue',
            self.cal.CPL_REV_KEY: 'tab:red',
            self.cal.SIG_KEY: 'tab:green',
        }
        labels = {
            self.cal.CPL_FWD_KEY: 'Forward coupling',
            self.cal.CPL_REV_KEY: 'Reverse coupling',
            self.cal.SIG_KEY: 'Hydrophone signal',
        }

        for ax, (key, result) in zip(axes, profile_results.items()):
            pulse_nums = result['pulse_nums']
            bursts = result['bursts'] * 1000
            mean_profile = result['mean'] * 1000
            std_profile = result['std'] * 1000
            color = colors.get(key, 'black')
            label = labels.get(key, key)

            for burst_idx, burst_profile in enumerate(bursts, start=1):
                ax.plot(
                    pulse_nums,
                    burst_profile,
                    color=color,
                    linewidth=1.0,
                    alpha=0.20,
                    marker='o',
                    markersize=3,
                    label='individual burst' if burst_idx == 1 else None,
                )

            ax.plot(
                pulse_nums,
                mean_profile,
                color=color,
                linewidth=2.5,
                marker='o',
                markersize=5,
                label='mean amplitude',
            )
            if bursts.shape[0] > 1:
                ax.fill_between(
                    pulse_nums,
                    mean_profile - std_profile,
                    mean_profile + std_profile,
                    color=color,
                    alpha=0.15,
                    label='±1 std',
                )

            settled_mean = np.nanmean(mean_profile[1:]) if mean_profile.size > 1 else np.nanmean(mean_profile)
            ax.axhline(settled_mean, color='gray', linestyle='--', linewidth=1.2, label='settled mean (2-N)')
            ax.axvline(1, color='black', linestyle=':', linewidth=1.0, alpha=0.5)
            ax.set_ylabel('Amplitude (mV)')
            ax.set_title(label)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')

        axes[-1].set_xlabel('Pulse Number Within Burst')
        axes[-1].set_xticks(profile_results[next(iter(profile_results))]['pulse_nums'])
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)
