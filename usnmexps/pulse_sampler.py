# this file is teh toolbox ocntaining timimg and enevelope and amplitude extractionimport numpy as np
from scipy.signal import hilbert

from .utils import apply_rolling_window
from .constants import NAVG_PENV, NTRIM_PENV
import numpy as np


def midpoint_time(pulse_num, PRF, DC):
    """
    Return the time from burst start to the midpoint of the given pulse.

    For a square envelope the midpoint is simply half-way through the ON
    period.  For a ramped (smooth) envelope it is also the point where the
    signal has been at full amplitude for longest before it starts to fall,
    so it is always the safest sampling point.

    :param pulse_num: 1-indexed pulse number within the burst (e.g. 10)
    :param PRF: pulse repetition frequency (Hz)
    :param DC: duty cycle in percent (%)
    :return: midpoint time in seconds from burst start

    Example
    -------
    >>> midpoint_time(10, PRF=100, DC=50)
    0.0925   # 9 × 10 ms  +  5 ms / 2  =  92.5 ms
    """
    pulse_period = 1.0 / PRF           # seconds per pulse period
    pulse_dur = DC / 100.0 / PRF       # ON duration within one period
    return (pulse_num - 1) * pulse_period + pulse_dur / 2.0


def extract_envelope(y, navg=NAVG_PENV, ntrim=NTRIM_PENV):
    """
    Hilbert-transform envelope of a zero-mean sinusoidal signal.

    :param y: 1-D signal array (should be zero-mean before calling)
    :param navg: moving-average window length (odd int >= 1); smooths the
                 Hilbert envelope.  Default = NAVG_PENV (101 samples).
    :param ntrim: number of samples at each edge to set to NaN, removing
                  Hilbert ringing artefacts.  Default = NTRIM_PENV (500).
    :return: envelope array with the same length as *y*
    """
    env = np.abs(hilbert(y))
    if navg > 1:
        env = apply_rolling_window(env, navg)
    # Trim the edges to remove Hilbert artefacts    
    if ntrim > 0:
        env = env.copy()
        env[:ntrim] = np.nan
        env[-ntrim:] = np.nan
    return env


def steady_state_amplitude(t, y, navg=NAVG_PENV, ntrim=NTRIM_PENV, tail_fraction=1.0):
    """
    Scalar steady-state amplitude of a captured carrier burst.

    The captured window is assumed to lie entirely in the steady-state
    region (i.e. it was triggered at or near the pulse midpoint via
    scope trigger holdoff).  The result is the mean Hilbert envelope
    over the valid portion of the window.

    :param t: time vector (s) — kept for API consistency with
              Calibrator.process_waveform; not used internally.
    :param y: zero-mean sinusoidal signal (1-D array)
    :param navg: moving-average window for envelope smoothing
    :param ntrim: edge samples to discard
    :param tail_fraction: fraction of the valid envelope window to average,
                          taken from the end of the capture. Use values < 1
                          to focus on the late steady-state plateau.
    :return: mean envelope amplitude (V) as a Python float

    If all envelope values are NaN (extremely short signal or degenerate
    case) falls back to the peak of the raw Hilbert magnitude.
    """
    env = extract_envelope(y, navg=navg, ntrim=ntrim)
    valid = np.isfinite(env)
    if not np.any(valid):
        return float(np.max(np.abs(hilbert(y))))
    env_valid = env[valid]
    tail_fraction = float(np.clip(tail_fraction, 0.0, 1.0))
    if tail_fraction <= 0.0:
        tail_fraction = 1.0
    start_idx = int(np.floor((1.0 - tail_fraction) * env_valid.size))
    env_window = env_valid[start_idx:]
    if env_window.size == 0:
        env_window = env_valid
    return float(np.nanmean(env_window))
