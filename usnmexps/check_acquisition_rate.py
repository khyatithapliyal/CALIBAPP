import time

from instrulink.logger import logger


def check_scope_acquisition_rate(scope, channels, n_trials=20, acq_interval=None):
    """Measure how long the scope takes to return waveform data."""
    if not channels:
        raise ValueError('channels must not be empty')

    sample_rate = scope.get_sample_rate()
    logger.info(f'scope sample rate: {sample_rate:.2f} Sa/s')

    trial_times = []
    for itrial in range(n_trials):
        start = time.perf_counter()
        scope.arm_acquisition()
        for channel in channels:
            scope.get_waveform_data(channel)
        elapsed = time.perf_counter() - start
        trial_times.append(elapsed)
        logger.info(f'trial {itrial + 1}: {elapsed * 1000:.1f} ms')

    mean_time = sum(trial_times) / len(trial_times)
    refresh_rate = 1.0 / mean_time if mean_time > 0 else float('inf')

    logger.info(f'mean acquisition time: {mean_time * 1000:.1f} ms')
    logger.info(f'effective refresh rate: {refresh_rate:.2f} Hz')

    if acq_interval is not None:
        target_rate = 1.0 / acq_interval if acq_interval > 0 else float('inf')
        logger.info(f'target interval: {acq_interval:.3f} s ({target_rate:.2f} Hz)')
        if mean_time > acq_interval:
            logger.warning('scope acquisition is slower than the target interval')

    return {
        'sample_rate': sample_rate,
        'trial_times': trial_times,
        'mean_time': mean_time,
        'refresh_rate': refresh_rate,
    }
