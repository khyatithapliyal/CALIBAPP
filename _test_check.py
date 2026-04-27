"""Quick sanity check of all modified modules."""
import sys
errors = []

# 1. Test EfficiencyMonitor
try:
    from usnmexps.efficiency_monitor import EfficiencyMonitor
    m = EfficiencyMonitor()
    m.set_references(coupling_db=-8.0, fwd_v=0.04)
    m.start_session('test')
    r1 = m.check_pulse(fwd_v=0.040, rev_v=0.016)
    assert r1['status'] == 'OK', f"Expected OK, got {r1['status']}"
    r2 = m.check_pulse(fwd_v=0.040, rev_v=0.010)
    assert r2['status'] in ('WARNING', 'ERROR'), f"Expected alert, got {r2['status']}"
    summary = m.end_session()
    assert isinstance(summary, str) and 'measurements' in summary
    print('  [PASS] EfficiencyMonitor: check_pulse, set_references, sessions')
except Exception as e:
    errors.append(f'EfficiencyMonitor: {e}')
    print(f'  [FAIL] EfficiencyMonitor: {e}')

# 2. Test Calibrator factory
try:
    from usnmexps.calibrators import Calibrator
    assert hasattr(Calibrator, 'create_for_burst_check')
    assert hasattr(Calibrator, 'run_burst_efficiency_check')
    assert hasattr(Calibrator, 'setup_efficiency_monitoring')
    print('  [PASS] Calibrator: factory + burst check methods exist')
except Exception as e:
    errors.append(f'Calibrator: {e}')
    print(f'  [FAIL] Calibrator: {e}')

# 3. Test run_acquisition signature (on_pulse_callback kwarg)
try:
    import inspect
    sig = inspect.signature(Calibrator.run_acquisition)
    assert 'on_pulse_callback' in sig.parameters, 'on_pulse_callback not in signature'
    assert sig.parameters['on_pulse_callback'].default is None, 'default should be None'
    print('  [PASS] run_acquisition: on_pulse_callback param (default=None)')
except Exception as e:
    errors.append(f'run_acquisition sig: {e}')
    print(f'  [FAIL] run_acquisition sig: {e}')

# 4. Test pltutils backward compatibility
try:
    import inspect
    from usnmexps.pltutils import plot_calibration_data, plot_calibration_curves
    sig1 = inspect.signature(plot_calibration_data)
    assert 'details' in sig1.parameters, 'details param missing'
    assert 'mode' in sig1.parameters, 'mode param missing'
    assert sig1.parameters['details'].default == False
    assert sig1.parameters['mode'].default is None
    sig2 = inspect.signature(plot_calibration_curves)
    assert 'details' in sig2.parameters
    assert 'mode' in sig2.parameters
    print('  [PASS] pltutils: backward-compatible signatures (details + mode)')
except Exception as e:
    errors.append(f'pltutils: {e}')
    print(f'  [FAIL] pltutils: {e}')

# 5. Test USNMApp class structure
try:
    from usnmexps.usnm_app import USNMApp
    assert 'burst_check' in USNMApp.ACTIONS_DICT
    assert hasattr(USNMApp, 'on_burst_check')
    assert hasattr(USNMApp, 'setup_efficiency_monitoring_from_transducer')
    # Verify removed attributes are gone
    assert not hasattr(USNMApp, '_monitor_burst'), '_monitor_burst should be removed'
    assert not hasattr(USNMApp, 'setup_oscilloscope_for_bursts'), 'setup_oscilloscope_for_bursts should be removed'
    print('  [PASS] USNMApp: burst_check button, no dead methods')
except Exception as e:
    errors.append(f'USNMApp: {e}')
    print(f'  [FAIL] USNMApp: {e}')

# 6. Test CalibrationApp (uscalib)
try:
    from usnmexps.uscalib_app import CalibrationApp
    assert hasattr(CalibrationApp, 'on_check_efficiency')
    # Verify no EfficiencyMonitor import (we removed it)
    import usnmexps.uscalib_app as mod
    source = inspect.getsource(mod)
    assert 'from .efficiency_monitor import' not in source, 'Dead import should be removed'
    print('  [PASS] CalibrationApp: on_check_efficiency exists, no dead imports')
except Exception as e:
    errors.append(f'CalibrationApp: {e}')
    print(f'  [FAIL] CalibrationApp: {e}')

# 7. Test that coupling ratio is back in run_acquisition log path
try:
    source = inspect.getsource(Calibrator.run_acquisition)
    assert 'coupling ratio' in source, 'coupling ratio log line should be restored'
    assert 'check_pulse' in source, 'efficiency_monitor.check_pulse should still be called'
    print('  [PASS] run_acquisition: coupling ratio log + check_pulse both present')
except Exception as e:
    errors.append(f'run_acquisition content: {e}')
    print(f'  [FAIL] run_acquisition content: {e}')

# Summary
print()
if errors:
    print(f'FAILED: {len(errors)} check(s)')
    for e in errors:
        print(f'  - {e}')
    sys.exit(1)
else:
    print('ALL CHECKS PASSED')
    sys.exit(0)
