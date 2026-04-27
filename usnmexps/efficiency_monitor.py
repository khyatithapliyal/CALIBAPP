

import numpy as np
from datetime import datetime
from instrulink import logger




# Thresholds for alerts
COUPLING_WARNING_DB = 2.0       # Warn if coupling deviates by this much
COUPLING_ERROR_DB = 3.0         # Error if coupling deviates by this much
PRESSURE_WARNING_PCT = 30       # Warn if pressure deviates by this %
PRESSURE_ERROR_PCT = 50         # Error if pressure deviates by this %
FWD_VOLTAGE_WARNING_PCT = 20    # Warn if FWD voltage drops by this %




def vratio_to_db(ratio):
    """Convert voltage ratio to dB."""
    if ratio <= 0:
        return -np.inf
    return 20 * np.log10(ratio)


def percent_deviation(measured, reference):
    """Calculate percent deviation from reference."""
    if reference == 0:
        return np.inf
    return 100 * (measured - reference) / reference




class EfficiencyMonitor:
    """
    Unified efficiency monitor for calibration and experiments.
    
    Tracks coupling ratio (REV/FWD) and optional pressure measurements,
    compares against calibration references, and flags deviations.
    
    Usage:
        monitor = EfficiencyMonitor()
        monitor.set_references(coupling_db=-5.1, pressure_mpa=0.19, fwd_v=0.041)
        monitor.start_session('experiment')
        # After each measurement (from burst efficiency check or calibration):
        status = monitor.check_pulse(fwd_v=0.041, rev_v=0.023, pressure_mpa=0.19)
    """
    
    def __init__(self):
        # Reference values (from calibration)
        self.ref_coupling_db = None
        self.ref_pressure_mpa = None
        self.ref_fwd_v = None
        
        # Session tracking
        self.session_type = None  # 'calibration' or 'experiment'
        self.session_start = None
        self.measurements = []
        self.alerts = []
        
        # Baseline (first measurement of session)
        self.baseline_coupling_db = None
        self.baseline_fwd_v = None
        

    
    def set_references(self, coupling_db=None, pressure_mpa=None, fwd_v=None):
        """
        Set reference values from historical calibration.
        
        :param coupling_db: Reference coupling ratio in dB (REV/FWD)
        :param pressure_mpa: Reference pressure amplitude in MPa
        :param fwd_v: Reference forward voltage in V
        """
        if coupling_db is not None:
            self.ref_coupling_db = coupling_db
        if pressure_mpa is not None:
            self.ref_pressure_mpa = pressure_mpa
        if fwd_v is not None:
            self.ref_fwd_v = fwd_v
            
        logger.info(f'Efficiency monitor references set: '
                   f'coupling={self.ref_coupling_db} dB, '
                   f'pressure={self.ref_pressure_mpa} MPa, '
                   f'FWD={self.ref_fwd_v} V')
    
    def start_session(self, session_type='calibration'):
        """
        Start a new monitoring session.
        
        :param session_type: 'calibration' or 'experiment'
        """
        self.session_type = session_type
        self.session_start = datetime.now()
        self.measurements = []
        self.alerts = []
        self.baseline_coupling_db = None
        self.baseline_fwd_v = None
        
        logger.info(f'Efficiency monitoring session started: {session_type}')
    
    def end_session(self):
        """End session and return summary."""
        summary = self.get_summary()
        logger.info(f'Efficiency monitoring session ended. {summary}')
        return summary
    

    def check_pulse(self, fwd_v, rev_v, pressure_mpa=None, pulse_num=None):
        """
        Check efficiency for a single pulse (calibration mode).
        
        :param fwd_v: Forward coupling voltage (V)
        :param rev_v: Reverse coupling voltage (V)
        :param pressure_mpa: Measured pressure (MPa), if hydrophone available
        :param pulse_num: Optional pulse number for logging
        :return: dict with status and diagnostics
        """
        # Guard against invalid inputs (NaN from failed acquisitions, zero/negative)
        if not np.isfinite(fwd_v) or not np.isfinite(rev_v) or fwd_v <= 0:
            logger.warning(f'Invalid voltage inputs: FWD={fwd_v}, REV={rev_v}')
            return {'status': 'ERROR', 'message': 'Invalid FWD/REV voltage'}
        
        coupling_ratio = rev_v / fwd_v
        coupling_db = vratio_to_db(coupling_ratio)
        
        # Set baseline on first measurement
        if self.baseline_coupling_db is None:
            self.baseline_coupling_db = coupling_db
            self.baseline_fwd_v = fwd_v
        
        # Calculate deviations
        result = {
            'timestamp': datetime.now(),
            'fwd_v': fwd_v,
            'rev_v': rev_v,
            'coupling_ratio': coupling_ratio,
            'coupling_db': coupling_db,
            'pressure_mpa': pressure_mpa,
            'status': 'OK',
            'alerts': []
        }
        
        # Check coupling deviation from reference
        if self.ref_coupling_db is not None:
            coupling_dev = coupling_db - self.ref_coupling_db
            result['coupling_dev_db'] = coupling_dev
            
            if abs(coupling_dev) > COUPLING_ERROR_DB:
                result['alerts'].append(f'COUPLING ERROR: {coupling_dev:+.2f} dB')
                result['status'] = 'ERROR'
            elif abs(coupling_dev) > COUPLING_WARNING_DB:
                result['alerts'].append(f'COUPLING WARNING: {coupling_dev:+.2f} dB')
                if result['status'] == 'OK':
                    result['status'] = 'WARNING'
        
        # Check pressure deviation (dual-domain monitoring)
        if pressure_mpa is not None and self.ref_pressure_mpa is not None:
            pressure_dev_pct = percent_deviation(pressure_mpa, self.ref_pressure_mpa)
            result['pressure_dev_pct'] = pressure_dev_pct
            
            if abs(pressure_dev_pct) > PRESSURE_ERROR_PCT:
                result['alerts'].append(f'PRESSURE ERROR: {pressure_dev_pct:+.1f}%')
                result['status'] = 'ERROR'
            elif abs(pressure_dev_pct) > PRESSURE_WARNING_PCT:
                result['alerts'].append(f'PRESSURE WARNING: {pressure_dev_pct:+.1f}%')
                if result['status'] == 'OK':
                    result['status'] = 'WARNING'
            
            # CRITICAL: Diagnose problem type
            result['diagnosis'] = self._diagnose(result)
        
        # Store measurement
        self.measurements.append(result)
        if result['alerts']:
            self.alerts.extend(result['alerts'])
        
        # Log result
        pulse_str = f'pulse {pulse_num}: ' if pulse_num else ''
        pressure_str = f', P={pressure_mpa:.4f} MPa' if pressure_mpa else ''
        alert_str = f' [{", ".join(result["alerts"])}]' if result['alerts'] else ''
        
        logger.info(f'{pulse_str}FWD={fwd_v:.4f}V, REV={rev_v:.4f}V, '
                   f'ratio={coupling_ratio:.3f} ({coupling_db:.1f} dB){pressure_str}{alert_str}')
        
        return result

    def _diagnose(self, result):
        """
        Diagnose the type of problem based on which domain failed.
        
        Returns diagnosis dict with:
          - problem_type: 'none', 'electrical', 'acoustic', 'both'
          - likely_cause: string describing most likely cause
          - recommended_action: what to do about it
        """
        coupling_ok = abs(result.get('coupling_dev_db', 0)) <= COUPLING_ERROR_DB
        pressure_ok = abs(result.get('pressure_dev_pct', 0)) <= PRESSURE_ERROR_PCT
        
        if coupling_ok and pressure_ok:
            return {
                'problem_type': 'none',
                'likely_cause': 'System healthy',
                'recommended_action': 'None needed'
            }
        
        elif not coupling_ok and not pressure_ok:
            return {
                'problem_type': 'both',
                'likely_cause': 'Cable disconnected, connector loose, or transducer failure',
                'recommended_action': 'Check all cables and BNC connections'
            }
        
        elif coupling_ok and not pressure_ok:
            # THIS IS THE GEL BUBBLE CASE
            return {
                'problem_type': 'acoustic',
                'likely_cause': 'AIR BUBBLES IN GEL or poor acoustic coupling',
                'recommended_action': 'Remove gel, clean transducer face, reapply fresh gel carefully'
            }
        
        else:  # not coupling_ok and pressure_ok
            return {
                'problem_type': 'electrical',
                'likely_cause': 'Impedance mismatch or connector issue',
                'recommended_action': 'Check cable connections and transducer port'
            }
    
    def get_summary(self):
        """Get session summary string."""
        if not self.measurements:
            return 'No measurements recorded'
        
        n = len(self.measurements)
        n_ok = sum(1 for m in self.measurements if m.get('status') == 'OK')
        n_warn = sum(1 for m in self.measurements if m.get('status') == 'WARNING')
        n_err = sum(1 for m in self.measurements if m.get('status') == 'ERROR')
        
        # Calculate drift from baseline
        if len(self.measurements) > 1:
            first_coupling = self.measurements[0].get('coupling_db')
            last_coupling = self.measurements[-1].get('coupling_db')
            drift = last_coupling - first_coupling if first_coupling and last_coupling else 0
            drift_str = f', session drift: {drift:+.2f} dB'
        else:
            drift_str = ''
        
        status = 'PASS' if n_err == 0 else 'FAIL'
        
        return (f'{status}: {n} measurements ({n_ok} OK, {n_warn} warnings, {n_err} errors)'
                f'{drift_str}')



def get_coupling_reference_from_calibration(calibration_data, vpp):
    """
    Extract reference coupling ratio from calibration data at given voltage.
    
    :param calibration_data: DataFrame with CPLFWD and CPLREV columns
    :param vpp: Voltage setting (Vpp)
    :return: coupling ratio in dB
    """
    # Find closest voltage
    idx = (calibration_data['Vin (mVpp)'] - vpp * 1000).abs().idxmin()
    
    # Get FWD and REV values
    fwd_cols = [c for c in calibration_data.columns if 'CPLFWD' in c]
    rev_cols = [c for c in calibration_data.columns if 'CPLREV' in c]
    
    if not fwd_cols or not rev_cols:
        raise ValueError('Calibration data missing coupling columns')
    
    # Use most recent calibration
    fwd = calibration_data.loc[idx, fwd_cols[-1]]
    rev = calibration_data.loc[idx, rev_cols[-1]]
    
    ratio = rev / fwd
    return vratio_to_db(ratio), fwd, rev


def get_pressure_reference_from_calibration(calibration_data, vpp):
    """
    Extract reference pressure from calibration data at given voltage.
    
    :param calibration_data: DataFrame with pressure columns
    :param vpp: Voltage setting (Vpp)
    :return: pressure in MPa
    """
    # Find closest voltage
    idx = (calibration_data['Vin (mVpp)'] - vpp * 1000).abs().idxmin()
    
    # Get pressure columns
    pout_cols = [c for c in calibration_data.columns if 'Pout' in c and 'MPa' in c]
    
    if not pout_cols:
        raise ValueError('Calibration data missing pressure columns')
    
    # Use most recent calibration, average if multiple
    pressures = calibration_data.loc[idx, pout_cols[-3:]].mean()
    return pressures