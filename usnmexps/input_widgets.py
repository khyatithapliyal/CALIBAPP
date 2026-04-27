# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2025-02-11 15:38:07
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-06-12 15:21:38


# External packages
from PyQt6.QtWidgets import QSpinBox, QDoubleSpinBox, QGroupBox, QGridLayout, QLabel, QLineEdit, QHBoxLayout
import numpy as np

# Internal modules
from .constants import *
from .utils import today


class IntegerInput:
    ''' Integer input for GUI applications'''

    def __init__(self, value, min=None, max=None):
        self.value = value
        self.min = min
        self.max = max
    
    def to_dict(self):
        return {'value': self.value, 'min': self.min, 'max': self.max}
    
    def render(self):
        ''' Render as QSpinBox widget for Qt application '''
        w = QSpinBox()
        w.setMinimum(np.iinfo(np.int32).min if self.min is None else self.min)
        w.setMaximum(np.iinfo(np.int32).max if self.max is None else self.max)
        w.setValue(self.value)
        return w


class FloatInput:
    ''' Float input for GUI applications '''

    def __init__(self, value, min=None, max=None, step=.01):
        self.value = value
        self.min = min
        self.max = max
        self.step = step
    
    def to_dict(self):
        return {'value': self.value, 'min': self.min, 'max': self.max, 'step': self.step}
    
    def render(self):
        ''' Render as QDoubleSpinBox widget for Qt application '''
        w = QDoubleSpinBox()
        w.setSingleStep(self.step)
        x = int(np.floor(np.log10(self.step)))
        if x < 0:
            w.setDecimals(-x)
        w.setMinimum(-np.inf if self.min is None else self.min)
        w.setMaximum(np.inf if self.max is None else self.max)
        w.setValue(self.value)
        return w
    

class StringInput:

    def __init__(self, value):
        self.value = value

    @property
    def value(self):
        if self._value == '\\today':
            return today()
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
    
    def to_dict(self):
        return {'value': self.value}
    
    def render(self):
        ''' Render as QLineEdit widget for Qt application '''
        return QLineEdit(self.value)


class InputsTable:
    ''' Inputs table for GUI applications '''
    def __init__(self, title, inputs):
        self.title = title
        self.inputs = inputs
    
    def render(self):
        # Initialize widget dictionary
        wdict = {}
        # Start grid layout
        grid = QGridLayout()
        # Go through each input
        for i, (label, inputs) in enumerate(self.inputs.items()):
            # Add label on new row, first column
            grid.addWidget(QLabel(label), i, 0)
            
            # If widget is a list, wrap in HBox and add box in 2nd column
            if isinstance(inputs, list):
                wdict[label] = []
                wbox = QHBoxLayout()
                for j, item in enumerate(inputs):
                    w = item.render()
                    w.setMaximumWidth(40)
                    wdict[label].append(w)
                    wbox.addWidget(w)
                grid.addLayout(wbox, i, 1)
            
            # Otherwise, add widget in 2nd column
            else:
                w = inputs.render()
                grid.addWidget(w, i, 1)
                wdict[label] = w
        
        # Assign layout to group box
        group = QGroupBox(self.title)
        group.setLayout(grid)

        # Return widget dictionary and group box
        return wdict, group


# --------------------------- INPUT WIDGETS ---------------------------

CALIB_STIM_PARAMS = InputsTable('Stimulus parameters', {
    'channels [trig.] [sig.]': [
        IntegerInput(STIM_CH_TRIG, min=1, max=2),
        IntegerInput(STIM_CH_SIGNAL, min=1, max=2),
    ],
    F_KEY: FloatInput(DEFAULT_CARRIER_FREQ, min=0, step=0.001),
    VIN_KEY: FloatInput(REF_VPP_CALIB / MV_TO_V, min=0, step=100),
    PRF_KEY: FloatInput(DEFAULT_PRF, min=0, step=0.1),
    NCYCLES_PER_PULSE_KEY: IntegerInput(NCYCLES_CALIB, min=0), 
})

CALIB_ACQ_PARAMS = InputsTable('Acquisition parameters', {
    'probe attenuation (dB)': FloatInput(0, max=0, step=1),
    'hydrophone sensitivity (V/MPa)': FloatInput(0., min=0, step=1e-4),
    'channels [trig.] [sig.] [cplfwd.] [cplrev.]': [
        IntegerInput(ACQ_CH_TRIG, min=1, max=4),
        IntegerInput(ACQ_CH_SIGNAL, min=1, max=4),
        IntegerInput(ACQ_CH_CPLFWD, min=1, max=4),
        IntegerInput(ACQ_CH_CPLREV, min=1, max=4),
    ],
    # '# points / acq': IntegerInput(OSC_ACQ_NPOINTS, min=1),
    'inter-acq int. (s)': FloatInput(OSC_ACQ_INTERVAL, min=0, step=.05),
    '# sweeps / acq': IntegerInput(OSC_ACQ_NSWEEPS, min=0)
})

CALIB_SCAN_PARAMS = InputsTable('Scanning parameters', {
    'XZ angle (°)': FloatInput(THETAY_SCAN, step=.1),
    'XYZ range (mm)': [
        FloatInput(DELTA_SCAN['X'] / MM_TO_UM, min=0, step=.1),
        FloatInput(DELTA_SCAN['Y'] / MM_TO_UM, min=0, step=.1),
        FloatInput(DELTA_SCAN['Z'] / MM_TO_UM, min=0, step=.1)
    ],
    'resolution (mm)': FloatInput(RES_SCAN / MM_TO_UM, min=0, step=.1),
    '# focus iters': IntegerInput(NITERS_SEARCH, min=1, max=4),
})


USNM_ACQ_PARAMS = InputsTable('Acquisition parameters', {
    'session name': StringInput('\\today'),
    'sampling rate (Hz)': FloatInput(EXP_SR, min=0),
    '# frames / acq': IntegerInput(EXP_NFRAMES, min=1),
    '# acqs / cond': IntegerInput(EXP_NACQS, min=1),
    'stim delay (s)': FloatInput(EXP_STIMDELAY, min=0),
})