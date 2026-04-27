# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2023-04-05 11:36:08
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-02-21 11:13:55

# External packages
from PyQt6.QtWidgets import QLabel, QPushButton
from PyQt6.QtGui import QColor, QPainter, QBrush
from PyQt6.QtCore import Qt
from instrulink.visa_instrument import VisaInstrument
from instrulink import logger

# Internal modules
from .constants import *


class LEDIndicator(QLabel):
    ''' Simple LED indicator widget '''
    
    def __init__(self, parent=None, color=Qt.GlobalColor.red):
        super().__init__(parent)
        self._color = color
        self.setFixedSize(20, 20)  # Set a small circle size

    def set_color(self, color):
        self._color = color
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QBrush(QColor(self._color)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(0, 0, self.width(), self.height())


class VirtualInstrument:
    ''' Virtual instrument interface to facilitate management within GUIs '''

    def __init__(self, ID, label, connector, testmode=False, lock=False):
        '''
        Constructor 

        :param ID: unique identifier         
        :param label: instrument label
        :param connector: instrument connection function
        :param disconnector: instrument disconnector function
        :param testmode: whether to use a dummy instrument
        '''
        self.ID = ID
        self.label = label
        self.connector = connector
        self.error = None
        self.obj = None
        self.iscon = False
        self.testmode = testmode
        self.lock = lock
        self.is_rendered = False

    def get_name(self):
        ''' Get instrument name '''
        if self.obj is not None:
            if isinstance(self.obj, str):
                return self.obj
            elif hasattr(self.obj, 'workspace'):
                return self.obj.matlab.engine.engineName()
        return self.obj.get_name()
    
    def get_formatted_name(self):
        ''' Get instrument name formatted for display '''
        if self.obj is None:
            if self.error is None:
                return UNKNOWN
            else:
                return f'Connection error: {self.error}'
        else:
            return self.get_name()

    def connect(self):
        ''' Attempt to connect to instrument '''
        logger.info(f'attempting to connect to {self.label}...')
        if self.testmode:
            self.obj = f'dummy {self.label}'
            logger.info(f'connected to dummy {self.label}')
        else:
            try:
                self.obj = self.connector()
                logger.info(f'connected to {self.get_name()}')
            except Exception as e:
                self.error = e
    
    def disconnect(self):
        ''' Disconnect from instrument '''
        logger.info(f'disconnecting from {self.get_name()}')
        if not self.testmode and self.obj is not None:
            if isinstance(self.obj, VisaInstrument) and hasattr(self.obj, 'disconnect'):
                self.obj.disconnect()
        self.obj = None

    def is_connected(self):
        ''' Assess whether the instrument is connected to the PC '''
        if self.obj is None:
            # logger.info(f'no {self.label} connected...')
            return False
        elif self.testmode:
            return True
        else:
            idn = self.get_name()
            if idn is None:
                logger.error(f'{self.label} got disconnected...')
                self.obj = None
            # else:
            #     logger.info(f'{label} {idn} still connected...')
            return idn is not None
    
    def render(self, parent):
        ''' Render as QT GroupBox '''
        self.qlabel = QLabel(self.label)
        self.connect_button = QPushButton('Connect')
        self.connect_button.clicked.connect(lambda: self.update(parent))
        self.led = LEDIndicator()
        self.id_label = QLabel(self.get_formatted_name())
        self.is_rendered = True
        return [
            self.qlabel,
            self.led,
            self.id_label,
            self.connect_button
        ]

    def update(self, parent):
        '''
        Update instrument status and ID upon connect or connection check
        '''
        if not self.is_rendered:
            raise ValueError(f'cannot update "{self.label}" virtual instrument: not rendered yet')
        # If instrument detected -> try to disconnect
        if self.iscon:
            self.disconnect()
        # Otherwise, try to connect
        else:
            self.connect()
        # Check connection in either case
        self.iscon = self.is_connected()
        # Adapt button text
        self.connect_button.setText('Disconnect' if self.iscon else 'Connect')
        # Adapt LED color
        self.led.set_color(Qt.GlobalColor.green if self.iscon else Qt.GlobalColor.red)
        # Adapt ID label
        self.id_label.setText(self.get_formatted_name())
        parent.on_instrument_update()
