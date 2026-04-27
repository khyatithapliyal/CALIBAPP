# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-03-15 18:08:26
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-03-27 13:01:02

# External packages
import os
import time
import pandas as pd
import logging
from functools import wraps
from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QWidget, QLabel, QPushButton, QMessageBox, QGridLayout, QLineEdit, QGroupBox, QFileDialog, QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QHBoxLayout
from PyQt6.QtCore import QThread, QObject, pyqtSignal, QSize
from PyQt6.QtGui import QResizeEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Internal modules
from instrulink.logger import logger
from .calib_utils import *
from .constants import *
from .utils import update_last_file, flatten_dict


def get_worker(obj, method, *args, **kwargs):
    ''' Get worker object for a given method '''
    # Define worker class
    class Worker(QObject):  
        ''' Worker class to run the method in a separate thread '''

        # Signal to emit when the method is finished
        finished = pyqtSignal()

        def __init__(self):
            ''' Constructor '''
            logger.debug(f'starting {self}')
            super().__init__()
            # Control flag for stopping execution
            self._running = True
        
        def __str__(self):
            return f'{self.__class__.__name__}({str(obj)}.{method.__name__})'

        def run(self):
            ''' Run the method '''
            # Pass self to allow stopping
            logger.debug(f'running {self}')
            method(obj, self, *args, **kwargs)  
            self.finished.emit()  

        def stop(self):
            ''' Stop execution '''
            logger.debug(f'stopping {self}')
            self._running = False  # Set flag to stop execution
        
    # Create and return worker instance
    worker = Worker()
    return worker


def start_background_action(obj, key, method, *args, **kwargs):
    '''
    Start app action in background Qt thread

    :param obj: app object
    :param key: action key
    :param method: action method
    :param args: method arguments
    :param kwargs: method keyword arguments
    '''
    # Start action on GUI
    obj.start_action(key)    
    
    # Create thread and worker to execute action method in background
    obj.background_thread = QThread()
    def mymethod(*x, **y):
        method(*x, **y)
        obj.stop_action(key)
    obj.background_worker = get_worker(obj, mymethod, *args, **kwargs)

    # Move worker to thread
    obj.background_worker.moveToThread(obj.background_thread)
    obj.background_thread.started.connect(obj.background_worker.run)

    # Set up actions to be taken when thread finishes
    obj.background_worker.finished.connect(obj.background_thread.quit)
    obj.background_worker.finished.connect(obj.background_worker.deleteLater)
    obj.background_thread.finished.connect(obj.background_worker.deleteLater)

    # Link end of execution to post-action method, if any
    post_method = getattr(obj, f'post_{key}', None)
    if post_method is not None:
        obj.background_thread.finished.connect(post_method)

    # Ensure objects are fully removed when the thread finishes
    obj.background_thread.finished.connect(lambda: setattr(obj, 'background_thread', None))
    obj.background_thread.finished.connect(lambda: setattr(obj, 'background_worker', None))

    # Start thread
    obj.background_thread.start()


def stop_background_action(obj, key):
    ''' 
    Stop app action running in background Qt thread
    
    :param obj: app object
    :param key: action key
    '''
    # Send stop signal to worker
    obj.send_stop_signal(key)
    
    # Stop worker and thread
    obj.background_worker.stop()
    obj.background_thread.quit()
    obj.background_thread.wait()
    
    # Clear thread and worker attributes
    obj.background_thread = None
    obj.background_worker = None
    
    # If action is still running on GUI, stop it
    if obj.is_action_running(key):
        obj.stop_action(key)
    

def get_method_key(method):
    ''' Get method key from method name '''
    # Get everything after the first underscore in the method name
    return method.__name__.split('_', maxsplit=1)[-1]


def exec_pre_method(key, obj):
    ''' 
    Execute pre-method (if any) before main method
    
    :param key: action key
    :param obj: app object
    :return: pre-method result, i.e. whether to proceed with main method
    '''
    pre_method = getattr(obj, f'pre_{key}', None)
    if pre_method is not None:
        return pre_method()
    return True


def exec_post_method(key, obj):
    ''' 
    Execute post-method (if any) after main method
    
    :param key: action key
    :param obj: app object
    '''
    post_method = getattr(obj, f'post_{key}', None)
    if post_method is not None:
        return post_method()


def lock_actions(stoppable=True, unlock_on_finish=True, background=False):
    ''' 
    Decorator that wraps an action method to lock GUI action buttons during its execution,
    with various behavior options.

    :param stoppable: whether the action can be stopped from the GUI during
        its execution (default: True)
    :param unlock_on_finish: whether to unlock action buttons that where locked 
        when the method finishes (default: True)
    :param background: whether to run the method in a background thread (default: False)
    :return: wrapped method
    '''
    def decorator(method):
        ''' Inner method decorator '''
        # Get key
        key = get_method_key(method)

        # Error message prefix
        errprefix = f'{method.__name__} decoration error:'

        # Case 1: action not stoppable
        if not stoppable:
            # If background flag on, raise error
            if background:
                raise ValueError(
                    f'{errprefix} background option unavailable for non-stoppable actions')
            
            # Case 1a: buttons kept locked on on finish
            if not unlock_on_finish:
                @wraps(method)
                def wrapper(obj, *args, **kwargs):
                    obj.start_action(key, stoppable=False)
                    if exec_pre_method(key, obj):
                        result = method(obj, *args, **kwargs)
                    else:
                        result = None
                    exec_post_method(key, obj)
                    return result
                return wrapper
        
            # Case 1b: buttons unlocked on finish
            else:
                @wraps(method)
                def wrapper(obj, *args, **kwargs):
                    obj.start_action(key, stoppable=False)
                    if exec_pre_method(key, obj):
                        result = method(obj, *args, **kwargs)
                    else:
                        result = None
                    exec_post_method(key, obj)
                    obj.stop_action(key)
                    return result
                return wrapper
        
        # Case 2: action stoppable, buttons unlocked on finish
        else:
            # If not unlock-on-finish, raise error
            if not unlock_on_finish:
                raise ValueError(
                    f'{errprefix} unlock-on-finish option is mandatory for stoppable actions')

            # Case 2a: run in background
            if background:
                @wraps(method)
                def wrapper(obj, *args, **kwargs):
                    if obj.background_thread is not None and obj.background_thread.isRunning():
                        logger.warning(f'stopping background "{key}" action')
                        stop_background_action(obj, key)        
                    else:
                        if exec_pre_method(key, obj):
                            start_background_action(obj, key, method, *args, **kwargs)
                return wrapper
        
            # Case 2b: run in main thread
            else:
                @wraps(method)
                def wrapper(obj, *args, **kwargs):
                    if obj.is_action_running(key):
                        logger.warning(f'stopping "{key}" action')
                        obj.stop_action(key)
                        return
                    else:
                        obj.start_action(key)
                        if exec_pre_method(key, obj):
                            result = method(obj, *args, **kwargs)
                        else:
                            result = None
                        exec_post_method(key, obj)
                        if obj.is_action_running(key):
                            obj.stop_action(key)
                    return result
                return wrapper

    # Return inner decorator 
    return decorator


class ResizableFigureCanvas(FigureCanvas):
    ''' Generic class for a resizable matplotlib figure canvas '''

    def __init__(self, figure=None, parent=None):
        ''' Constructor '''
        super().__init__(figure=figure)
        self.setParent(parent)
 
    def resize_figure(self):
        ''' Resize figure to fit canvas dimensions '''
        w = self.size().width()
        h = self.size().height()
        self.figure.set_size_inches(
            w / self.figure.dpi, h / self.figure.dpi)
        self.adjust_figure_layout()
    
    def adjust_figure_layout(self):
        self.figure.tight_layout()
        self.figure.subplots_adjust(
            bottom=0.2,
            wspace=0.3,
            right=.9,
        )
    
    def trigger_resize(self):
        self.resizeEvent(QResizeEvent(self.size(), QSize()))

    def resizeEvent(self, event):
        ''' Resize event handler '''
        if self.figure is not None:
            self.resize_figure()
        super().resizeEvent(event)


class PopupCanvas(ResizableFigureCanvas):
    ''' Pop-up canvas for plotting matplotlib figure on a separate window '''

    def __init__(self, fig, name='plot', on_close=None, parent=None):
        super().__init__(figure=fig, parent=parent)
        self.setWindowTitle(name)
        self.draw()
        if on_close is None:
            self.on_close = lambda: None
        else:
            self.on_close = on_close
    
    def adjust_figure_layout(self):
        pass
    
    def closeEvent(self, event):
        ''' Close event handler '''
        self.close()
        self.deleteLater()
        super().closeEvent(event)
        self.on_close()


class QTextEditLogger(logging.Handler):
    ''' Custom logging handler to redirect logs to a QTextEdit widget with color formatting. '''

    def __init__(self, text_edit):
        super().__init__()
        self.text_edit = text_edit
    
    def detect_dark_mode(self):
        ''' Detects if the application is in dark mode based on palette background. '''
        bg_color = self.text_edit.palette().color(self.text_edit.backgroundRole()).name()
        return bg_color in ['#000000', '#121212', '#1E1E1E', '#323232']  # Common dark mode colors

    def emit(self, record):
        msg = self.format(record)
        color = self.get_color(record.levelname)
        formatted_msg = f'<span style="color:{color};">{msg}</span>'
        self.text_edit.append(formatted_msg)
        self.text_edit.verticalScrollBar().setValue(self.text_edit.verticalScrollBar().maximum())  # Auto-scroll

    def get_color(self, levelname):
        ''' Returns color codes for different log levels. '''
        colors = {
            'DEBUG': 'gray',
            'INFO': 'white' if self.detect_dark_mode() else 'black',
            'WARNING': 'orange',
            'ERROR': 'red',
            'CRITICAL': 'darkred',
        }
        return colors.get(levelname, 'black')  # Default to black if level is unknown


class CustomDialog(QDialog):
    ''' Generic dialog window with OK and Cancel buttons '''

    def __init__(self, parent=None):
        ''' Constructor '''
        # Initialize parent class
        super().__init__(parent=parent)

        # Add window title
        self.setWindowTitle(self.__class__.__name__)

        # Set layout    
        self.box = QVBoxLayout()
        self.add_widgets()
        self.add_action_buttons()
        
        # Set layout
        self.setLayout(self.box)
    
    def add_widgets(self):
        ''' Add widgets to dialog '''
        return NotImplementedError

    def add_action_buttons(self):
        ''' Add OK and Cancel buttons '''
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.box.addWidget(self.button_box)
    
    def get_values(self):
        ''' Retrieve input values '''
        return NotImplementedError


class VirtualBenchApp(QWidget):
    ''' Generic interface for PyQt-based desktop apps to control lab instruments '''

    STOPTXT = 'STOP'

    def __init__(self, instruments, testmode=False):
        '''
        Initialize app
        
        :param instruments: list of virtual instruments
        :param testmode: whether to run the app in test mode (no instrument needed)
        '''
        # Assign attributes
        self.testmode = testmode
        self.instruments = {v.ID: v for v in instruments}
        for k in self.instruments.keys():
            self.instruments[k].testmode = self.testmode

        # Log 
        logger.info(f'starting {str(self)} in {self.modestr} mode')

        # Initialize parent class and set window title to class name
        super().__init__()
        self.setWindowTitle(str(self))

        # Set up data storage dictionaries
        self.fwidget = {}
        self.fselect = {}
        self.fcreate = {}
        self.fpaths = {}
        self.fdata = {}

        # Set up placeholders for background processes
        self.background_thread = None
        self.background_worker = None

        # Set layout
        self.set_layout()

        # Set up logging, if applicable
        self.setup_logging()

        # Register callbacks
        self.register_callbacks()

        # Update instruments status
        self.update_instruments_status()

        # Placeholder for pop-up canvas
        self.popup_canvas = None
    
    def __str__(self):
        ''' String representation of the app '''
        return self.__class__.__name__
    
    def update_instruments_status(self):
        ''' Update instruments status '''
        for instrument in self.instruments.values():
            instrument.update(self)
    
    def show_popup(self, fig, name='plot', **kwargs):
        ''' Show a pop-up window with a matplotlib figure '''
        # If pop-up canvas already exists, close it
        if self.popup_canvas is not None:
            self.popup_canvas.close()
            self.popup_canvas = None
        
        # Show plot in pop-up window
        self.popup_canvas = PopupCanvas(fig, name=name, **kwargs)
        self.popup_canvas.show()
    
    def set_layout(self):
        return NotImplementedError

    def register_callbacks(self):
        return NotImplementedError

    def connect_actions(self):
        ''' Connect action buttons to their respective methods '''
        for k, v in self.actions_btns.items():
            v.clicked.connect(getattr(self, f'on_{k}'))
    
    @property
    def modestr(self):
        ''' String representation of the app running mode '''
        return {True: 'test', False: 'normal'}[self.testmode]
    
    def file_input(self, label, file_filter, add_create=False, **kwargs):
        ''' Create a file input widget '''
        self.fwidget[label] = QLineEdit()
        self.fwidget[label].setReadOnly(True)
        self.fselect[label] = QPushButton('Select')
        self.fselect[label].clicked.connect(lambda: self.select_input_file_dialog(label, file_filter, **kwargs))
        self.fpaths[label] = None
        self.fdata[label] = None
        outs = [QLabel(label), self.fwidget[label], self.fselect[label]]
        if add_create:
            self.fcreate[label] = QPushButton('Create')
            self.fcreate[label].clicked.connect(lambda: self.create_file_dialog(label, **kwargs))
            outs.append(self.fcreate[label])
        return outs

    def update_input_file(self, key, fpath):
        ''' Update file input widget '''
        if fpath is None:
            return None
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f'file not found: "{fpath}"')
        self.fpaths[key] = fpath
        self.fdata[key] = self.load_dataframe(fpath)
        self.fwidget[key].setText(os.path.basename(fpath))
        if fpath is not None:
            update_last_file(key, fpath)        
        return fpath
    
    def select_file_dialog(self, file_filter, initialdir='', title='Select File'):
        ''' Open a file dialog to select a file '''
        fpath, _ = QFileDialog.getOpenFileName(
            self,
            title,
            initialdir,
            file_filter
        )
        if fpath == '':
            fpath = None
        return fpath
    
    def select_input_file_dialog(self, key, *args, **kwargs):
        ''' Open a file dialog to select a file '''
        fpath = self.select_file_dialog(*args, **kwargs)
        return self.update_input_file(key, fpath)

    def select_folder_dialog(self, initialdir=''):
        out = QFileDialog.getExistingDirectory(
            self,
            'Select Folder',
            initialdir,
        )
        if out == '':
            return None
        return out
    
    def create_file_dialog(self, key, initialdir=''):
        ''' Open a file dialog to create a file '''
        dirpath = self.select_folder_dialog(initialdir=initialdir)
        if dirpath == '':
            dirpath = None
        return self.create_input_file(key, dirpath)
    
    def create_input_file(self, key, dirpath):
        return NotImplementedError
    
    def clear_file_input(self, key):
        ''' Clear file input widget and associated data '''
        self.fpaths[key] = None
        self.fdata[key] = None
        self.fwidget[key].setText('')
    
    def input_files_panel(self, settings_per_ftype):
        '''       
        Create input files selection panel
        
        :param settings_per_ftype: dictionary of settings per file type, containing for each type a 3-tuple of:
            - file filter
            - initial directory
            - whether to add a 'Create' button

            Example:
            {
                Type1: ('CSV files (*.csv)', '/path/to/dir1/', True),
                Type2: ('Excel files (*.xls *.xlsx)', '/path/to/dir2/', False),
                ...
            }
        :return: Qt group box
        '''
        # Create file input widgets for each file type
        group = QGroupBox('Input Files')
        grid = QGridLayout()
        for i, (ftype, (ffilter, initialdir, add_create)) in enumerate(settings_per_ftype.items()):
            widgets = self.file_input(ftype, ffilter, initialdir=initialdir, add_create=add_create)
            for j, widget in enumerate(widgets):
                grid.addWidget(widget, i, j)
        # Set layout and return group
        group.setLayout(grid)
        return group

    def df_to_table(self, df, table=None):
        '''
        Convert a dataframe to a HTML table
        
        :param df: dataframe
        :param table (optional): Qt table widget to update
        :return: Qt table 
        '''
        if table is None:
            table = QTableWidget()
        if df is None:
            return table
        table.setRowCount(df.shape[0])
        table.setColumnCount(df.shape[1])
        table.setHorizontalHeaderLabels(df.columns)
        for row in range(df.shape[0]):
            for col in range(df.shape[1]):
                item = QTableWidgetItem(str(df.iat[row, col]))
                table.setItem(row, col, item)
        table.resizeColumnsToContents()
        return table

    def actions_panel(self, d):
        '''
        Create panel with action buttons, from a dictionary of action types
        
        :param d: dictionary of (action ID: action label) pairs 
        '''
        # Create horizontal layout
        layout = QHBoxLayout()
        
        # Initialize action buttons dictionary
        self.actions_btns = {}
        # For each input dict entry
        for k, v in d.items():
            # If value is a dictionary, create a sub-group
            if isinstance(v, dict):
                sublayout = QVBoxLayout()
                for kk, vv in v.items():
                    self.actions_btns[kk] = QPushButton(vv)
                    sublayout.addWidget(self.actions_btns[kk])
                subgroup = QGroupBox(k)
                subgroup.setLayout(sublayout)
                layout.addWidget(subgroup)
            # Otherwise, create a single button
            else:
                self.actions_btns[k] = QPushButton(v)
                layout.addWidget(self.actions_btns[k])

        # Embed in groupbox
        group = QGroupBox('Actions')
        group.setLayout(layout)

        # Return
        return group
    
    def instruments_panel(self):
        '''
        Return instruments panel 
        
        :return: Qt group box
        '''
        group = QGroupBox('Instruments')
        grid = QGridLayout()
        for i, instrument in enumerate(self.instruments.values()):
            for j, qwidget in enumerate(instrument.render(self)):
                grid.addWidget(qwidget, i, j)
        group.setLayout(grid)
        return group
    
    def on_instrument_update(self):
        ''' Update instrument status and ID upon connect or connection check '''
        return self.update_actions()
    
    def update_actions(self):
        ''' Update action buttons '''
        return NotImplementedError

    def get_dataframe_file_type(self, fname):
        ''' Parse dataframe file type from file name '''
        if fname.endswith('.csv'):
            ftype = 'CSV'
        elif fname.endswith('.xls') or fname.endswith('.xlsx'):
            ftype = 'Excel'
        else:
            # Raise Error if invalid file type
            raise ValueError(f'invalid file type: "{fname.split(".")[-1]}"')
        return ftype

    def select_cols(self, fname, df, columns=None):
        ''' Select specific columns in dataframe '''
        if columns is None:
            return df
        absentcols = list(filter(lambda k: k not in df.columns, columns))
        if absentcols:
            raise ValueError(f'columns [{", ".join(absentcols)}] not found in "{fname}"')
        return df.loc[:, columns]

    def load_dataframe(self, fpath, columns=None):
        '''
        Load dataframe from CSV or Excel file
        
        :param fpath: path to dataframe file
        :return: parsed dataframe
        '''
        # Parse file type
        filename = os.path.abspath(fpath)
        ftype = self.get_dataframe_file_type(filename)
        
        # Load dataframe content appropriately depending on file type
        if ftype == 'CSV':
            df = pd.read_csv(fpath)
        elif ftype == 'Excel':
            df = pd.read_excel(fpath, engine='openpyxl')
        
        # Select specific columns, and return parsed dataframe
        return self.select_cols(filename, df, columns=columns)

    def instruments_status(self):
        ''' Assess connection statusof each virtual instrument '''
        return {k: v.is_connected(testmode=self.testmode) for k, v in self.instruments.items()}
    
    def yesno_dialog(self, question):
        ''' Display a yes/no dialog and get user response '''
        reply = QMessageBox.question(
            self, 'Confirmation', question,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes
        )
        return reply == QMessageBox.StandardButton.Yes
    
    def log_panel(self):
        ''' Creates a QTextEdit widget to display logs. '''
        group = QGroupBox('Log')
        layout = QVBoxLayout()
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)  # Make it read-only
        layout.addWidget(self.log_output)
        group.setLayout(layout)
        return group

    def setup_logging(self):
        ''' Sets up the logger to output to the QTextEdit widget. '''
        if not hasattr(self, 'log_output') or self.log_output is None:
            return
        log_handler = QTextEditLogger(self.log_output)
        log_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S:'))
        logger.addHandler(log_handler)

    def set_enable(self, w, enable):
        ''' Enable or disable a widget '''
        if not isinstance(enable, bool):
            raise ValueError(f'invalid enable value: {enable}')
        if isinstance(w, dict):
            for ww in w.values():
                self.set_enable(ww, enable)
            return
        if isinstance(w, list):
            for ww in w:
                self.set_enable(ww, enable)
            return
        w.setEnabled(enable)

    def disable_actions(self, *keys):
        ''' Disable all/specific action buttons '''
        if len(keys) == 0:
            keys = list(self.actions_btns.keys())
        for k in keys:
            self.actions_btns[k].setEnabled(False)
        self.disabled_actions = keys
    
    def enable_actions(self, *keys):
        ''' Enable all/specific action buttons '''
        if len(keys) == 0:
            keys = list(self.actions_btns.keys())
        for k in keys:
            self.actions_btns[k].setEnabled(True)
    
    @property
    def to_lock(self):
        return []
    
    def set_enable_inputs(self, enable):
        ''' Toggle enabled state of all app input widgets '''
        self.set_enable(self.to_lock, enable)
        for v in self.instruments.values():
            self.set_enable(v.connect_button, enable)
    
    def is_action_running(self, key):
        ''' Check whether an action is currently running '''
        return self.actions_btns[key].text() == self.STOPTXT

    def start_action(self, key, stoppable=True):
        ''' Start an action and disable action buttons and input widgets '''
        # Gather keys for action buttons to lock
        lock_keys = [k for k, v in self.actions_btns.items() if v.isEnabled()]

        # If action is stoppable, update button text and color, 
        # and remove key from lock list
        if stoppable:
            self.actions_btns[key].setText(self.STOPTXT)
            self.actions_btns[key].setStyleSheet('QPushButton {background-color: red}')
            lock_keys = [k for k in lock_keys if k != key]
        
        # Lock action buttons and disable input widgets
        self.disable_actions(*lock_keys)
        self.set_enable_inputs(False)
    
    @property
    def actions_dict(self):
        ''' Dictionary of action buttons '''
        return flatten_dict(self.ACTIONS_DICT)

    def stop_action(self, key):
        ''' Stop an action and re-enable action buttons and input widgets'''
        self.actions_btns[key].setText(self.actions_dict[key])
        self.actions_btns[key].setStyleSheet('')
        self.enable_actions(*self.disabled_actions)
        self.disabled_actions = []
        self.set_enable_inputs(True)

    def send_stop_signal(self, key):
        ''' Stop background thread '''
        self.runproc = False

    @lock_actions(background=True)
    def on_sleep(self, *args, **kwargs):
        ''' Test callback for sleeping 2 seconds '''
        logger.info('going to sleep')
        self.runproc = True
        ttot, dt = 2, 0.1
        for i in range(int(ttot / dt)):
            time.sleep(dt)
            if not self.runproc:
                break
        logger.info('waking up')