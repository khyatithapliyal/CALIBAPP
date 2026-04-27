# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2025-02-10 16:18:29
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-03-27 13:09:49

''' Entry script to run US calibration & mapping desktop app '''

# External packages
import sys
from argparse import ArgumentParser
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon
from instrulink.logger import logger

# Internal modules
from usnmexps.uscalib_app import CalibrationApp

def main():
    ''' Main function '''
    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument(
        '-t', '--test', default=False, action='store_true', help='Run in test mode (no instrument)')
    args = parser.parse_args()

    # Instantiate app
    app = QApplication([])

    # Set app icon and stylesheet
    app.setStyleSheet('QGroupBox::title {font-weight: bold;}')
    app.setWindowIcon(QIcon('assets/calibapp.ico'))

    # Instantiate and display main window
    window = CalibrationApp(testmode=args.test)
    window.show()

    # Execute the app and gather exit code
    exit_code = app.exec()
    if exit_code == 0:
        logger.info(f'{window} exited normally')
    else:
        logger.error(f'Error: {window} exited with code', exit_code)

    # Exit with exit code
    sys.exit(exit_code)

if __name__ == '__main__':
    main()
