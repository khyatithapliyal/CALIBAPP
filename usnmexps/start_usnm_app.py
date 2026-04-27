# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2025-02-10 16:18:29
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-02-21 15:38:11

''' Entry script to run USNM desktop app '''

# External packages
import sys
from argparse import ArgumentParser
from PyQt6.QtWidgets import QApplication

# Internal modules
from usnmexps.usnm_app import USNMApp

def main():
    ''' Main function '''
    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument(
        '-t', '--test', default=False, action='store_true', help='Run in test mode (no instrument)')
    args = parser.parse_args()

    # Start app
    app = QApplication(sys.argv)
    app.setStyleSheet('QGroupBox::title {font-weight: bold;}')
    window = USNMApp(testmode=args.test)
    window.show()

    # Exit handler
    sys.exit(app.exec())


if __name__ == '__main__':
    main()