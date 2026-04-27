# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2025-02-18 12:50:10
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-02-18 12:51:58

# Select files from user dialog
from usnmexps.dialog import open_file_dialog
from usnmexps.config import CALIBRATION_DATA_FOLDER
from usnmexps.calib_utils import convert_calibration_file

# Open dialog to select files
fpaths = open_file_dialog('xlsx', dirname=CALIBRATION_DATA_FOLDER, multiple=True)

# If files were selected, convert each of them
if fpaths is not None:
    for fpath in fpaths:
        convert_calibration_file(fpath)