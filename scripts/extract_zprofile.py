# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2023-10-05 16:04:22
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-02-14 11:47:16

from usnmexps.utils import check_conda_env
check_conda_env()

import os
import matplotlib.pyplot as plt
from glob import glob
from instrulink import logger

from usnmexps.config import MAPPING_DATA_FOLDER
from usnmexps.calib_utils import parse_transducer_calibration_file, compare_calibration_and_mapping_params, extract_relative_amp


def get_mapping_file_from_calib_file(calib_fpath, hydrophone):
    '''
    Infer full path to mapping file corresponding to a calibration file
    and hydrophone model.

    :param_calib_fpath: full path to calibration file
    :param hydrophone: hydrophone model
    :param ztarget (optional): target working distance (m)
    '''
    calib_file = os.path.basename(calib_fpath)
    pdict = parse_transducer_calibration_file(calib_file)
    calib_fcode = os.path.splitext(os.path.basename(calib_file))[0]
    mapping_dir = f'{MAPPING_DATA_FOLDER}/transducer_{pdict["transducer_id"]}_{pdict["conditions"]}/{pdict["freq (MHz)"]:.3f}MHz/{hydrophone}'
    mapping_file_pattern = f'{calib_fcode}_mapping_{hydrophone.split("_")[-1]}_*_thetaY15.00deg_21x21y49z.csv'
    mapping_file = os.path.basename(glob(f'{mapping_dir}/{mapping_file_pattern}')[0])
    return f'{mapping_dir}/{mapping_file}'


# Calibration file(s) and associated hydrophones
calib_dict = {
    'transducer_4-18-23#1_glass2layers_2.053MHz.xlsx': 'Onda_HNR500',
    'transducer_10-5-23#3_glass2layers_2.059MHz.xlsx': 'Onda_HNR500',
}

# Target depth(s) (mm)
ztargets = [
    3.4,
    4.35
]

# Loop over calibration files and hydrophones
for calib_file, hydrophone in calib_dict.items():
    # Loop over target depths
    for ztarget in ztargets:
        # Extract and plot relative pressure amplitude profile along the z-axis
        try:
            # Infer mapping file path from calibration file path and hydrophone model
            mapping_fpath = get_mapping_file_from_calib_file(calib_file, hydrophone)

            # Verify that selected calibration file parameters (transducer ID, US frequency,
            # acquisition conditions, ...) match those of reference mapping file
            common_params = compare_calibration_and_mapping_params(calib_file, mapping_fpath)
            logger.info(f'transducer calibration parameters: \n{common_params}')
            rel_Ptarget = extract_relative_amp(mapping_fpath, ztarget, plot=True, offset=0.5)
            logger.info(rel_Ptarget)
        except ValueError as e:
            logger.error(e)
            quit()


plt.show()
