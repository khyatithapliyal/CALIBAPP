# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-03-07 17:11:50
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-03-13 17:36:39

''' Display table of gains (in dB) and amplitude ratios '''

from usnmexps.utils import check_conda_env
check_conda_env()

import numpy as np
import pandas as pd
from usnmexps.calib_utils import vratio_to_gain
from usnmexps.constants import *

# Compute gains for for a set of reference ratiosamp_
amp_ratios = np.array([.01, .02, .05, .1, .2, .5, 1., 2., 5., 10., 20., 50., 100.])
gains = vratio_to_gain(amp_ratios)

# To dataframe
df = pd.DataFrame({
    'Amplitude ratio': amp_ratios, 
    GAIN_KEY: gains
})
print(df)