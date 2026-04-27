# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-06-08 12:03:07
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-05-11 15:34:50

''' List all available VISA resources '''

from usnmexps.utils import check_conda_env
check_conda_env()

from instrulink import list_visa_resources

list_visa_resources()