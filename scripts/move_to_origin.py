# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-04-27 18:30:00
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-05-11 18:13:39

''' Move the micro-manipulator to its origin '''

from usnmexps.utils import check_conda_env
check_conda_env()

from instrulink import logger, grab_manipulator, SutterError

# Grab micro-manipulator and move to origin, and quit if failed
try:
    mp = grab_manipulator()
    mp.move_to_origin()
    mp.disconnect()
except SutterError as e:
    logger.error(e)
    quit()