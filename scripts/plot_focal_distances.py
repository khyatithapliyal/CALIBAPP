# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2025-04-04 14:02:25
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-04-04 14:31:18

''' Plot focal distances of transducers. '''

from argparse import ArgumentParser
import matplotlib.pyplot as plt

from usnmexps.pltutils import plot_focal_distances
from usnmexps.utils import logger

if __name__ == '__main__':

    # Parse command line arguments
    parser = ArgumentParser(description='Plot focal distances of transducers.')
    parser.add_argument('--cbydate', default=False, action='store_true', help='color-code by date')
    args = parser.parse_args()

    # Plot focal distances
    try:
        fig = plot_focal_distances(cbydate=args.cbydate)
        plt.show()
    except ValueError as e:
        logger.error(e)
        quit()
