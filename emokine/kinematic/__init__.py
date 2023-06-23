# -*- coding:utf-8 -*-


"""
This module contains kinematic feature functionality that is also used
elsewhere.
"""


import numpy as np
from shapely.geometry import MultiPoint
from scipy.spatial import ConvexHull


# ##############################################################################
# # MAD COMPUTATION
# ##############################################################################
def median_absdev(arr):
    """
    Returns the MAD from a given array (dimensions will be flattened).
    MAD(x) is defined as the median(abs(x-median(x)))
    """
    flat = arr.flatten()
    result = np.median(np.absolute(flat - np.median(flat)))
    return result


# ##############################################################################
# # CONVEX HULL COMPUTATION
# ##############################################################################
def get_2d_convex_hull(data_row):
    """
    :param data_row: A 1D numpy array, expected to be a succession of
      x, y, z values.
    :returns: ``(area, ch)``, Where area of the 2D convex hull is formed
      by the x,y points, as a ratio, where 1 means covering the full area
      and 0 no area. The ``ch`` is the full ConvexHull object
    """
    xvals = data_row[0::3]
    yvals = data_row[1::3]
    xyvals = np.stack([xvals, yvals]).T
    # ch = ConvexHull(xyvals)
    # area = ch.volume  # for 2D, volume is actually surface
    mp = MultiPoint(xyvals)
    area = mp.convex_hull.area
    return area, mp


def get_3d_convex_hull(data_row):
    """
    :param data_row: A 1D numpy array, expected to be a succession of
      x, y, z values.
    :returns: ``(area, ch)``, Where area of the 2D convex hull is formed
      by the x,y points, as a ratio, where 1 means covering the full area
      and 0 no area. The ``ch`` is the full ConvexHull object
    """
    xvals = data_row[0::3]
    yvals = data_row[1::3]
    zvals = data_row[2::3]
    xyzvals = np.stack([xvals, yvals, zvals]).T
    ch = ConvexHull(xyzvals)
    volume = ch.volume
    mp = MultiPoint(xyzvals)
    return volume, mp
