#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
This module contains reusable functionality to perform technical validation
of EMOKINE-style datasets.
"""


from multiprocessing import Pool
from itertools import repeat
#
import numpy as np
from shapely.ops import unary_union  # union of polygons
import skimage
#
from .utils import load_bw_vid, load_bg_vid, JsonBlenderPositions
from .kinematic import get_2d_convex_hull


# ##############################################################################
# # COMPUTE BRIGHTNESS RATIOS AND DENSITIES FOR BW VIDEOS
# ##############################################################################
def techval_bw_vid(path, ignore_below=0):
    """
    Given a path to a video, assumed to be black-and-white, returns a pair
    ``(vid_ratio, histogram)``, where the former is the number of non-black
    pixels divided by the total pixels (across the whole video), and the
    latter is a single integer frame which, for each pixel, contains the
    number of times it was non-black across the whole sequence.

    :param ignore_below: Any frames with less than this many non-black pixels
      will be ignored.
    """
    vid, vid_fps = load_bw_vid(path, ignore_below)
    vid_ratio = vid.sum() / vid.size
    histogram = vid.sum(axis=0)
    return vid_ratio, histogram


def techval_bw_vid_multi(paths, num_processes=1, ignore_below=0):
    """
    Multiprocessing version of ``techval_bw_vid``, accepts multiple paths and
    distributes them across multiple processes.

    See ``techval_bw_vid`` for parameter descriptions.
    """
    with Pool(num_processes) as pool:
        ratios, hist = zip(
            *pool.starmap(techval_bw_vid, zip(paths, repeat(ignore_below))))
    return ratios, hist


# ##############################################################################
# # COMPUTE BRIGHTNESS RATIOS AND DENSITIES FOR AVATAR VIDEOS
# ##############################################################################
def techval_avatar_vid(path, bg_rgb=(208, 240, 241), bg_rgb_margin=(1, 1, 1),
                       ignore_below=100, ignore_above=100_000):
    """
    Avatar videos have a flat background, but they are not black-and-white.
    This function converts them to binary by identifying the given color range
    as background (black/false) and everything else as foreground (white/true).

    :param path: Path to the avatar video.
    :param bg_rgb: RGB color of the background (uint8 triple).
    :param bg_rgb_margin: Any pixel of ``bg_rgb`` color, plus minus this range,
      will be considered background. Set to ``(0, 0, 0)`` for precise
      extraction.
    :param ignore_below: Any frames with less than this many foreground pixels
      will be ignored.
    :param ignore_above: Any frames with more than this many foreground pixels
      will be ignored.
    """
    vid, vid_fps = load_bg_vid(
        path, bg_rgb, bg_rgb_margin, ignore_below, ignore_above)
    vid_ratio = vid.sum() / vid.size
    histogram = vid.sum(axis=0)
    return vid_ratio, histogram


def techval_avatar_vid_multi(paths, bg_rgb=(208, 240, 241),
                             bg_rgb_margin=(1, 1, 1),
                             ignore_below=100, ignore_above=100_000,
                             num_processes=1):
    """
    Multiprocessing version of ``techval_avatar_vid``, accepts multiple paths
    and distributes them across multiple processes.

    See ``techval_avatar_vid`` for parameter descriptions.
    """
    aa, bb = techval_avatar_vid(paths[0], bg_rgb, bg_rgb_margin)
    with Pool(num_processes) as pool:
        ratios, hist = zip(*pool.starmap(techval_avatar_vid, zip(
            paths, repeat(bg_rgb), repeat(bg_rgb_margin),
            repeat(ignore_below), repeat(ignore_above))))
    return ratios, hist


# ##############################################################################
# # COMPUTE AREA RATIOS, DENSITIES AND BOUNDS FOR CAMPOS DATA
# ##############################################################################
def techval_campos(path, output_hw):
    """
    :param path: Path to a CamPos JSON file as the ones present in ``EMOKINE``
      (and/or the ones produced by the ``emokine`` scripts).
    :returns: A tuple ``(poly_ratio, hist, (min_x, min_y, max_x, max_y))``.

    The CamPos data presents the human keypoints in the sequence from the
    perspective of a given camera. This allows us to compute bounding 2D
    polygons, as well as the leftmost, rightmost... etc positions given that
    camera perspective.

    The returned values are respectively: The ratio of polygon surface divided
    by total surface, a single integer frame which, for each pixel, contains
    the number of times it was inside a polygon across the whole sequence, and
    the extremal values found in the whole sequence.

    See the ``TechVal`` plots in the ``EMOKINE`` dataset for an example of
    the outcome.
    """
    out_h, out_w = output_hw
    jbp = JsonBlenderPositions(path, which_pos="sphere_pos")
    #
    polygons = [get_2d_convex_hull(jbp.data.iloc[i].array)[1].convex_hull
                for i in range(len(jbp.data))]
    seq_hull = unary_union(polygons)
    min_x, min_y, max_x, max_y = seq_hull.bounds
    #
    hist = np.zeros(output_hw, dtype=np.int64)
    for poly in polygons:
        xxx, yyy = poly.boundary.xy
        yyy, xxx = skimage.draw.polygon((np.array(yyy) * out_h).round(),
                                        (np.array(xxx) * out_w).round())
        hist[yyy, xxx] += 1
    #
    poly_ratio = sum(p.area for p in polygons) / len(polygons)
    hist = hist[::-1]  # flip vertical dimension to go from top to bottom
    return poly_ratio, hist, (min_x, min_y, max_x, max_y)


def techval_campos_multi(paths, output_hw, num_processes=1):
    """
    Multiprocessing version of ``techval_campos``, accepts multiple paths
    and distributes them across multiple processes.

    See ``techval_campos`` for parameter descriptions.
    """
    with Pool(num_processes) as pool:
        ratios, hist, xyxy_bounds = zip(
            *pool.starmap(techval_campos, zip(paths, repeat(output_hw))))
    return ratios, hist, xyxy_bounds
