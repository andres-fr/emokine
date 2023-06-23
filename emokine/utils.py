#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
General utilities.
"""


import os
import math
from datetime import datetime
import json
from ast import literal_eval as make_tuple
from math import floor, ceil
#
import pytz
import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
import torch
from torchvision.io import read_video
import skimage
from skimage.filters import threshold_multiotsu, \
    apply_hysteresis_threshold, median
import matplotlib.pyplot as plt
import matplotlib.font_manager as plt_fm


# ##############################################################################
# # IMAGE PROCESSING
# ##############################################################################
def get_lab_distance(img1, img2, out_dtype=np.float32):
    """
    :img1: Array of shape ``(h, w, 3)`` in RGB format
    :img2: Array of shape ``(h, w, 3)`` in RGB format
    :returns: Array of shape ``(h, w)`` with the LAB distances based on
      the CIEDE2000 specification ``skimage.color.deltaE_ciede2000``.
    """
    img1 = skimage.color.rgb2lab(img1)
    img2 = skimage.color.rgb2lab(img2)
    dist = skimage.color.deltaE_ciede2000(img1, img2).astype(out_dtype)
    return dist


def otsu_hist_median(arr, median_size=15):
    """
    :param arr: Heatmap with floats as an array of shape ``(h, w)``
    :returns:
    """
    num_unique_vals = len(np.unique(arr))
    if num_unique_vals <= 1:
        result = np.zeros_like(arr, dtype=bool)
    else:
        otsu_low_t, otsu_hi_t = threshold_multiotsu(arr)
        result = apply_hysteresis_threshold(arr, otsu_low_t, otsu_hi_t)
        result = median(result, footprint=np.ones((median_size, median_size)))
    return result


def make_elliptic_mask(msk, stretch=1.0):
    """
    Inspired by:
    https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
    """
    yyy, xxx = msk.nonzero()
    if len(yyy) == 0:
        return msk

    # extract first and second-order information
    center_xy = np.array([xxx.mean(), yyy.mean()])
    cov = np.cov(xxx, yyy)
    ew, ev = np.linalg.eigh(cov)
    stretch_xy = ew ** 0.5 * stretch * 4  # maybe there is a good reason for x4

    # draw ellipse with given resolution and rotate by corr
    patch_wh = np.int32(np.ceil(stretch_xy)) + 2
    patch = Image.new("L", tuple(patch_wh))
    draw = ImageDraw.Draw(patch)
    draw.ellipse([1, 1, *(patch_wh - 2)], fill="white")
    #
    angle = np.degrees(np.arctan2(*(ev @ [0, 1])))
    patch = patch.rotate(angle, expand=True)
    patch = patch.crop(patch.getbbox())

    # paste
    result = Image.new("1", msk.T.shape)
    patch_wh_half = np.float32(patch.size) / 2
    patch_xy = [int(x) for x in (center_xy - patch_wh_half).round()]
    result.paste(patch, patch_xy)
    result = np.array(result)
    # plt.clf(); plt.imshow(msk.astype(np.uint8) + result); plt.show()
    return result


def resize_crop_bbox(x0, x1, y0, y1,
                     min_x=0, max_x=None, min_y=0, max_y=None,
                     expansion_ratio=1.0):
    """
    Given a bounding box as float coordinates, this function (optionally)
    resizes it around its center, rounds the result to integers and
    (optionally) clips the result within given boundaries.
    The ``x0, x1, y0, y1`` parameters represent the input bounding box
    coordinates. The optional ``min`` and ``max`` parameters clip the
    output bounding box coordinates. Note that the clipping parameters will
    themselves be rounded towards 0 before clipping is applied.
    """
    center_x, center_y = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    w, h = x1 - x0, y1 - y0
    w_exp_half = expansion_ratio * w * 0.5
    h_exp_half = expansion_ratio * h * 0.5
    #
    x0 = round(center_x - w_exp_half)
    x1 = round(center_x + w_exp_half)
    y0 = round(center_y - h_exp_half)
    y1 = round(center_y + h_exp_half)
    #
    if min_x is not None:
        x0 = max(x0, ceil(min_x))
    if max_x is not None:
        x1 = min(x1, floor(max_x))
    if min_y is not None:
        y0 = max(y0, ceil(min_y))
    if max_y is not None:
        y1 = min(y1, floor(max_y))
    #
    return (x0, x1, y0, y1)


def resize_hw(t, new_hw):
    """
    :param t: Tensor of shape ``(h, w)``
    :returns: Tensor of shape ``new_hw``.
    """
    input_type = t.dtype
    t = t.type(torch.float32).unsqueeze(0).unsqueeze(0)
    t = torch.nn.functional.interpolate(t, new_hw, mode="bilinear")[0, 0]
    t = t.type(input_type)
    return t


# ##############################################################################
# # STRING PROCESSING
# ##############################################################################
def make_timestamp(timezone="Europe/London", with_tz_output=True):
    """
    Output example: day, month, year, hour, min, sec, milisecs:
    10_Feb_2018_20:10:16.151
    """
    ts = datetime.now(tz=pytz.timezone(timezone)).strftime(
        "%Y_%m_%d_%H:%M:%S.%f")[:-3]
    if with_tz_output:
        return "%s(%s)" % (ts, timezone)
    else:
        return ts


def delta_str_to_seconds(delta_ts, ts_format="%H:%M:%S.%f"):
    """
    :param delta_ts: A string representing duration in the given ``ts_format``.
    :returns: A float corresponding to the duration in ``delta_ts`` as seconds.
    """
    dt = datetime.strptime(delta_ts, ts_format) - datetime.strptime("0", "%S")
    result = dt.total_seconds()
    return result


def seconds_to_hhmmssxxx_str(seconds):
    """
    :param float seconds: Float number in seconds
    :returns: String in the form ``..hh:mm:ss.xxx``. Note that hh can have
      more than 2 digits if enough seconds given. The ``xxx`` miliseconds are
      always given in 3 digits, rounded.
    """
    hours, rest = divmod(seconds, 3600)
    minutes, rest = divmod(rest, 60)
    rest = round(rest, 3)
    seconds, miliseconds = divmod(round(rest * 1000), 1000)
    result = "{:02d}:{:02d}:{:02d}.{:03d}".format(
        int(hours), int(minutes), int(seconds), round(miliseconds, 3))
    return result


# ##############################################################################
# # I/O
# ##############################################################################
def load_imgs(dir_path, out_hw=None, extension=".png", verbose=False):
    """
    """
    img_paths = sorted([os.path.join(dir_path, x)
                        for x in os.listdir(dir_path)
                        if x.endswith(extension)])
    #
    imgs = []
    ori_imgs = []
    for ip in img_paths:
        if verbose:
            print("loading", ip)
        image = Image.open(ip)
        arr = np.array(image)
        ori_imgs.append(arr)
        if out_hw is not None:
            out_h, out_w = out_hw
            image = image.resize((out_w, out_h))
            arr = np.array(image)
            imgs.append(arr)
    #
    return imgs, ori_imgs


class JsonBlenderPositions:
    """
    The Blender script renders ``world_to_camera_view`` coordinates for
    a given MVNX sequence and saves them as JSON. This class loads the
    JSON file into a ``self.data`` pandas DataFrame for convenient processing.
    """

    def __init__(self, json_path, which_pos="sphere_pos"):
        """
        :param str which_pos: One of ``sphere_pos, bone_head, bone_tail``.
        """
        with open(json_path, "r") as f:
            j = json.load(f)
            self.fps = int(j[0]["frame_rate"])
            self.pos_explanation = j[0]["pos_explanation"]
            #
            flattened = []
            for frame in j[1:]:
                flat = self.flatten_frame(frame, which_pos)
                flattened.append(flat)
            self.data = pd.DataFrame(flattened)
            #
            self.path = json_path
            self.which_pos = which_pos

    @staticmethod
    def flatten_frame(frame, which_pos="sphere_pos"):
        """
        The frame is expected to be a dictionary with a ``frame`` field
        (ignored) and other string fields in the form ``"(kp_name, ico_name)"``
        each one of them containing a dictionary with ``pos->[x, y, z]``
        positions.

        The ``which_pos`` parameter tells which one of those positions to take.
        """
        flat = {f"{make_tuple(k)[0]}_{coord}": v[which_pos][i]
                for k, v in frame.items() if k != "frame"
                for i, coord in enumerate(["x", "y", "z"])}
        return flat


def load_bw_vid(path, ignore_below=1):
    """
    :param path: Assumed to be the path to a black and white video, so that
      a threshold exactly between the "black" and the "white" color values.
    :param ignore_below: If a given frame in the video has less than this
      number of entries above threshold, the frame will be skipped
    :returns: A boolean numpy array of shape ``(frames, h, w)``.
    """
    vid, _, fpsdict = read_video(path, pts_unit="pts")
    fps = fpsdict["video_fps"]
    # Convert vid(frames, h, w, ch) to boolean(frames, h, w)
    result = []
    for i, f in enumerate(vid, 1):
        thresh = f.max().item() / 2
        f_bool = (f > thresh).any(dim=-1)
        num_true = f_bool.sum().item()
        if num_true >= ignore_below:
            result.append(f_bool.numpy())
        else:
            print(i, "WARNING: ignored frame with", num_true, "active!")
    result = np.stack(result)
    return result, fps


def load_bg_vid(path, bg_rgb=[0, 0, 0], margin_rgb=[0, 0, 0],
                ignore_below=1, ignore_above=None):
    """
    :param path: Assumed to be the path to a color video, with a background
      corresponding to the ``bg_rgb`` color, +/- margin.
    :param ignore_below: If a given frame in the video has less than this
      number of detected non-background pixels, the frame will be skipped
    :param ignore_below: If a given frame in the video has more than this
      number of detected non-background pixels, (e.g. if the background is not
      static) the frame will be skipped. Optional.
    :returns: A boolean numpy array of shape ``(frames, h, w)``, where the
      pixels close to ``bg_rgb`` are false, and true otherwise.
    """
    vid, _, fpsdict = read_video(path, pts_unit="pts")
    fps = fpsdict["video_fps"]
    t, h, w, c = vid.shape
    bg_rgb = torch.tensor(bg_rgb)
    assert c == 3 == len(bg_rgb), "Expected color (RGB) video!"
    #
    r_lo, r_hi = bg_rgb[0] - margin_rgb[0], bg_rgb[0] + margin_rgb[0]
    g_lo, g_hi = bg_rgb[1] - margin_rgb[1], bg_rgb[1] + margin_rgb[1]
    b_lo, b_hi = bg_rgb[2] - margin_rgb[2], bg_rgb[2] + margin_rgb[2]
    #
    result = (r_lo <= vid[:, :, :, 0])
    result &= (r_hi >= vid[:, :, :, 0])
    result = (g_lo <= vid[:, :, :, 1])
    result &= (g_hi >= vid[:, :, :, 1])
    result = (b_lo <= vid[:, :, :, 2])
    result &= (b_hi >= vid[:, :, :, 2])
    result = (~result).numpy()
    #
    num_fg = [(i, x.sum()) for i, x in enumerate(result)]
    idxs = [(i, x) for i, x in num_fg if x >= ignore_below]
    if ignore_above is not None:
        idxs = [(i, x) for i, x in idxs if x <= ignore_above]
    idxs, _ = zip(*idxs)
    result = result[idxs, :, :]
    # import matplotlib.pyplot as plt
    # plt.clf(); plt.imshow(result[10]); plt.show()
    return result, fps


# ##############################################################################
# # DS
# ##############################################################################
def find_nearest_sorted(array, value):
    """
    * WARNING::
      ``array`` must be sorted!

    From https://stackoverflow.com/a/26026189
    """
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or
                    math.fabs(value - array[idx-1]) <
                    math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]


def str_to_vec(x):
    """
    Converts a node with a text like '1.23, 2.34 ...' into a list
    like [1.23, 2.34, ...]
    """
    try:
        return [float(y) for y in x.text.split(" ")]
    except Exception as e:
        print("Could not convert to vector (skip conversion):", e)
        return x


def split_list_in_equal_chunks(l, chunk_size):
    """
    """
    return [l[i:i + chunk_size] for i in range(0, len(l), chunk_size)]


# ##############################################################################
# # PLOTTING
# ##############################################################################
class PltFontManager:
    """
    Sometimes matplotlib finds the system font paths, but setting them can
    still be challenging due to using the wrong name or matplotlib complaining
    about missing elements.

    This manager static class is intended to aid with that, by facilitating
    the found paths and their corresponding font names, and providing a
    ``set_font`` convenience method that will provide the allowed names if
    the given one didn't work out.
    """
    @staticmethod
    def get_font_paths():
        """
        :returns: List of paths to the fonts found by matplotlib in the system.
        """
        result = plt_fm.findSystemFonts()
        return result

    @classmethod
    def get_font_names(cls):
        """
        :returns: A tuple ``(fontnames, errors)``, where ``fontnames`` is a
          list with the valid font names that can be used, and ``errors`` is
          a dictionary in the form ``font_name: error`` containing fonts that
          couldn't be successfully loaded.
        """
        fpaths = cls.get_font_paths()
        fnames = set()
        errors = {}
        for fp in fpaths:
            try:
                fname = plt_fm.FontProperties(fname=fp).get_name()
                fnames.add(fname)
            except Exception as e:
                errors[fp] = e
        #
        return fnames, errors

    @classmethod
    def set_font(cls, font_name="Liberation Mono"):
        """
        :param font_name: A valid font name as the ones retrieved by the
          ``get_font_names`` method

        This method attempts to set the given font. If that is not possible,
        it will inform the user and provide a list of available fonts.
        """
        try:
            plt_fm.findfont(font_name, fallback_to_default=False)
            plt.rcParams["font.family"] = font_name
        except ValueError as ve:
            print(ve)
            print("Available fonts:")
            print(sorted(cls.get_font_names()[0]))


def outlined_hist(ax, data, **hist_kwargs):
    """
    Histogram with outlined borders
    """
    hist_kwargs.pop("histtype", None)
    lines1 = ax.hist(data, histtype="stepfilled", **hist_kwargs)
    #
    hist_kwargs.pop("alpha", None)
    hist_kwargs.pop("color", None)
    hist_kwargs.pop("label", None)
    lines2 = ax.hist(data, alpha=1, histtype="step", color="black",
                     **hist_kwargs)
    #
    return lines1, lines2
