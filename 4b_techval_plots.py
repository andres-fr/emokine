#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
Given the path to the technical validation data file as produced by
``4a_techval_compute.py``, this script renders and saves to disk the
corresponding validation plots.

When called on the ``EmokineDataset``, it produces the plots included under
``data/TechVal``, and used in the EMOKINE paper.
See the ``README``, and its companion script, ``4b_techval_compute.py``, for
more details.
"""


import os
import pickle
from collections import defaultdict
# for OmegaConf
from dataclasses import dataclass
from typing import Optional, Tuple
#
from omegaconf import OmegaConf, MISSING
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
#
from emokine.utils import PltFontManager, outlined_hist


# ##############################################################################
# # CLI
# ##############################################################################
@dataclass
class ConfDef:
    """
    :cvar EMOKINE_PATH: Path to the ``EmokineDataset``.
    :cvar OUTPUT_DIR: Where to store the rendered plots.
    :cvar HIST_MARGINS: List of ``(bottom, top, left, right, hspace, wspace)``
      margins, given as ratios.
    :cvar HIST_SELECTION: Arbitrary subset of sequences to be featured in the
      histogram selection, in the form ``(seq1_angry, seq7_neutral, ...)``

    The ``EMOTIONS`` and ``KEYPOINTS`` parameters are collections of strings
    that, by default, include all of ``EmokineDataset``. The ``KIN_FEATURES``
    and ``KIN_FEATURES_SINGLE`` also cover all the computed features, the only
    difference is that the latter is computed only once for the full body (i.e.
    it is not a function of the keypoint).
    """
    TECHVAL_PICKLE_PATH: str = MISSING
    OUTPUT_DIR: Optional[str] = None
    # filenames
    HIST_SEL_NAME: str = "histograms_selection.png"
    HIST_NAME: str = "histograms_{}.png"
    FG_NAME: str = "foreground_stats.png"
    KIN_NAME: str = "kinematics_{}.png"
    # kinematic selection
    EMOTIONS: Tuple[str] = ("angry", "content", "fearful", "joy", "neutral",
                            "sad")
    KEYPOINTS: Tuple[str] = ("Pelvis", "L5", "L3", "T12", "T8", "Neck", "Head",
                             "RightShoulder", "RightUpperArm", "RightForeArm",
                             "RightHand", "LeftShoulder", "LeftUpperArm",
                             "LeftForeArm", "LeftHand", "RightUpperLeg",
                             "RightLowerLeg", "RightFoot", "RightToe",
                             "LeftUpperLeg", "LeftLowerLeg", "LeftFoot",
                             "LeftToe"
                             )
    KIN_FEATURES: Tuple[str] = ("avg. vel.", "avg. accel.",
                                "avg. angular vel.", "avg. angular accel.",
                                "dimensionless jerk",
                                "avg. CoM dist.")
    KIN_FEATURES_SINGLE: Tuple[str] = (
        "avg. limb contraction",
        "avg. head angle (w.r.t. vertical)",
        "avg. head angle (w.r.t. back)",
        "avg. QoM",
        "avg. convex hull 2D",
        "avg. convex hull 3D")
    #
    HIST_SELECTION: Tuple[str] = ("seq1_angry", "seq1_neutral",
                                  "seq7_angry", "seq7_neutral")
    #
    DPI: int = 300
    TRANSPARENT_BG: bool = True
    PLT_TEXT_FONT: str = "Carlito"  # "Carlito" "Arial" "Open Sans"
    FIG_TITLE_SIZE: int = 50
    AXIS_TITLE_SIZE: int = 30
    AXIS_TICK_SIZE: int = 15
    UNITS_SIZE: int = 25
    LEGEND_SIZE: int = 14
    #
    HIST_MARGINS: Tuple[float] = (0.02, 0.85, 0.05, 0.95, 0.03, 0.08)
    FG_MARGINS: Tuple[float] = (0.08, 0.9, 0.07, 0.95, 0.03, 0.2)
    FG_ALPHA: float = 0.85
    #
    KIN_MARGINS: Tuple[float] = (0.08, 0.9, 0.18, 0.98, 0.12, 0.05)
    KIN_SINGLE_MARGINS: Tuple[float] = (0.08, 0.9, 0.08, 0.98, 0.3, 0.1)
    KIN_ALPHA: float = 0.85
    KIN_BINS: int = 50


# ##############################################################################
# # MAIN ROUTINE
# ##############################################################################
if __name__ == "__main__":
    CONF = OmegaConf.structured(ConfDef())
    cli_conf = OmegaConf.from_cli()
    CONF = OmegaConf.merge(CONF, cli_conf)
    print("\n\nCONFIGURATION:")
    print(OmegaConf.to_yaml(CONF), end="\n\n")

    # ##########################################################################
    # # GLOBAL CONFIG
    # ##########################################################################
    PltFontManager.set_font(CONF.PLT_TEXT_FONT)
    plt.rcParams["figure.titlesize"] = CONF.FIG_TITLE_SIZE
    plt.rcParams["axes.titlesize"] = CONF.AXIS_TITLE_SIZE
    plt.rcParams["xtick.labelsize"] = CONF.AXIS_TICK_SIZE
    plt.rcParams["ytick.labelsize"] = CONF.AXIS_TICK_SIZE
    plt.rcParams["legend.fontsize"] = CONF.LEGEND_SIZE
    #
    hist_margins = dict(zip(("bottom", "top", "left", "right", "hspace",
                             "wspace"), CONF.HIST_MARGINS))
    fg_margins = dict(zip(("bottom", "top", "left", "right", "hspace",
                           "wspace"), CONF.FG_MARGINS))
    kin_margins = dict(zip(("bottom", "top", "left", "right", "hspace",
                            "wspace"), CONF.KIN_MARGINS))
    kin_single_margins = dict(zip(("bottom", "top", "left", "right", "hspace",
                                   "wspace"), CONF.KIN_SINGLE_MARGINS))

    feature_titles = {
        "avg. vel.": "Mean Velocity",
        "avg. accel.": "Mean Acceleration",
        "avg. angular vel.": "Mean Angular Velocity",
        "avg. angular accel.": "Mean Angular Acceleration",
        "dimensionless jerk": "Dimensionless Jerk",
        "avg. limb contraction": "Mean Limb Contraction",
        "avg. CoM dist.": "Mean CoM Distance",
        "avg. head angle (w.r.t. vertical)":
        "avg. head angle (w.r.t. vertical)",
        "avg. head angle (w.r.t. back)":
        "avg. head angle (w.r.t. back)",
        "avg. QoM": "Mean Quantity of Motion",
        "avg. convex hull 2D": "Mean 2D Convex Hull",
        "avg. convex hull 3D": "Mean 3D Convex Hull"}

    feature_units = {
        "avg. vel.": r"$\frac{m}{s}$",
        "avg. accel.": r"$\frac{m}{s^2}$",
        "avg. angular vel.": r"$\frac{rad}{s}$",
        "avg. angular accel.": r"$\frac{rad}{s^2}$",
        "dimensionless jerk": "dimensionless",
        "avg. limb contraction": "m",
        "avg. CoM dist.": "m",
        "avg. head angle (w.r.t. vertical)":
        "Radians (w.r.t. vertical)",
        "avg. head angle (w.r.t. back)":
        "Radians (w.r.t. Back)",
        "avg. QoM": "dimensionless",
        "avg. convex hull 2D": "dimensionless",
        "avg. convex hull 3D": r"$m^3$"
    }

    cb_colors = sns.color_palette("colorblind", 10)

    # ##########################################################################
    # # LOAD TECHVAL DATA
    # ##########################################################################
    with open(CONF.TECHVAL_PICKLE_PATH, "rb") as f:
        techval = pickle.load(f)
    #
    sil_paths = techval["sil_paths"]
    avatar_paths = techval["avatar_paths"]
    pld_paths = techval["pld_paths"]
    campos_paths = techval["campos_paths"]
    kin_paths = techval["kin_paths"]
    #
    pld_ratios = techval["pld_ratios"]
    sil_ratios = techval["sil_ratios"]
    avatar_ratios = techval["avatar_ratios"]
    campos_ratios = techval["campos_ratios"]
    #
    sil_hists = techval["sil_hists"]
    pld_hists = techval["pld_hists"]
    avatar_hists = techval["avatar_hists"]
    campos_hists = techval["campos_hists"]
    #
    campos_xyxy_bounds = techval["campos_xyxy_bounds"]
    #
    kin_dfs = {p: pd.read_csv(p) for p in kin_paths}

    # ##########################################################################
    # # SILHOUETTE/PLD/AVATAR HISTOGRAMS
    # ##########################################################################
    def imshow_with_cbar(fig, ax, data, cmap="YlGnBu", cbar_ticks=None,
                         cbar_loc="right", cbar_orientation="vertical",
                         cbar_size="3%", cbar_pad=0.05):
        """
        Plots an ``imshow`` with a colorbar attached to the axis and with
        controlled relative size and position.
        """
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(cbar_loc, size=cbar_size, pad=cbar_pad)
        plot = ax.imshow(data, cmap=cmap)
        if plot is None:
            print("Warning: Empty plot!")
        else:
            fig.colorbar(plot, cax=cax, orientation=cbar_orientation,
                         ticks=cbar_ticks)

    def plot_histograms(idxs, cmap="YlGnBu", figsize=(20, 15),
                        title="Pixel Histograms", cbar_ticks=None):
        """
        Ad-hoc helper function that uses the global histograms (``sil_hists,
        campos_hists, ...``) to plot multiple histograms in their respective
        axes.
        :returns: The pair ``(fig, axes)`` with the plots.
        """
        fig, axes = plt.subplots(nrows=len(idxs), ncols=4, figsize=figsize)
        for row_i, idx in enumerate(idxs):
            # plot histograms as images
            imshow_with_cbar(fig, axes[row_i, 0], np.log(sil_hists[idx]),
                             cmap, cbar_ticks, "right", "vertical", "3%", 0.05)
            imshow_with_cbar(fig, axes[row_i, 1], np.log(campos_hists[idx]),
                             cmap, cbar_ticks, "right", "vertical", "3%", 0.05)
            imshow_with_cbar(fig, axes[row_i, 2], np.log(pld_hists[idx]),
                             cmap, cbar_ticks, "right", "vertical", "3%", 0.05)
            imshow_with_cbar(fig, axes[row_i, 3], np.log(avatar_hists[idx]),
                             cmap, cbar_ticks, "right", "vertical", "3%", 0.05)
            # remove ticks
            for ax in axes[row_i]:
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
            # set column titles
            axes[0, 0].set_title("Silhouette")
            axes[0, 1].set_title("CamPos Convex Hull")
            axes[0, 2].set_title("PLD")
            axes[0, 3].set_title("Avatar")
            # set row titles
            for i, idx in enumerate(idxs):
                lbl = os.path.splitext(os.path.basename(sil_paths[idx]))[0]
                lbl = "_".join(lbl.split("_")[1:])
                axes[i, 0].set_ylabel(lbl, rotation=90,
                                      fontsize=CONF.AXIS_TITLE_SIZE)
        #
        if title is not None:
            fig.suptitle(title)
        return fig, axes

    hist_selection_idxs = [[idx for idx, sp in enumerate(sil_paths)
                            if sel in sp][0] for sel in CONF.HIST_SELECTION]
    fig, _ = plot_histograms(hist_selection_idxs, figsize=(20, 12),
                             title=None, cbar_ticks=[1, 2, 3, 4, 5])
    fig.subplots_adjust(**hist_margins)

    if CONF.OUTPUT_DIR is None:
        fig.show()
        breakpoint()
    else:
        outpath = os.path.join(CONF.OUTPUT_DIR, CONF.HIST_SEL_NAME)
        fig.savefig(outpath, dpi=CONF.DPI, transparent=CONF.TRANSPARENT_BG)
        print("Saved figure to", outpath)

    full_idxs = [list(range(54))[i:i+12] for i in range(0, 54, 12)]
    for i, idxs in enumerate(full_idxs, 1):
        fig_w, fig_h = 20, (3 * len(idxs))
        fig, _ = plot_histograms(idxs, figsize=(fig_w, fig_h),
                                 title=None, cbar_ticks=[1, 2, 3, 4, 5])
        fig.subplots_adjust(**hist_margins)

        if CONF.OUTPUT_DIR is None:
            fig.show()
            breakpoint()
        else:
            outpath = os.path.join(CONF.OUTPUT_DIR, CONF.HIST_NAME.format(i))
            fig.savefig(outpath, dpi=CONF.DPI, transparent=CONF.TRANSPARENT_BG)
            print("Saved figure to", outpath)

    # ##########################################################################
    # # FOREGROUND STATS (PIXEL RATIO, BOUNDARIES)
    # ##########################################################################
    def plot_fg(bins=1000, brightness_range=(0, 0.05), alpha=0.7,
                figsize=(15, 5), top_margin=0.1, y_log=True):
        """
        Ad-hoc helper function that uses the global ratios (``sil_ratios,
        pld_ratios, ...``) to plot multiple histograms in their respective
        axes.
        :returns: The pair ``(fig, axes)`` with the plots.
        """
        lim_range = [0, 1]
        min_x, min_y, max_x, max_y = zip(*campos_xyxy_bounds)
        fig, axes = plt.subplots(ncols=3, figsize=figsize)
        #
        outlined_hist(axes[0], sil_ratios, bins=bins, range=brightness_range,
                      log=y_log, alpha=alpha, color=cb_colors[1],
                      label="Silhouette")
        outlined_hist(axes[0], pld_ratios, bins=bins, range=brightness_range,
                      log=y_log, alpha=alpha, color=cb_colors[2], label="PLD")
        outlined_hist(axes[0], avatar_ratios, bins=bins,
                      range=brightness_range, log=y_log, alpha=alpha,
                      color=cb_colors[4], label="Avatar")
        outlined_hist(axes[1], min_x, bins=bins, range=lim_range,
                      log=y_log, alpha=alpha, color=cb_colors[0],
                      label="Horiz. minima")
        outlined_hist(axes[1], max_x, bins=bins, range=lim_range,
                      log=y_log, alpha=alpha, color=cb_colors[5],
                      label="Horiz. maxima")
        outlined_hist(axes[2], min_y, bins=bins, range=lim_range,
                      log=y_log, alpha=alpha, color=cb_colors[0],
                      label="Vert. minima")
        outlined_hist(axes[2], max_y, bins=bins, range=lim_range,
                      log=y_log, alpha=alpha, color=cb_colors[5],
                      label="Vert. maxima")
        #
        axes[0].set_title("Foreground ratios")
        axes[1].set_title("CamPos Limits")
        axes[2].set_title("CamPos Limits")
        #
        axes[0].set_xlim(brightness_range)
        axes[1].set_xlim(lim_range)
        axes[2].set_xlim(lim_range)
        axes[1].set_xticks([0, 0.5, 1])
        axes[1].set_xticklabels(["left", "center", "right"])
        axes[2].set_xticks([0, 0.5, 1])
        axes[2].set_xticklabels(["bottom", "center", "top"])
        #
        ylabel = "Number of videos"
        if y_log:
            ylabel += " (scale: log)"
        axes[0].set_ylabel(ylabel, fontsize=CONF.AXIS_TITLE_SIZE)
        #
        for ax in axes:
            y0, y1 = ax.get_ylim()
            y1 *= (1 + top_margin)
            ax.set_ylim((y0, y1))
        #
        axes[0].legend(loc="upper right")
        axes[1].legend(loc="upper right")
        axes[2].legend(loc="upper right")
        #
        return fig, axes

    fig, axes = plot_fg(bins=50, brightness_range=(0, 0.05),
                        alpha=CONF.FG_ALPHA, top_margin=1.2)
    fig.subplots_adjust(**fg_margins)

    if CONF.OUTPUT_DIR is None:
        fig.show()
        breakpoint()
    else:
        outpath = os.path.join(CONF.OUTPUT_DIR, CONF.FG_NAME)
        fig.savefig(outpath, dpi=CONF.DPI, transparent=CONF.TRANSPARENT_BG)
        print("Saved figure to", outpath)

    # ##########################################################################
    # # PER-JOINT KINEMATICS
    # ##########################################################################
    def kinematic_histograms(kin_dfs, feature, keypoints, emotions,
                             num_bins=50, color=cb_colors[0], alpha=0.85,
                             figsize=(23, 36),
                             num_xticks=3, xtick_digits=2, xtick_rotation=0,
                             omit_last_tick=True,
                             xlabel_pos="center", xlabel_offset=75,
                             xlabel_bbox_edge="black",
                             xlabel_bbox_fill=(0.9, 0.9, 0.9)):
        """
        Ad-hoc helper function that uses the kinematic data to plot multiple
        histograms for a given kinematic ``feature``. The histograms are
        arranged such that there is one emotion per column, and one keypoint
        per row.
        :returns: The pair ``(fig, axes)`` with the plots.
        """
        fig, axes = plt.subplots(nrows=len(keypoints), ncols=len(emotions),
                                 figsize=figsize, sharex=True)
        # retrieve kinematic data by emotion and keypoint for this feature
        data = defaultdict(dict)
        for i, emotion in enumerate(emotions):
            for j, kp in enumerate(keypoints):
                data[emotion][kp] = np.array([df[df["keypoint"] == kp][feature]
                                              for k, df in kin_dfs.items()
                                              if emotion in k]).flatten()
        # plot histograms and gather hist modes
        modes = defaultdict(dict)
        x_ranges = []
        for i, emotion in enumerate(emotions):
            # get x-domain range for this row
            # min_x = min(min(v) for v in data[emotion].values())
            min_x = 0
            max_x = max(max(v) for v in data[emotion].values())
            x_range = (min_x, max_x)
            x_ranges.append(x_range)
            for j, kp in enumerate(keypoints):
                (counts, _, _), _ = outlined_hist(
                    axes[j][i], data[emotion][kp], bins=num_bins,
                    range=x_range, density=True,
                    alpha=alpha, color=color)
                modes[emotion][kp] = max(counts)
        # set ax row and column titles
        for i, emotion in enumerate(emotions):
            axes[0][i].set_title(emotion.capitalize())
        for i, kp in enumerate(keypoints):
            axes[i][0].set_ylabel(
                kp, rotation=0, ha="right", va="center",
                fontsize=CONF.AXIS_TITLE_SIZE)  # global!
        # remove unnecessary axis ticks/labels
        for i, axrow in enumerate(axes[::-1]):
            for j, ax in enumerate(axrow):
                if j >= 0:
                    ax.set_yticks([])
                    ax.set_yticks([], minor=True)
                if i >= 1:
                    ax.get_xaxis().set_visible(False)
        # set y ranges for histograms based on max mode
        max_mode = max([max(d.values()) for d in modes.values()])
        for axrow in axes:
            for ax in axrow:
                ax.set_ylim((0, max_mode * 1.05))

        # set axis x-labels and x-ticks
        x_mins, x_maxs = zip(*x_ranges)
        for ax in axes[-1]:
            xticks = np.linspace(min(x_mins), max(x_maxs),
                                 num_xticks).round(xtick_digits)
            if omit_last_tick:
                xticks = xticks[:-1]
            ax.set_xticks(xticks)
            for xtl in ax.get_xticklabels():
                xtl.set_rotation(xtick_rotation)
            xlbl = ax.set_xlabel(feature_units[feature], loc=xlabel_pos,
                                 labelpad=xlabel_offset,
                                 fontsize=CONF.UNITS_SIZE)  # global!
            xlbl.set_bbox(dict(facecolor=xlabel_bbox_fill,
                               edgecolor=xlabel_bbox_edge))
        #
        return fig, axes

    # end of 'kinematic_histograms' def
    for feat in CONF.KIN_FEATURES:
        # get plot
        fig, _ = kinematic_histograms(
            kin_dfs, feat, CONF.KEYPOINTS, CONF.EMOTIONS, CONF.KIN_BINS,
            cb_colors[0], CONF.KIN_ALPHA,
            figsize=(23, (1.5 * len(CONF.KEYPOINTS))),
            num_xticks=4, xtick_rotation=0, xlabel_offset=10)
        fig.suptitle(feature_titles[feat])
        fig.subplots_adjust(**kin_margins)

        # show/save plot
        if CONF.OUTPUT_DIR is None:
            fig.show()
            breakpoint()
        else:
            outname = CONF.KIN_NAME.format(
                feature_titles[feat].replace(" ", "_"))
            outpath = os.path.join(CONF.OUTPUT_DIR, outname)
            fig.savefig(outpath, dpi=CONF.DPI, transparent=CONF.TRANSPARENT_BG)
            print("Saved figure to", outpath)

    # ##########################################################################
    # # KINEMATICS (SINGLE)
    # ##########################################################################
    def kinematic_histograms_single(kin_dfs, features, emotions,
                                    num_bins=50, color=cb_colors[0],
                                    alpha=0.85, figsize=(23, 36),
                                    num_xticks=4, xtick_digits=2,
                                    xtick_rotation=0, omit_last_tick=True,
                                    xlabel_pos="center", xlabel_offset=15,
                                    xlabel_bbox_edge="black",
                                    xlabel_bbox_fill=(0.9, 0.9, 0.9)):
        """
        Variation of ``kinematic_histograms`` adapted for
        ``KIN_FEATURES_SINGLE``, where all features are in the same plot, with
        one emotion per column and one feature per row (since we don't need to
        plot one histogram per keypoint).
        :returns: The pair ``(fig, axes)`` with the plots.
        """
        fig, axes = plt.subplots(nrows=len(emotions), ncols=len(features),
                                 figsize=figsize, sharex=False)
        # retrieve kinematic data by emotion and keypoint for this feature
        data = defaultdict(dict)
        for i, emotion in enumerate(emotions):
            for j, feat in enumerate(features):
                data[emotion][feat] = np.array([df[feat][0]
                                                for k, df in kin_dfs.items()
                                                if emotion in k])

        # gather per-feature x ranges
        x_ranges = []
        for i, feat in enumerate(features):
            min_x = min(min(v[feat]) for v in data.values())
            max_x = max(max(v[feat]) for v in data.values())
            x_ranges.append((min_x, max_x))

        # plot histograms and gather hist modes
        modes = defaultdict(dict)
        for i, emotion in enumerate(emotions):
            for j, feat in enumerate(features):
                (counts, aa, bb), _ = outlined_hist(
                    axes[i][j], data[emotion][feat], bins=num_bins,
                    range=x_ranges[j], density=False,
                    alpha=alpha, color=color)
                modes[emotion][feat] = max(counts)

        # set ax row and column titles
        for i, feat in enumerate(features):
            if "(" in feat:
                feat = feat[:feat.index("(")]
            axes[0][i].set_title(feat, y=1.08)
        for i, emotion in enumerate(emotions):
            axes[i][0].set_ylabel(
                emotion, rotation=0, ha="right", va="center",
                fontsize=CONF.AXIS_TITLE_SIZE)  # global!

        # remove unnecessary axis ticks/labels
        for axrow in axes:
            for ax in axrow:
                ax.set_yticks([])
                ax.set_yticks([], minor=True)

        # set y ranges for histograms based on max mode
        max_mode = max([max(d.values()) for d in modes.values()])
        for axrow in axes:
            for ax in axrow:
                ax.set_ylim((0, max_mode * 1.05))

        # set axis x-labels and x-ticks
        for i, (feat,  (min_x, max_x)) in enumerate(zip(features, x_ranges)):
            xticks = np.linspace(min_x, max_x, num_xticks).round(xtick_digits)
            if omit_last_tick:
                xticks = xticks[:-1]
            for j in range(len(emotions)):
                ax = axes[j][i]
                ax.set_xlim(min_x, max_x)
                ax.set_xticks(xticks)
            xlbl = axes[-1][i].set_xlabel(
                feature_units[feat], loc=xlabel_pos,
                labelpad=xlabel_offset,
                fontsize=CONF.UNITS_SIZE)  # global!
            xlbl.set_bbox(dict(facecolor=xlabel_bbox_fill,
                               edgecolor=xlabel_bbox_edge))

        return fig, axes

    # get plot
    fig, _ = kinematic_histograms_single(
        kin_dfs, CONF.KIN_FEATURES_SINGLE, CONF.EMOTIONS, CONF.KIN_BINS,
        cb_colors[0], CONF.KIN_ALPHA,
        figsize=(27, (0.55 * len(CONF.KEYPOINTS))))
    fig.subplots_adjust(**kin_single_margins)

    # show/save plot
    if CONF.OUTPUT_DIR is None:
        fig.show()
        breakpoint()
    else:
        outname = CONF.KIN_NAME.format("single")
        outpath = os.path.join(CONF.OUTPUT_DIR, outname)
        fig.savefig(outpath, dpi=CONF.DPI, transparent=CONF.TRANSPARENT_BG)
        print("Saved figure to", outpath)
