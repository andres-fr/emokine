#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
Given the path to the ``EmokineDataset``, this script computes the data
used to produce the technical validation plots, and saves it as a pickle file.
Technical validation data is useful to explore and understand general
properties of the dataset, like e.g. the distribution between foreground
and background pixels.

See the ``README``, and its companion script, ``4b_techval_plots.py``, for
more details.
"""


import glob
import os
import pickle
# for OmegaConf
from dataclasses import dataclass
from typing import Optional, Tuple
#
from omegaconf import OmegaConf, MISSING
#
from emokine.utils import load_bw_vid
from emokine.technical_validation import techval_bw_vid_multi
from emokine.technical_validation import techval_avatar_vid_multi
from emokine.technical_validation import techval_campos_multi
#
# import matplotlib.pyplot as plt


# ##############################################################################
# # HELPERS
# ##############################################################################
def accept_condition(path, ignore_explanation=True):
    """
    Return true if path is a file. If ``ignore_explanation`` is true, return
    false also for paths that contain the  'explanation' substring.
    """
    result = os.path.isfile(path)
    if ignore_explanation and "explanation" in path:
        result = False
    return result


# ##############################################################################
# # CLI
# ##############################################################################
@dataclass
class ConfDef:
    """
    :cvar EMOKINE_PATH: Path to the EmokineDataset.
    :cvar OUTPUT_DIR: Where to store the computed techval data.
    :cvar OUTPUT_NAME: Name of the produced techval file, in pickle format.
    :cvar IGNORE_EXPLANATION: If true (default), ``explanation`` files in
      EmokineDataset won't be included in the technical validation.
    :cvar AVATAR_BG_RGB: RGB color value of the background in the Avatar
      stimuli, used to perform background extraction.
    :cvar AVATAR_BG_RGB_MARGIN: Margin around ``AVATAR_BG_RGB`` to consider a
      given color part of the background
    :cvar TRUNCATE: Useful for debugging, if given, only this many files will
      be processed.
    :cvar NUM_PROCESSES: Can be used to speed up processing in multi-core CPUs.
    """
    EMOKINE_PATH: str = MISSING
    OUTPUT_DIR: str = MISSING
    OUTPUT_NAME: str = "techval.pickle"
    IGNORE_EXPLANATION: bool = True
    #
    AVATAR_BG_RGB: Tuple[int] = (208, 240, 241)
    AVATAR_BG_RGB_MARGIN: Tuple[int] = (0, 0, 0)
    #
    TRUNCATE: Optional[int] = None
    NUM_PROCESSES: int = 1


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
    # # GET PATHS
    # ##########################################################################
    sil_path = os.path.join(CONF.EMOKINE_PATH, "Stimuli", "Silhouette")
    avatar_path = os.path.join(CONF.EMOKINE_PATH, "Stimuli", "Avatar")
    pld_path = os.path.join(CONF.EMOKINE_PATH, "Stimuli", "PLD")
    cam_path = os.path.join(CONF.EMOKINE_PATH, "Data", "CamPos")
    kin_path = os.path.join(CONF.EMOKINE_PATH, "Data", "Kinematic")
    #
    sil_paths = [p for p in glob.glob(os.path.join(sil_path, "**", "*"),
                                      recursive=True)
                 if accept_condition(p, CONF.IGNORE_EXPLANATION)]
    avatar_paths = [p for p in glob.glob(os.path.join(avatar_path, "**", "*"),
                                         recursive=True)
                    if accept_condition(p, CONF.IGNORE_EXPLANATION)]
    pld_paths = [p for p in glob.glob(os.path.join(pld_path, "**", "*"),
                                      recursive=True)
                 if accept_condition(p, CONF.IGNORE_EXPLANATION)]
    campos_paths = [p for p in glob.glob(os.path.join(cam_path, "**", "*"),
                                         recursive=True)
                    if accept_condition(p, CONF.IGNORE_EXPLANATION)]
    kin_paths = [p for p in glob.glob(os.path.join(kin_path, "**", "*"),
                                      recursive=True)
                 if accept_condition(p, CONF.IGNORE_EXPLANATION)]
    #
    sil_paths = sorted(sil_paths)[:CONF.TRUNCATE]
    avatar_paths = sorted(avatar_paths)[:CONF.TRUNCATE]
    pld_paths = sorted(pld_paths)[:CONF.TRUNCATE]
    campos_paths = sorted(campos_paths)[:CONF.TRUNCATE]
    kin_paths = sorted(kin_paths)[:CONF.TRUNCATE]

    # ##########################################################################
    # # SILHOUETTE/PLD/AVATAR
    # ##########################################################################
    print("Processing silhouette videos...")
    sil_ratios, sil_hists = techval_bw_vid_multi(
        sil_paths, ignore_below=0,
        num_processes=CONF.NUM_PROCESSES)
    print("Processing PLD videos...")
    pld_ratios, pld_hists = techval_bw_vid_multi(
        pld_paths, ignore_below=0,
        num_processes=CONF.NUM_PROCESSES)
    print("Processing Avatar videos...")
    avatar_ratios, avatar_hists = techval_avatar_vid_multi(
        avatar_paths, CONF.AVATAR_BG_RGB, CONF.AVATAR_BG_RGB_MARGIN,
        ignore_below=100, ignore_above=100_000,
        num_processes=CONF.NUM_PROCESSES)

    ##########################################################################
    # CAMPOS
    ##########################################################################
    campos_hw = load_bw_vid(sil_paths[0], ignore_below=0)[0][0].shape
    print("Processing CamPos files...")
    campos_ratios, campos_hists, campos_xyxy_bounds = techval_campos_multi(
        campos_paths, campos_hw, CONF.NUM_PROCESSES)
    min_x, min_y, max_x, max_y = zip(*campos_xyxy_bounds)

    ##########################################################################
    # SAVE RESULTS
    ##########################################################################
    techval_data = {
        "sil_paths": sil_paths,
        "avatar_paths": avatar_paths,
        "pld_paths": pld_paths,
        "campos_paths": campos_paths,
        "kin_paths": kin_paths,
        #
        "sil_ratios": sil_ratios, "sil_hists": sil_hists,
        "pld_ratios": pld_ratios, "pld_hists": pld_hists,
        "avatar_ratios": avatar_ratios, "avatar_hists": avatar_hists,
        "campos_ratios": campos_ratios, "campos_hists": campos_hists,
        "campos_xyxy_bounds": campos_xyxy_bounds}
    #
    outpath = os.path.join(CONF.OUTPUT_DIR, CONF.OUTPUT_NAME)
    #
    with open(outpath, "wb") as f:
        pickle.dump(techval_data, f)
        print("Saved result to", outpath)
