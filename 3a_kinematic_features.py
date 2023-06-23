#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
Given a path to a silhouette video and its corresponding MVNX sequence and
JSON CamPos data (as extracted by ``1b_mvnx_blender.py``, see README), this
script extracts statistics for several kinematic features, informative of the
type and quantity of movement.
The kinematics are exported as a CSV file, with one row per keypoint and colum
per feature.
"""


from pathlib import Path
# for OmegaConf
from dataclasses import dataclass
from typing import Optional, Tuple
#
from omegaconf import OmegaConf, MISSING
import numpy as np
import pandas as pd
from shapely.ops import unary_union  # union of polygons
#
from emokine.mvnx import Mvnx, MvnxToTabular
from emokine.utils import load_bw_vid, JsonBlenderPositions
from emokine.kinematic import median_absdev
from emokine.kinematic import get_2d_convex_hull, get_3d_convex_hull
from emokine.kinematic.kinematic_features import quantity_of_motion
from emokine.kinematic.kinematic_features import dimensionless_jerk, \
    mvnx_3d_mean_max_mad_magnitudes, limb_contraction, cmass_distances, \
    head_angle


# ##############################################################################
# # CLI
# ##############################################################################
@dataclass
class ConfDef:
    """
    :cvar SILHOUETTE_PATH: Path to b&w video expected to contain a moving
      silhouette. Expected to be consistent with the JSON and MVNX files.
    :cvar JSON_PATH: Path to a JSON file compatible with the
      ``emokine.utils.JsonBlenderPositions`` class, like the ones produces
      by the ``mvnx_blender`` script. The file contains a sequence of 3D
      positions represented from the perspective of a given camera. Expected
      to be consistent with the SILHOUETTE and MVNX files.
    :cvar MVNX_PATH: Path to a MVNX sequence, expected to be consistent with
      the SILHOUETTE and JSON files.
    :cvar CSV_OUTPATH: Output path to save the computed kinematic features
      as CSV.
    :cvar KEYPOINTS: For the keypoint-specific kinematic features, which
      MVNX keypoints to compute (e.g. left shoulder, left toe...). Default
      is the full body configuration by MVNX.
    :cvar QOM_SMI_SECONDS: When computing quantity of motions, a range of SMI
      frames from the past is used. This tells in seconds how many frames.
    :SIL_IGNORE_FRAMES_BELOW: When loading the silhouette video, any frames
      with less than this many nonzero pixels will be ignored.
    """
    SILHOUETTE_PATH: str = MISSING
    JSON_PATH: str = MISSING
    MVNX_PATH: str = MISSING
    CSV_OUTPATH: str = MISSING
    KEYPOINTS: Tuple[str] = (
        "Pelvis", "L5", "L3", "T12", "T8", "Neck", "Head",
        "RightShoulder", "RightUpperArm", "RightForeArm", "RightHand",
        "LeftShoulder", "LeftUpperArm", "LeftForeArm", "LeftHand",
        "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToe",
        "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToe")
    #
    MVNX_SCHEMA: Optional[str] = None
    #
    QOM_SMI_SECONDS: float = 0.2  # e.g. 0.2 is 5 frames at 25fps
    # Hack to avoid current artifacts in silhouette vid, shouldn't be needed in
    # the future. Any frames with less than this many nonzeros are ignored
    SIL_IGNORE_FRAMES_BELOW: float = 100


# ##############################################################################
# # MAIN FUNCTION
# ##############################################################################
def main(silhouette_path, json_path, mvnx_path, keypoints,
         qom_smi_seconds=0.2,
         mvnx_schema=None, ignore_frames_below=100,
         verbose=False):
    """
    Convenience wrapper for the main routine, see docstring of the script's
    ConfDef for info on the input parameters.

    :returns: A pandas dataframe with one joint per row and one kinematic
      feature per column.
    """
    # load silhouette video, JSON blender positions and MVNX
    sil, sil_fps = load_bw_vid(
        silhouette_path, ignore_below=ignore_frames_below)
    _, sil_h, sil_w = sil.shape
    sil_dur_s = float(len(sil)) / sil_fps
    #
    jbp = JsonBlenderPositions(json_path, which_pos="bone_head")
    jbp_dur_s = len(jbp.data) / jbp.fps
    #
    m = Mvnx(mvnx_path, mvnx_schema)
    mvnx_fps = int(m.mvnx.subject.attrib["frameRate"])
    mvnx_dataframes = MvnxToTabular(m)()
    mvnx_len = len(next(iter(mvnx_dataframes.values())))
    mvnx_dur_s = mvnx_len / mvnx_fps
    #
    print("\nSilhouette duration in seconds:", sil_dur_s)
    print("JSON duration in seconds:", jbp_dur_s)
    print("MVNX duration in seconds:", mvnx_dur_s)

    # # compute silhouette features ###########################################
    # quantity of motion
    qom, _, _ = quantity_of_motion(
        sil, smi_frames=round(qom_smi_seconds * sil_fps))
    mean_qom = np.mean(qom)
    mad_qom = median_absdev(np.array(qom))
    integral_qom = np.sum(qom)

    # # compute JSON features #################################################
    # 2D convex hull:
    # per-frame, global hull, and union of per-frame hulls.
    # Union can't be greater than global, and it will be similar unless
    # many jumps, crouching or other vertical movements happen.
    ch2d, multipoints2d = zip(*[get_2d_convex_hull(jbp.data.iloc[i].array)
                                for i in range(len(jbp.data))])
    mean_ch2d = np.mean(ch2d)
    mad_ch2d = median_absdev(np.array(ch2d))
    global_ch2d = get_2d_convex_hull(jbp.data.to_numpy().flatten())[0]
    union_ch2d = unary_union([x.convex_hull for x in multipoints2d]).area

    # # compute MVNX features #################################################
    # 3D convex hull: analogous to 2D
    ch3d, multipoints3d = zip(
        *[get_3d_convex_hull(mvnx_dataframes["position"].iloc[i, 2:].array)
          for i in range(len(mvnx_dataframes["position"]))])
    mean_ch3d = np.mean(ch3d)
    mad_ch3d = median_absdev(np.array(ch3d))
    global_ch3d = get_3d_convex_hull(
        mvnx_dataframes["position"].iloc[:, 2:].to_numpy().flatten())[0]
    union_ch3d = unary_union([x.convex_hull for x in multipoints3d]).area

    # # compute MVNX features
    # built-in features
    mean_pos, max_pos, mad_pos = mvnx_3d_mean_max_mad_magnitudes(
        mvnx_dataframes["position"], keypoints)
    mean_vels, max_vels, mad_vels = mvnx_3d_mean_max_mad_magnitudes(
        mvnx_dataframes["velocity"], keypoints)
    mean_accels, max_accels, mad_accels = mvnx_3d_mean_max_mad_magnitudes(
        mvnx_dataframes["acceleration"], keypoints)
    (mean_ang_vels, max_ang_vels,
     mad_ang_vels) = mvnx_3d_mean_max_mad_magnitudes(
        mvnx_dataframes["angularVelocity"], keypoints)
    (mean_ang_accels, max_ang_accels,
     mad_ang_accels) = mvnx_3d_mean_max_mad_magnitudes(
         mvnx_dataframes["angularAcceleration"], keypoints)
    # dimensionless jerk
    dimensionless_jerks_vmean = dimensionless_jerk(
        mvnx_dataframes["velocity"], keypoints, srate=mvnx_fps)
    # limb contracton from Poyo Solanas (dist. between limbs and head)
    lcont = limb_contraction(mvnx_dataframes["position"])
    lcont_t_mean = lcont.mean(axis=1)
    lcont_global_mean = lcont.mean()
    lcont_t_mad = median_absdev(lcont_t_mean)
    # our limb contraction: dist between each keypoint and center of mass
    cmass_dists = cmass_distances(
        mvnx_dataframes["position"], mvnx_dataframes["centerOfMass"],
        keypoints)
    cmass_dists_mean_per_kp = {k: v.mean() for k, v in cmass_dists.items()}
    cmass_dists_mad_per_kp = {k: median_absdev(v) for k, v
                              in cmass_dists.items()}
    # neck-head angle: vert is global, rel is against the t8-neck line
    head_t_angle_vert, head_t_angle_rel = head_angle(
        mvnx_dataframes["position"])
    mean_head_angle_vert = head_t_angle_vert.mean()
    mad_head_angle_vert = median_absdev(head_t_angle_vert)
    mean_head_angle_rel = head_t_angle_rel.mean()
    mad_head_angle_rel = median_absdev(head_t_angle_rel)

    # animation_3d_mvnx(mvnx_dataframes["position"])
    # arr=mvnx_dataframes["position"]["LeftHand_z"]; plt.clf(); plt.plot(arr); plt.show()
    # plt.clf(); plt.plot(qom); plt.show()
    # plt.clf(); plt.plot(ch2d); plt.show()
    # plt.clf(); plt.axis("equal"); plt.scatter(xvals.to_numpy(), yvals.to_numpy()); plt.show()
    # plt.clf(); plt.plot(ch3d); plt.show()
    # arr = lcont_t_mean; plt.clf(); plt.plot(arr); plt.show()

    # Prepare CSV output
    columns = ["MVNX frame rate", "MVNX num frames",
               "silhouette frame rate", "silhouette num frames",
               # row_basic ends here
               "keypoint",
               "avg. vel.", "max. vel.", "MAD vel.",
               "avg. accel.", "max. accel.", "MAD accel.",
               "avg. angular vel.", "max. angular vel.", "MAD angular vel.",
               "avg. angular accel.", "max. angular accel.",
               "MAD angular accel.",
               #
               "dimensionless jerk",
               "avg. limb contraction",
               "MAD limb contraction",
               "avg. CoM dist.", "MAD CoM dist.",
               "avg. head angle (w.r.t. vertical)",
               "MAD head angle (w.r.t. vertical)",
               "avg. head angle (w.r.t. back)",
               "MAD head angle (w.r.t. back)",
               "avg. QoM",
               "MAD QoM",
               "integral QoM",
               "avg. convex hull 2D", "MAD convex hull 2D",
               "global convex hull 2D", "union convex hull 2D",
               "avg. convex hull 3D", "MAD convex hull 3D",
               "global convex hull 3D", "union convex hull 3D"]
    print("\n", columns)

    row_basic = [mvnx_fps, mvnx_len, sil_fps, len(sil)]
    df = pd.DataFrame(columns=columns)
    for i, kp in enumerate(keypoints):
        r = row_basic + [
            kp,
            mean_vels[kp], max_vels[kp], mad_vels[kp],
            mean_accels[kp], max_accels[kp], mad_accels[kp],
            mean_ang_vels[kp], max_ang_vels[kp], mad_ang_vels[kp],
            mean_ang_accels[kp], max_ang_accels[kp], mad_ang_accels[kp],
            dimensionless_jerks_vmean[kp],
            lcont_global_mean, lcont_t_mad,
            cmass_dists_mean_per_kp[kp], cmass_dists_mad_per_kp[kp],
            mean_head_angle_vert, mad_head_angle_vert,
            mean_head_angle_rel, mad_head_angle_rel,
            mean_qom, mad_qom, integral_qom,
            mean_ch2d, mad_ch2d,  global_ch2d, union_ch2d,
            mean_ch3d, mad_ch3d,  global_ch3d, union_ch3d]
        df.loc[i] = r
        if verbose:
            print(", ".join(map(str, r)))
    #
    return df


# ##############################################################################
# # MAIN ROUTINE
# ##############################################################################
if __name__ == "__main__":
    CONF = OmegaConf.structured(ConfDef())
    cli_conf = OmegaConf.from_cli()
    CONF = OmegaConf.merge(CONF, cli_conf)
    print("\n\nCONFIGURATION:")
    print(OmegaConf.to_yaml(CONF), end="\n\n")

    df = main(CONF.SILHOUETTE_PATH, CONF.JSON_PATH, CONF.MVNX_PATH,
              CONF.KEYPOINTS, CONF.QOM_SMI_SECONDS,
              CONF.MVNX_SCHEMA, CONF.SIL_IGNORE_FRAMES_BELOW)

    Path(CONF.CSV_OUTPATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CONF.CSV_OUTPATH, index=False)
    print("\nSaved CSV to", CONF.CSV_OUTPATH)
