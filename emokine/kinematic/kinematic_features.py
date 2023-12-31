# -*- coding:utf-8 -*-


"""
This module implements several kinematic features of a single-person movement,
that can be extracted from different data modalities like silhouette videos
and MVNX MoCap data.
"""


import numpy as np
#
from . import median_absdev


# ##############################################################################
# # FROM SILHOUETTES
# ##############################################################################
def silhouette_motion_mask(fhw_segment):
    """
    :param fhw_segment: Boolean array with shape ``(frames, h, w)``.
    :returns: Boolean array with shape ``(h, w)``, generated by adding
      together all frames, and subtracting the last one, to reflect the
      variations that happened across the ``f-1`` frames.
    """
    last_frame = fhw_segment[-1]
    smi = fhw_segment.any(axis=0)  # gather all activations
    smi[last_frame.nonzero()] = 0  # remove last frame
    return smi, last_frame


def quantity_of_motion(bool_fhw_vid, smi_frames=5):
    """
    :param fhw_segment: Boolean array with shape ``(frames, h, w)``, containing
      the silhouettes. Note that ``False`` is interpreted as background.
    :param int smi_frames: How many frames will be used to compute the SMIs.
    :returns: The triple of lists ``(all_qom, smi_areas, sil_areas)``. Each
      list has ``frames-smi_frames+1`` elements, where the first list element
      corresponds to ``frame=smi_frames``, and the next elements follow. The
      ``smi_areas`` contain the number of SMI active pixels, ``sil_areas``
      contain the number of pixels for the area of the silhouette in the last
      SMI frame, and ``all_qom`` contain the rations between SMI areas and
      silhouette areas.

    For more details see also section 3.1 in "Recognising Human Emotions from
    Body Movement and Gesture Dynamics" (Castellano, Villalba, Camurri 2007).
    """
    f, h, w = bool_fhw_vid.shape
    smi_areas, sil_areas, all_qom = [], [], []
    for i in range(0, f - smi_frames + 1):
        smi, last_frame = silhouette_motion_mask(bool_fhw_vid[i:i+smi_frames])
        #
        area_smi = smi.sum()
        area_silhouette = last_frame.sum()
        qom = float(area_smi) / area_silhouette
        #
        smi_areas.append(area_smi)
        sil_areas.append(area_silhouette)
        all_qom.append(qom)

    return all_qom, smi_areas, sil_areas


# ##############################################################################
# # FROM MVNX MOCAP
# ##############################################################################
def mvnx_3d_mean_max_mad_magnitudes(df, keypoints,
                                    beg_frame=None, end_frame=None):
    """
    Given a pandas ``df`` for a 3-dimensional MVNX feature, this function
    gathers the (x,y,z) values for all the given ``keypoints`` at the
    given ``[beg, end)`` range, computes the ``norm(xyz)`` magnitudes,
    and returns the mean, max and MAD magnitude for each keypoint.

    :param df: data frame as the ones provided by ``MvnxToTabular``
    :param keypoints: Collection containing e.g. ``"Pelvis", "L5", "L3"...``
    :returns: the pair ``(mean_vals, max_vals)``, as dictionaries in form
      ``{kp_name: value, ...}``
    """
    mean_vals = {}
    max_vals = {}
    mad_vals = {}
    if beg_frame is None:
        beg_frame = 0
    if end_frame is None:
        end_frame = len(df)
    for kp in keypoints:
        # xyz shape is (frames, 3)
        xyz = df.loc[beg_frame:end_frame-1, kp+"_x":kp+"_z"].values
        magnitudes = np.linalg.norm(xyz, axis=1)
        mean_vals[kp] = magnitudes.mean()
        max_vals[kp] = max(magnitudes)
        mad_vals[kp] = median_absdev(magnitudes)
    #
    return mean_vals, max_vals, mad_vals


def dimensionless_jerk(mvnx_vel_dataframe, keypoints, beg_frame=None,
                       end_frame=None, srate=1):
    """
    :param mvnx_vel_dataframe: Data frame as the ones provided by
      ``MvnxToTabular`` containing the keypoint velocities.
    :param keypoints: Collection containing e.g. ``"Pelvis", "L5", "L3"...``
    :param srate: The jerk is integrated for the time between beg and end. But
      here we have a discrete sum, so knowing the sample rate allows us to
      adjust "dt" in the integral. Of course the result is dimensionless, but
      having high frequencies with a srate of 1 may return extremely high
      values, so keeping everything on SI helps, also to compare among
      different srates.

    :returns: A dict ``{kp: val, ...}`` where val is the dimensionless jerk
      between beg and end for the kp, using the mean velocity as denominator.
      For an explanation see: "Sensitivity of Smoothness Measures
      to Movement Duration, Amplitude and Arrests (Hogan and Sternad)".
    """
    result = {}
    #
    if beg_frame is None:
        beg_frame = 0
    if end_frame is None:
        end_frame = len(mvnx_vel_dataframe)
    #
    for kp in keypoints:
        # note the end+1 to gather 2 extra samples if possible.
        # we don't integrate from accel because it is way more noisy than vel
        # also integrating from pos is bad, discretizes a lot
        vel_xyz = mvnx_vel_dataframe.loc[
            beg_frame:end_frame+1, kp+"_x":kp+"_z"].values
        acc_xyz = (vel_xyz[1:] - vel_xyz[:-1]) * srate
        jerk_xyz = (acc_xyz[1:] - acc_xyz[:-1]) * srate
        #
        vel_norm = np.linalg.norm(vel_xyz, axis=1)
        jerk_norm = np.linalg.norm(jerk_xyz, axis=1)
        #
        #
        # jerk_sq_integral = (jerk_norm**2).sum() / srate
        jerk_sq_integral = jerk_norm.sum() / srate  # WE DON'T NEED NORM**2
        time_cubed = ((end_frame - beg_frame) / srate)**3
        mean_vel_squared = vel_norm.mean() ** 2
        #
        dless_jerk = (time_cubed * jerk_sq_integral) / mean_vel_squared
        # !!!!
        dless_jerk *= mean_vel_squared

        result[kp] = dless_jerk
        # vel2_xyz = mvnx_whole_dataframe["velocity"].loc[
        #     beg_frame:end_frame+1, kp+"_x":kp+"_z"].values
        # acc2_xyz = (vel2_xyz[1:] - vel2_xyz[:-1]) * srate
        # plt.clf(); plt.plot(vel_xyz); plt.show()
        # plt.clf(); plt.plot(jerk_xyz); plt.show()
        # plt.clf(); plt.plot(vel_norm); plt.show()
        # plt.clf(); plt.plot(jerk_norm); plt.show()
    #
    return result


def limb_contraction(mvnx_pos_dataframe, beg_frame=None, end_frame=None):
    """
    Limb contraction from Poyo Solanas 2020: For each frame, this metric
    is the mean euclideam distance between the 4 endpoints and the head.
    :param mvnx_pos_dataframe: Data frame as the ones provided by
      ``MvnxToTabular`` containing the keypoint positions.
    :returns: an array of shape ``(frames, 4)``, where each entry is the 3D
      eucl. distance from ``(rHand, lHand, rToe, lToe)`` to ``head``.
    """
    if beg_frame is None:
        beg_frame = 0
    if end_frame is None:
        end_frame = len(mvnx_pos_dataframe)
    xyz_pos_head = mvnx_pos_dataframe.loc[beg_frame:end_frame-1,
                                          "Head_x":"Head_z"]
    xyz_pos_lhand = mvnx_pos_dataframe.loc[beg_frame:end_frame-1,
                                           "LeftHand_x":"LeftHand_z"]
    xyz_pos_rhand = mvnx_pos_dataframe.loc[beg_frame:end_frame-1,
                                           "RightHand_x":"RightHand_z"]
    xyz_pos_rtoe = mvnx_pos_dataframe.loc[beg_frame:end_frame-1,
                                          "RightToe_x":"RightToe_z"]
    xyz_pos_ltoe = mvnx_pos_dataframe.loc[beg_frame:end_frame-1,
                                          "LeftToe_x":"LeftToe_z"]
    #
    eucl_rhand = np.linalg.norm(
        xyz_pos_head.values - xyz_pos_rhand.values, axis=1)
    eucl_lhand = np.linalg.norm(
        xyz_pos_head.values - xyz_pos_lhand.values, axis=1)
    eucl_rtoe = np.linalg.norm(
        xyz_pos_head.values - xyz_pos_rtoe.values, axis=1)
    eucl_ltoe = np.linalg.norm(
        xyz_pos_head.values - xyz_pos_ltoe.values, axis=1)
    #
    result = np.stack([eucl_rhand, eucl_lhand, eucl_rtoe, eucl_ltoe]).T
    assert result.shape == (end_frame - beg_frame, 4), \
        "This shouldn't happen!"
    return result


def cmass_distances(mvnx_pos_dataframe, mvnx_cmass_dataframe,
                    keypoints, beg_frame=None, end_frame=None):
    """
    :param mvnx_pos_dataframe: Data frame as the ones provided by
      ``MvnxToTabular`` containing the keypoint positions.
    :param mvnx_cmass_dataframe: Data frame as the ones provided by
      ``MvnxToTabular`` containing the full-body center of mass.
    :param keypoints: Collection containing e.g. ``"Pelvis", "L5", "L3"...``
    :returns: A dict in the form ``kp: arr``, with one kp key per
      ``PROTOTYPE_KEYPOINT``, and arr has one entry per frame between
      ``beg_frame`` and ``end_frame`` containing the eucl. distance.
    """
    if beg_frame is None:
        beg_frame = 0
    if end_frame is None:
        end_frame = len(mvnx_pos_dataframe)
    #
    xyz_cmass = mvnx_cmass_dataframe.loc[beg_frame:end_frame-1,
                                         "centerOfMass_x":"centerOfMass_z"]
    result = {}
    for kp in keypoints:
        xyz = mvnx_pos_dataframe.loc[beg_frame:end_frame-1,
                                     kp+"_x":kp+"_z"].values
        dist_per_frame = np.linalg.norm(xyz_cmass - xyz, axis=1)
        result[kp] = dist_per_frame
    return result


def head_angle(mvnx_pos_dataframe, beg_frame=None, end_frame=None):
    """
    :param mvnx_pos_dataframe: Data frame as the ones provided by
      ``MvnxToTabular`` containing the keypoint positions.
    :returns: A tuple ``(a, b)``. A is the absolute angle per frame compared
      to the vertical line (regardless of the direction). B measures the
      head inclination with respect to the body, as follows: MVNX provides a
      small 4-keypoint diamond on the chest area, formed by a 'vertical' line
      from T8 (5th) to Neck (6th), and a 'horizontal' line from LeftShoulder
      (8th) to RightShoulder (12th). The provided head inclination is the angle
      between the neck-to-head vector and the T8-to-Neck one. If we want to
      ignore lateral inclination, we can use the Shoulder-to-Shoulder line to
      find the plane, and then we just need to compute the angle between the
      Neck-to-Head vector and its projection on the plane.

    ..Note::
      We assume that all relative angles are in the same direction, towards
      the chest, and never towards the back. This may be robustified using
      the angle info from MVNX. All angles are given in radians.
    """
    if beg_frame is None:
        beg_frame = 0
    if end_frame is None:
        end_frame = len(mvnx_pos_dataframe)
    # COMPUTE RELATIVE ANGLES
    xyz_pos_t8 = mvnx_pos_dataframe.loc[beg_frame:end_frame-1,
                                        "T8_x":"T8_z"].values
    xyz_pos_neck = mvnx_pos_dataframe.loc[beg_frame:end_frame-1,
                                          "Neck_x":"Neck_z"].values
    xyz_pos_head = mvnx_pos_dataframe.loc[beg_frame:end_frame-1,
                                          "Head_x":"Head_z"].values
    #
    t8_to_neck = xyz_pos_neck - xyz_pos_t8
    t8_to_neck_unit = t8_to_neck / np.linalg.norm(t8_to_neck, axis=1)[:, None]
    #
    neck_to_head = xyz_pos_head - xyz_pos_neck
    neck_to_head_unit = neck_to_head / np.linalg.norm(
        neck_to_head, axis=1)[:, None]
    #
    relative_cosines = (t8_to_neck_unit * neck_to_head_unit).sum(axis=1)
    assert (relative_cosines > 0).all(), "Head angle >90deg. Person broken??"
    relative_radians = np.arccos(np.clip(relative_cosines, -1.0, 1.0))
    # NOW COMPUTE ABSOLUTE ANGLES: the z component of the unit head is the cos
    # abs angles are always w.r.t [0, 0, 1]
    vertical_cosines = neck_to_head_unit[:, 2]
    vertical_radians = np.arccos(np.clip(vertical_cosines, -1.0, 1.0))
    # to convert rads to degrees, multiply by 180/np.pi
    return vertical_radians, relative_radians
