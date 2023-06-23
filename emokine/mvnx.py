#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This module contains functionality concerning the adaption of the
XSENS MVNX (the XML version of their proprietary MVN format) into
our Python setup.
The adaption tries to be as MVN-version-agnostc as possible. Still,
it is possible to validate the file against a given schema.

The official explanation can be found in section 14.4 of the
*XSENS MVN User Manual*:

  https://usermanual.wiki/Document/MVNUserManual.1147412416.pdf
"""


import pandas as pd
import numpy as np
from lxml import etree, objectify  # https://lxml.de/validation.html
#
from .utils import str_to_vec


# #############################################################################
# ## GLOBALS
# #############################################################################
KNOWN_STR_FIELDS = {"tc", "type"}
KNOWN_INT_FIELDS = {"segmentCount", "sensorCount", "jointCount",
                    "time", "index", "ms"}  # "audio_sample"
KNOWN_FLOAT_VEC_FIELDS = {"orientation", "position", "velocity",
                          "acceleration", "angularVelocity",
                          "angularAcceleration", "sensorFreeAcceleration",
                          "sensorMagneticField", "sensorOrientation",
                          "jointAngle", "jointAngleXZY", "jointAngleErgo",
                          "centerOfMass"}


# #############################################################################
# ## HELPERS
# #############################################################################
def process_dict(d, str_fields, int_fields, fvec_fields):
    """
    :returns: a copy of the given dict ``d`` where the values (expected str)
      whose keys are in the specified fields are converted to the specified
      type. E.g. If ``int_fields`` contains the ``index`` string and the given
      dict contains the ``index`` key, the corresponding value will be
      converted via ``int()``.
    """
    result = {}
    for k, v in d.items():
        if k in str_fields:
            result[k] = str(v)
        elif k in int_fields:
            result[k] = int(v)
        elif k in fvec_fields:
            result[k] = str_to_vec(v)
        else:
            result[k] = v
    return result


# #############################################################################
# ## MVNX CLASS
# #############################################################################
class Mvnx:
    """
    This class imports and adapts an XML file (expected to be in MVNX format)
    to a Python-friendly representation. See this module's docstring for usage
    examples and more information.
    """
    def __init__(self, mvnx_path, mvnx_schema_path=None,
                 str_fields=KNOWN_STR_FIELDS, int_fields=KNOWN_INT_FIELDS,
                 float_vec_fields=KNOWN_FLOAT_VEC_FIELDS):
        """
        :param str mvnx_path: a valid path pointing to the XML file to load
        :param str mvnx_schema_path: (optional): if given, the given MVNX will
          be validated against this XML schema definition.
        :param collection fields: List of strings with field names that are
          converted to the specified type when calling ``extract_frame_info``.
        """
        self.mvnx_path = mvnx_path
        #
        mvnx = etree.parse(mvnx_path, etree.ETCompatXMLParser())
        # if a schema is given, load it and validate mvn
        if mvnx_schema_path is not None:
            self.schema = etree.XMLSchema(file=mvnx_schema_path)
            self.schema.assertValid(mvnx)
        #
        self.mvnx = objectify.fromstring(etree.tostring(mvnx))
        #
        self.str_fields = str_fields
        self.int_fields = int_fields
        self.fvec_fields = float_vec_fields

    def export(self, filepath, pretty_print=True, extra_comment=""):
        """
        Saves the current ``mvnx`` attribute to the given file path as XML and
        adds the ``self.mvnx.attrib["pythonComment"]`` attribute with
        a timestamp.
        """
        #
        with open(filepath, "w") as f:
            if extra_comment:
                self.mvnx.attrib["pythonComment"] = str(extra_comment)
            s = etree.tostring(
                self.mvnx, pretty_print=pretty_print).decode("utf-8")
            f.write(s)
            print("[Mvnx] exported to", filepath)
            if extra_comment:
                print("   Comment:", extra_comment)

    # EXTRACTORS: LIKE "GETTERS" BUT RETURN A MODIFIED COPY OF THE CONTENTS
    def extract_frame_info(self):
        """
        :returns: The tuple ``(frames_metadata, config_frames, normal_frames)``
        """
        f_meta, config_f, normal_f = self.extract_frames(self.mvnx,
                                                         self.str_fields,
                                                         self.int_fields,
                                                         self.fvec_fields)
        frames_metadata = f_meta
        config_frames = config_f
        normal_frames = normal_f
        #
        assert (frames_metadata["segmentCount"] ==
                len(self.extract_segments())), "Inconsistent segmentCount?"
        return frames_metadata, config_frames, normal_frames

    @staticmethod
    def extract_frames(mvnx, str_fields, int_fields, fvec_fields):
        """
        The bulk of the MVNX file is the ``mvnx->subject->frames`` section.
        This function parses it and returns its information in a
        python-friendly format, mainly via the ``process_dict`` function.

        :param mvnx: An XML tree, expected to be in MVNX format
        :param collection fields: Collection of strings with field names that
          are converted to the specified type (fvec is a vector of floats).

        :returns: a tuple ``(frames_metadata, config_frames, normal_frames)``
          where the metadata is a dict in the form ``{'segmentCount': 23,
          'sensorCount': 17, 'jointCount': 22}``, the config frames are the
          first 3 frame entries (expected to contain special config info)
          and the normal_frames are all frames starting from the 4th.
          Fields found in the given int and vec field lists will be converted
          and the rest will remain as XML nodes.
        """
        frames_metadata = process_dict(mvnx.subject.frames.attrib,
                                       str_fields, int_fields, fvec_fields)
        # first 3 frames are config. types: "identity", "tpose", "tpose-isb"
        all_frames = mvnx.subject.frames.getchildren()
        # rest of frames contain proper data. type: "normal"
        config_frames = [process_dict({**f.__dict__, **f.attrib},
                                      str_fields, int_fields, fvec_fields)
                         for f in all_frames[:3]]
        normal_frames = [process_dict({**f.__dict__, **f.attrib},
                                      str_fields, int_fields, fvec_fields)
                         for f in all_frames[3:]]
        return frames_metadata, config_frames, normal_frames

    def extract_segments(self):
        """
        :returns: A list of the segment names in ``self.mvnx.subject.segments``
          ordered by id (starting at 1 and incrementing +1).
        """
        segments = [ch.attrib["label"] if str(i) == ch.attrib["id"] else None
                    for i, ch in enumerate(
                            self.mvnx.subject.segments.iterchildren(), 1)]
        assert all([s is not None for s in segments]),\
            "Segments aren't ordered by id?"
        return segments

    def extract_joints(self):
        """
        :returns: A tuple (X, Y). The element X is a list of the joint names
          ordered as they appear in the MVNX file.
          The element Y is a list in the original MVNX ordering, in the form
          [((seg_ori, point_ori), (seg_dest, point_dest)), ...], where each
          element contains 4 strings summarizing the origin->destiny of a
          connection.
        """
        names, connectors = [], []
        for j in self.mvnx.subject.joints.iterchildren():
            names.append(j.attrib["label"])
            #
            seg_ori, point_ori = j.connector1.text.split("/")
            seg_dest, point_dest = j.connector2.text.split("/")
            connectors.append(((seg_ori, point_ori), (seg_dest, point_dest)))
        return names, connectors


# #############################################################################
# ## MVNX TO CSV
# #############################################################################
class MvnxToTabular:
    """
    This class is specialized to convert full-body MVNX data, as provided by
    the XSENS system and parsed by the ``Mvnx`` companion class, into tabular
    form.

    To use it, instantiate it with the desired MVNX object, and call it with
    the desired frame fields to be extracted. Usage example: see the
    ``1a_mvnx_to_csv.py`` script.
    """

    ALLOWED_FIELDS = {"orientation", "position", "velocity", "acceleration",
                      "angularVelocity", "angularAcceleration", "footContacts",
                      "jointAngle", "centerOfMass", "ms"}
    # vectors of NUM_SEGMENTS*n where n is 4 or 3 respectively
    SEGMENT_4D_FIELDS = {"orientation"}
    SEGMENT_3D_FIELDS = {"position", "velocity",
                         "acceleration", "angularVelocity",
                         "angularAcceleration"}
    # check manual, 22.7.3, "JointAngle"s are in ZXY if not specified
    JOINT_ANGLE_3D_LIST = ["L5S1", "L4L3", "L1T12", "C1Head",
                           "C7LeftShoulder", "LeftShoulder", "LeftShoulderXZY",
                           "LeftElbow", "LeftWrist", "Lefthip", "LeftKnee",
                           "LeftAnkle", "LeftBallFoot",
                           "C7RightShoulder", "RightShoulder",
                           "RightShoulderXZY",
                           "RightElbow", "RightWrist", "Righthip", "RightKnee",
                           "RightAnkle", "RightBallFoot"]
    # a 4D boolean vector
    FOOT_CONTACTS = ["left_heel_on_ground", "left_toe_on_ground",
                     "right_heel_on_ground", "right_toe_on_ground"]

    def __init__(self, mvnx):
        """
        :param mvnx: An instance of the ``Mvnx`` class to be converted.
        """
        assert mvnx.mvnx.subject.attrib["configuration"] == "FullBody", \
            "This processor works only in FullBody MVNX configurations"
        # extract skeleton and frame info
        joints, seg_detail, seg_names_sorted = self._parse_skeleton(mvnx)
        frames_metadata, config_f, normal_f = mvnx.extract_frame_info()
        #
        self.mvnx = mvnx
        self.seg_names_sorted = seg_names_sorted
        self.seg_detail = seg_detail
        self.joints = joints
        self.n_seg = len(self.seg_names_sorted)
        self.n_j = len(self.joints)
        #
        self.frames_metadata = frames_metadata
        self.config_frames = config_f
        self.normal_frames = normal_f

    @staticmethod
    def _parse_skeleton(mvnx):
        """
        The MVNX files provide a description of the 'rig', or skeleton being
        captured. This method extracts this definition and returns it in
        a convenient form for other (non-protected) methods to use.
        """
        # Retrieve every segment keypoint with XYZ positions
        segproc = lambda seg: np.fromstring(
            seg.pos_b.text, dtype=np.float32, sep=" ")
        seg_detail = {(s.attrib["label"], ch.attrib["label"]): segproc(ch)
                      for s in mvnx.mvnx.subject.segments.iterchildren()
                      for ch in s.points.iterchildren()}
        # Retrieve every joint as a relation of segment details
        joints = [(j.attrib["label"],
                   j.connector1,
                   seg_detail[tuple(j.connector1.text.split("/"))],
                   j.connector2,
                   seg_detail[tuple(j.connector2.text.split("/"))])
                  for j in mvnx.mvnx.subject.joints.iterchildren()]
        # Retrieve segment names sorted by ID
        seg_names_sorted = [s.attrib["label"] for s in
                            sorted(mvnx.mvnx.subject.segments.iterchildren(),
                                   key=lambda elt: int(elt.attrib["id"]))]
        return joints, seg_detail, seg_names_sorted

    def __call__(self, frame_fields=None):
        """
        :param frame_fields: Collection of MVNX fields to be extracted as
          columns. For full list, see ``self.ALLOWED_FIELDS`` (default if
          none given).
        """
        # sanity check
        if frame_fields is None:
            frame_fields = self.ALLOWED_FIELDS
        assert all([f in self.ALLOWED_FIELDS for f in frame_fields]), \
            f"Error! allowed fields: {self.ALLOWED_FIELDS}"
        #
        dataframes = {}
        # process 3d segment data
        for fld in self.SEGMENT_3D_FIELDS:
            columns_3d = ["frame_idx", "ms"] + ["_".join([c, dim])
                                                for c in self.seg_names_sorted
                                                for dim in ["x", "y", "z"]]
            if fld in frame_fields:
                print("processing", fld)
                df = pd.DataFrame(([frm["index"], frm["ms"]] + frm[fld]
                                   for frm in self.normal_frames),
                                  columns=columns_3d)
                dataframes[fld] = df
        # process 4d segment data
        for fld in self.SEGMENT_4D_FIELDS:
            columns_4d = ["frame_idx", "ms"] + [
                "_".join([c, dim]) for c in self.seg_names_sorted
                for dim in ["q0", "q1", "q2", "q3"]]
            if fld in frame_fields:
                print("processing", fld)
                df = pd.DataFrame(([frm["index"], frm["ms"]] + frm[fld]
                                   for frm in self.normal_frames),
                                  columns=columns_4d)
                dataframes[fld] = df
        # process foot contacts
        fld = "footContacts"
        if fld in frame_fields:
            print("processing", fld)
            columns_foot = ["frame_idx", "ms"] + self.FOOT_CONTACTS
            df = pd.DataFrame(([frm["index"], frm["ms"]] +
                               [bool(x) for x in frm[fld].text.split(" ")]
                               for frm in self.normal_frames),
                              columns=columns_foot)
            dataframes[fld] = df
        # process center of mass
        fld = "centerOfMass"
        if fld in frame_fields:
            print("processing", fld)
            columns_com = ["frame_idx", "ms", "centerOfMass_x",
                           "centerOfMass_y", "centerOfMass_z"]
            df = pd.DataFrame(([frm["index"], frm["ms"]] +
                               frm[fld]
                               for frm in self.normal_frames),
                              columns=columns_com)
            dataframes[fld] = df
        #
        return dataframes
