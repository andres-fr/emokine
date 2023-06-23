#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
MVNX MoCap script for Blender. It has the following functionality:

1. Load MVNX sequence into Blender as an animated armature
2. Attach spheres to the armature's bones
3. Given a camera, calculate pixel-positions (and depth) of the spheres
4. Given a camera, render sequence in the form of "dots on a flat background"

Many of the hardcoded parameters in the GLOBALS section are app-specific and
should be revised for different applications, but this script provides the
functionality and structure to do so with ease.
"""


from math import radians, cos, sin
import json
import argparse
import sys
import os
#
import lxml
# blender imports
from mathutils import Vector, Euler  # mathutils is a blender package
import bpy
from bpy_extras.object_utils import world_to_camera_view
#
from io_anim_mvnx.mvnx_import import load_mvnx_into_blender

# Blender aliases
C = bpy.context
D = bpy.data
O = bpy.ops


# ##############################################################################
# BLENDER-SPECIFIC HELPERS
# ##############################################################################
class ArgumentParserForBlender(argparse.ArgumentParser):
    """
    This class is identical to its parent, except for the parse_args method
    (see docstring). It resolves the ambiguity generated when calling
    Blender from the CLI with a python script, and both Blender and the script
    have arguments. E.g., the following call will make Blender crash because
    it will try to process the script's -a and -b flags:
    >>> blender --python my_script.py -a 1 -b 2

    To bypass this issue this class uses the fact that Blender will ignore all
    arguments given after a double-dash ('--'). The approach is that all
    arguments before '--' go to Blender, arguments after go to the script.
    The following calls work fine:
    >>> blender --python my_script.py -- -a 1 -b 2
    >>> blender --python my_script.py --
    """

    def _get_argv_after_doubledash(self):
        """
        Given the sys.argv as a list of strings, this method returns the
        sublist right after the '--' element (if present, otherwise returns
        an empty list).
        """
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx+1:]  # the list after '--'
        except ValueError:  # '--' not in the list:
            return []

    # overrides superclass
    def parse_args(self):
        """
        This method is expected to behave identically as in the superclass,
        except that the sys.argv list will be pre-processed using
        _get_argv_after_doubledash before. See the docstring of the class for
        usage examples and details.
        """
        return super().parse_args(args=self._get_argv_after_doubledash())


def rot_euler_degrees(rot_x, rot_y, rot_z, order="XYZ"):
    """
    Returns an Euler rotation object with the given rotations (in degrees)
    and rotation order.
    """
    return Euler((radians(rot_x), radians(rot_y), radians(rot_z)), order)


def update_scene():
    """
    Sometimes changes don't show up due to lazy evaluation. This function
    triggers scene update and recalculation of all changes.
    """
    C.scene.update()


def save_blenderfile(filepath=D.filepath):
    """
    Saves blender file
    """
    O.wm.save_as_mainfile(filepath=filepath)


def open_blenderfile(filepath=D.filepath):
    """
    Saves blender file
    """
    O.wm.open_mainfile(filepath=filepath)


def set_render_resolution_percentage(p=100):
    """
    """
    D.scenes[0].render.resolution_percentage = p


def get_obj(obj_name):
    """
    Actions like undo or entering edit mode invalidate the object references.
    This function returns a reference that is always valid, assuming that the
    given obj_name is a key of bpy.data.objects.
    """
    return D.objects[obj_name]


def select_by_name(*names):
    """
    Given a variable number of names as strings, tries to select all existing
    objects in D.objects by their name.
    """
    for name in names:
        try:
            D.objects[name].select_set(True)
        except Exception as e:
            print(e)


def deselect_by_name(*names):
    """
    Given a variable number of names as strings, tries to select all existing
    objects in D.objects by their name.
    """
    for name in names:
        try:
            D.objects[name].select_set(False)
        except Exception as e:
            print(e)


def select_all(action="SELECT"):
    """
    Action can be SELECT, DESELECT, INVERT, TOGGLE
    """
    bpy.ops.object.select_all(action=action)


def delete_selected():
    bpy.ops.object.delete()


def set_mode(mode="OBJECT"):
    """
    """
    bpy.ops.object.mode_set(mode=mode)


def purge_unused_data(categories=[D.meshes, D.materials, D.textures, D.images,
                                  D.curves, D.lights, D.cameras, D.screens]):
    """
    Blender objects point to data. E.g., a lamp points to a given data lamp
    object. Removing the objects doesn't remove the data, which may lead to
    data blocks that aren't being used by anyone. Given an ORDERED collection
    of categories, this function removes all unused datablocks.
    See https://blender.stackexchange.com/a/102046
    """
    for cat in categories:
        for block in cat:
            if block.users == 0:
                cat.remove(block)


def set_shading_mode(mode="SOLID", screens=[]):
    """
    Performs an action analogous to clicking on the display/shade button of
    the 3D view. Mode is one of "RENDERED", "MATERIAL", "SOLID", "WIREFRAME".
    The change is applied to the given collection of bpy.data.screens.
    If none is given, the function is applied to bpy.context.screen (the
    active screen) only. E.g. set all screens to rendered mode:
      set_shading_mode("RENDERED", D.screens)
    """
    screens = screens if screens else [C.screen]
    for s in screens:
        for spc in s.areas:
            if spc.type == "VIEW_3D":
                spc.spaces[0].shading.type = mode
                break  # we expect at most 1 VIEW_3D space


def maximize_layout_3d_area():
    """
    TODO: this function assumes Layout is the bpy.context.workspace.
    It does the following:
    1. If there is an area with the given name:
       1.1. Minimizes any other maximized window
       1.2. Maximizes the desired area
    """
    screen_name = "Layout"
    area_name = "VIEW_3D"
    screen = D.screens[screen_name]
    for a in screen.areas:
        if a.type == area_name:
            # If screen is already in some fullscreen mode, revert it
            if screen.show_fullscreen:
                bpy.ops.screen.back_to_previous()
            # Set area to fullscreen (dict admits "window","screen","area")
            bpy.ops.screen.screen_full_area({"screen": screen, "area": a})
            break


if __name__ == "__main__":
    # ##########################################################################
    # GLOBALS
    # ##########################################################################
    parser = ArgumentParserForBlender()
    parser.add_argument("-x", "--mvnx", type=str, required=True,
                        help="MVNX motion capture file to be loaded")
    parser.add_argument(
        "-S", "--mvnx_schema", type=str, default=None,
        help="XML validation schema for the given MVNX (optional)")
    parser.add_argument("-r", "--render_headless", action="store_true",
                        help="If given, this script will actually render out")
    parser.add_argument("-o", "--output_dir", default=os.path.expanduser("~"),
                        type=str, help="Output dir for the renderings")
    parser.add_argument("-p", "--resolution_percentage", type=int, default=100,
                        help="Smaller resolution -> faster (but worse) render")
    parser.add_argument("-v", "--as_video", action="store_true",
                        help="if given, MP4 is exported (noisy background)")
    args = parser.parse_args()

    RENDER_HEADLESS = args.render_headless
    OUT_DIR = os.path.join(args.output_dir, "")  # ensure that it is a dir path
    try:
        os.makedirs(OUT_DIR)
    except FileExistsError:
        pass
    RESOLUTION_PERCENTAGE = args.resolution_percentage
    AS_VIDEO = args.as_video
    MVNX_PATH = args.mvnx
    SCHEMA_PATH = args.mvnx_schema
    MVNX_POSITION = (-0.1, -0.07, 0)
    MVNX_ROTATION = (0, 0, radians(-6.6))  # euler angle

    BACKGROUND_COLOR = (0, 0, 0, 0)
    DOT_COLOR = (100, 100, 100, 0)
    DOT_DIAMETER = 0.04
    ALL_KEYPOINTS = {"Pelvis", "L5", "L3", "T12", "T8", "Neck", "Head",
                     "RightShoulder", "RightUpperArm", "RightForeArm",
                     "RightHand",
                     "LeftShoulder", "LeftUpperArm", "LeftForeArm",
                     "LeftHand",
                     "RightUpperLeg", "RightLowerLeg", "RightFoot",
                     "RightToe",
                     "LeftUpperLeg", "LeftLowerLeg", "LeftFoot",
                     "LeftToe"}

    # Select which PLDs to display
    KEYPOINT_SELECTION = {
        # "Head",
        # "Pelvis",
        # "L5",
        "T12",
        # "Neck",
        "RightShoulder",
        "RightUpperArm",
        # "RightForeArm",
        # "RightHand",
        "LeftShoulder",
        "LeftUpperArm",
        # "LeftForeArm",
        # "LeftHand",
        "RightUpperLeg",
        # "RightLowerLeg",
        "RightFoot",
        # "RightToe",
        "LeftUpperLeg",
        # "LeftLowerLeg",
        # "LeftToe",
        "LeftFoot"
    }
    USED_BONES = KEYPOINT_SELECTION
    INIT_SHADING_MODE = "RENDERED"
    INIT_3D_MAXIMIZED = False
    # renderer
    EEVEE_RENDER_SAMPLES = 8
    EEVEE_VIEWPORT_SAMPLES = 0  # 1
    EEVEE_VIEWPORT_DENOISING = True
    RESOLUTION_WH = (1920, 1080)
    # sequencer
    FRAME_START = 0  # 1000 # 2  # 1 is T-pose if imported with MakeWalk
    FRAME_END = None  # 1500  # If not None sequence will be at most this

    # In Blender, x points away from the cam, y to the left and z up
    # (right-hand rule). Locations are in meters, rotation in degrees.
    # Positive rotation on an axis means counter-clockwise when
    # the axis points to the cam. 0,0,0 rotation points straight
    # to the bottom.

    # SUN_NAME = "SunLight"
    # SUN_LOC = Vector((0.0, 0.0, 10.0))
    # SUN_ROT = rot_euler_degrees(0, 0, 0)
    # SUN_STRENGTH = 1.0  # in units relative to a reference sun

    FRONTAL_CAM_NAME = "FrontalCam"
    FRONTAL_CAM_DIST = 8.16
    FRONTAL_CAM_ANGLE = 0
    # cam is on the front-right
    # FRONTAL_CAM_LOC = (FRONTAL_CAM_DIST * cos(radians(FRONTAL_CAM_ANGLE)),
    #                    FRONTAL_CAM_DIST * sin(radians(FRONTAL_CAM_ANGLE)),
    #                    1.6)
    FRONTAL_CAM_LOC = (11.96, 0.04, 1)
    # Vector((8.16, 0, 1.6))
    # human-like view at the origin
    FRONTAL_CAM_ROT = rot_euler_degrees(90.0, 0.0, 90.0)
    FRONTAL_CAM_LIGHT_NAME = "FrontalCamLight"
    FRONTAL_CAM_LIGHT_LOC = Vector((0.0, 1.0, 0.0))
    FRONTAL_CAM_LIGHT_WATTS = 40.0  # intensity of the bulb in watts
    FRONTAL_CAM_LIGHT_SHADOW = False
    FRONTAL_CAM_FOCAL_LENGTH = 100  # milimeters
    #
    SIDE_CAM_NAME = "SideCam"
    SIDE_CAM_DIST = FRONTAL_CAM_DIST
    SIDE_CAM_ANGLE = -60
    SIDE_CAM_LOC = (SIDE_CAM_DIST * cos(radians(SIDE_CAM_ANGLE)),
                    SIDE_CAM_DIST * sin(radians(SIDE_CAM_ANGLE)),
                    1.6)
    SIDE_CAM_ROT = rot_euler_degrees(86.0, 0.0, 90 + SIDE_CAM_ANGLE)
    SIDE_CAM_LIGHT_NAME = "SideCamLight"
    SIDE_CAM_LIGHT_LOC = Vector((0.0, 1.0, 0.0))
    SIDE_CAM_LIGHT_WATTS = 40.0  # intensity of the bulb in watts
    SIDE_CAM_LIGHT_SHADOW = False
    SIDE_CAM_FOCAL_LENGTH = 100  # milimeters

    # ##########################################################################
    # MAIN ROUTINE
    # ##########################################################################
    # general settings
    C.scene.world.node_tree.nodes["Background"].inputs[
        "Color"].default_value = BACKGROUND_COLOR

    # rendering
    # In older Blender versions, rendering wouldn't go above 60fps, causing
    # inconsistencies between data and renderings since MVNX has 240fps.
    # So we decided to skip 3 out of 4 frames, still yielding 60fps, which is
    # good for our purposes.
    C.scene.render.frame_map_old = 100
    C.scene.render.frame_map_new = 100
    C.scene.frame_step = 4  # jump by 4 frames

    #
    C.scene.render.resolution_x = RESOLUTION_WH[0]
    C.scene.render.resolution_y = RESOLUTION_WH[1]
    C.scene.render.resolution_percentage = RESOLUTION_PERCENTAGE
    C.scene.render.engine = "BLENDER_EEVEE"
    C.scene.eevee.use_taa_reprojection = EEVEE_VIEWPORT_DENOISING
    C.scene.eevee.taa_render_samples = EEVEE_RENDER_SAMPLES
    C.scene.eevee.taa_samples = EEVEE_VIEWPORT_SAMPLES
    if AS_VIDEO:
        C.scene.render.image_settings.file_format = "FFMPEG"
        C.scene.render.ffmpeg.format = "MPEG4"
        C.scene.render.ffmpeg.codec = "H264"
        C.scene.render.ffmpeg.audio_codec = "NONE"
        # also HIGH, MEDIUM, LOSSLESS...
        C.scene.render.ffmpeg.constant_rate_factor = "PERC_LOSSLESS"
    else:
        C.scene.render.image_settings.file_format = "PNG"
        C.scene.render.image_settings.color_depth = "16"
    #
    C.scene.render.image_settings.compression = 50
    C.scene.render.image_settings.color_mode = "BW"  # "RGBA"
    C.scene.render.filepath = OUT_DIR
    #
    # set all 3D screens to RENDERED mode
    set_shading_mode(INIT_SHADING_MODE, D.screens)

    # set fullscreen
    if INIT_3D_MAXIMIZED:
        maximize_layout_3d_area()

    # select and delete all objects
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    purge_unused_data()

    # # add a sun
    # bpy.ops.object.light_add(type="SUN", location=SUN_LOC, rotation=SUN_ROT)
    # C.object.name = SUN_NAME
    # C.object.data.name = SUN_NAME
    # C.object.data.energy = SUN_STRENGTH

    # add frontal cam
    bpy.ops.object.camera_add(location=FRONTAL_CAM_LOC,
                              rotation=FRONTAL_CAM_ROT)
    frontal_cam = C.object
    C.object.name = FRONTAL_CAM_NAME
    C.object.data.name = FRONTAL_CAM_NAME
    C.object.data.lens = FRONTAL_CAM_FOCAL_LENGTH
    # add side cam
    bpy.ops.object.camera_add(location=SIDE_CAM_LOC, rotation=SIDE_CAM_ROT)
    C.object.name = SIDE_CAM_NAME
    C.object.data.name = SIDE_CAM_NAME
    C.object.data.lens = SIDE_CAM_FOCAL_LENGTH

    # # add light as a child of cam
    # bpy.ops.object.light_add(type="POINT", location=CAM_LIGHT_LOC)
    # C.object.name = CAM_LIGHT_NAME
    # C.object.data.name = CAM_LIGHT_NAME
    # C.object.data.energy = CAM_LIGHT_WATTS
    # C.object.parent = get_obj(CAM_NAME)
    # C.object.data.use_shadow = False

    try:
        armature, mvnx = load_mvnx_into_blender(
            C, MVNX_PATH, SCHEMA_PATH,
            connectivity="CONNECTED",  # "INDIVIDUAL",
            scale=1.0,
            frame_start=FRAME_START,
            inherit_rotations=True,
            add_identity_pose=False,
            add_t_pose=False,
            verbose=True)
        mvnx_fps = int(mvnx.mvnx.subject.attrib["frameRate"])
        seq_len = len(
            armature.animation_data.action.fcurves[0].keyframe_points)
        RENDER_FPS = mvnx_fps // C.scene.frame_step  # expected: from 240 to 60
    except Exception as e:
        if isinstance(e, lxml.etree.DocumentInvalid):
            print("MNVX didn't pass given validation schema.",
                  "Remove schema path to bypass validation.")
        else:
            print("Something went wrong:", e)
    if FRAME_END is not None:
        assert FRAME_END > FRAME_START, "Frame end must be bigger than start!"
        fe = C.scene.frame_end
        new_fe = int(FRAME_END)
        if new_fe < fe:
            fe = new_fe
    else:
        # subtract 1 because [start, end] instead of [start, end) and Blender
        # would render a frame at the end with no animation
        FRAME_END = FRAME_START + seq_len - 1
    C.scene.frame_end = FRAME_END

    # readjust armature position
    armature.location = MVNX_POSITION
    armature.rotation_euler = MVNX_ROTATION

    # define glowing material for all spheres
    sphere_material = bpy.data.materials.new(name="sphere_material")
    sphere_material.use_nodes = True
    bsdf_inputs = sphere_material.node_tree.nodes["Principled BSDF"].inputs
    bsdf_inputs["Specular"].default_value = 0
    bsdf_inputs["Emission"].default_value = DOT_COLOR

    # frames_metadata, config_frames, normal_frames = mvnx.extract_frame_info()
    # fcurves = {pb.name: [] for pb in armature.pose.bones}
    # spheres = {}
    # for b in armature.data.bones:
    #     bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3,
    #                                           radius=DOT_DIAMETER / 2,
    #                                           location=(0, 0, 0))
    #     sph = C.object
    #     spheres[b.name] = sph
    #     sph.data.materials.append(sphere_material)

    # print(">>>>>>", fcurves, spheres)

    # This snippet creates icospheres at the tail of used_bones
    for b in armature.data.bones:
        if b.name in USED_BONES:
            bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3,
                                                  radius=DOT_DIAMETER / 2,
                                                  location=(0, 0, 0))
            sph = C.object
            sph.data.materials.append(sphere_material)
            sph.parent = armature
            sph.parent_type = "BONE"
            sph.parent_bone = b.name
            #
            # if b.name in ["LeftFoot", "RightFoot"]:
            #     # set heels to the floor if given
            #     constraint = bone.constraints.new('COPY_ROTATION')
            #     b.constraints["Child Of"].use_location_z = False

    # ADD CUSTOM SPHERES:

    # add right upper leg
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3,
                                          radius=DOT_DIAMETER / 2,
                                          location=(0, 0, 0))
    sph = C.object
    sph.data.materials.append(sphere_material)
    sph.parent = armature
    sph.parent_type = "BONE"
    sph.parent_bone = "RightUpperLeg"
    sph.location[1] -= armature.pose.bones["RightUpperLeg"].length
    # widen hips
    sph.location[2] -= armature.pose.bones["RightUpperLeg"].length * 0.15

    # add left upper leg
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3,
                                          radius=DOT_DIAMETER / 2,
                                          location=(0, 0, 0))
    sph = C.object
    sph.data.materials.append(sphere_material)
    sph.parent = armature
    sph.parent_type = "BONE"
    sph.parent_bone = "LeftUpperLeg"
    sph.location[1] -= armature.pose.bones["LeftUpperLeg"].length
    # widen hips
    sph.location[2] += armature.pose.bones["LeftUpperLeg"].length * 0.15

    # add right lower leg
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3,
                                          radius=DOT_DIAMETER / 2,
                                          location=(0, 0, 0))
    sph = C.object
    sph.data.materials.append(sphere_material)
    sph.parent = armature
    sph.parent_type = "BONE"
    sph.parent_bone = "RightLowerLeg"
    sph.location[1] += armature.pose.bones["RightLowerLeg"].length * 0.15

    # add left lower leg
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3,
                                          radius=DOT_DIAMETER / 2,
                                          location=(0, 0, 0))
    sph = C.object
    sph.data.materials.append(sphere_material)
    sph.parent = armature
    sph.parent_type = "BONE"
    sph.parent_bone = "LeftLowerLeg"
    sph.location[1] += armature.pose.bones["LeftLowerLeg"].length * 0.15

    # add column
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3,
                                          radius=DOT_DIAMETER / 2,
                                          location=(0, 0, 0))
    sph = C.object
    sph.data.materials.append(sphere_material)
    sph.parent = armature
    sph.parent_type = "BONE"
    sph.parent_bone = "L5"
    sph.location[1] -= armature.pose.bones["L5"].length

    # add head
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3,
                                          radius=DOT_DIAMETER / 2,
                                          location=(0, 0, 0))
    sph = C.object
    sph.data.materials.append(sphere_material)
    sph.parent = armature
    sph.parent_type = "BONE"
    sph.parent_bone = "Head"
    sph.location[1] -= armature.pose.bones["Head"].length * 0.618

    # add right hand
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3,
                                          radius=DOT_DIAMETER / 2,
                                          location=(0, 0, 0))
    sph = C.object
    sph.data.materials.append(sphere_material)
    sph.parent = armature
    sph.parent_type = "BONE"
    sph.parent_bone = "RightHand"
    sph.location[1] -= armature.pose.bones["RightHand"].length * 0.618

    # add left hand
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3,
                                          radius=DOT_DIAMETER / 2,
                                          location=(0, 0, 0))
    sph = C.object
    sph.data.materials.append(sphere_material)
    sph.parent = armature
    sph.parent_type = "BONE"
    sph.parent_bone = "LeftHand"
    sph.location[1] -= armature.pose.bones["LeftHand"].length * 0.618

    # Go over the to-be-rendered frames and record the pixel positions of the
    # spheres for a given cam
    CAM = frontal_cam
    CAM_COORDINATES = [
        {"frame_rate": RENDER_FPS,
         "frame_rate_explanation": "The given number is how many " +
         "frames of this JSON file takes in 1 second, regardless " +
         "of their actual frame value.",
         "pos_explanation": "The 3D positions are given with " +
         "respect to the camera as (x, y, z), where (x, y) go " +
         "from (0, 0) (left, bottom) to (1, 1) (right, top), and " +
         "z is the distance between the camera and the point in " +
         "world units (usually m)."}]
    icospheres = [(v.parent_bone, v) for k, v in D.objects.items()
                  if "Icosphere" in k]
    pose_bones = {pb.name: pb for pb in D.objects[armature.name].pose.bones}
    for frame_i in range(C.scene.frame_start, C.scene.frame_end + 1,
                         C.scene.frame_step):
        print("Collecting positions for frame >>>", frame_i)
        C.scene.frame_set(frame_i)
        # C.scene.frame_current = frame_i
        # bpy.context.view_layer.update()  # does nothing?
        data = {"frame": frame_i}
        for ico_bone, ico in icospheres:
            # get PoseBone global position
            pb_tail = pose_bones[ico_bone].tail
            pb_head = pose_bones[ico_bone].head
            # get PoseBone cam-relative position (x, y) where x goes from
            # left (0) to right(1), and y from bottom (0) to top(1)
            pb_tail_cam_xyz = world_to_camera_view(
                C.scene, frontal_cam, pb_tail)
            pb_head_cam_xyz = world_to_camera_view(
                C.scene, frontal_cam, pb_head)
            # get Icosphere global pos: contained in the last column
            # of "matrix_world"
            ico_loc_xyz = Vector(ico.matrix_world.transposed()[3][0:3])
            ico_cam_xyz = world_to_camera_view(C.scene, CAM, ico_loc_xyz)
            data[str((ico_bone, ico.name))] = {"bone_tail": pb_tail_cam_xyz[:],
                                               "bone_head": pb_head_cam_xyz[:],
                                               "sphere_pos": ico_cam_xyz[:]}
        CAM_COORDINATES.append(data)
    C.scene.frame_set(0)
    #
    mvnx_basename = os.path.splitext(os.path.basename(MVNX_PATH))[0]
    json_path = os.path.join(OUT_DIR, mvnx_basename + ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(CAM_COORDINATES, f, ensure_ascii=False, indent=4)
        print("Saved camera positions to", json_path)

    C.scene.render.filepath += "mvnx_basename"
    # Finally render the sequence
    C.scene.camera = frontal_cam
    if RENDER_HEADLESS:
        C.scene.render.fps = RENDER_FPS
        bpy.ops.render.render(animation=True)
    # bpy.ops.screen.animation_play()
