#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Functionality to visually inspect 3D keypoint sequences, mostly for debugging
and data inspection.
"""


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# import matplotlib.animation as pla


# #############################################################################
# # GLOBALS
# #############################################################################
# ["aliceblue", "antiquewhite", "aqua", "aquamarine", "azure",
# "beige", "bisque", "black", "blanchedalmond", "blue", "blueviolet",
# "brown", "burlywood", "cadetblue", "chartreuse", "chocolate",
# "coral", "cornflowerblue", "cornsilk", "crimson", "cyan",
# "darkblue", "darkcyan", "darkgoldenrod", "darkgray", "darkgreen",
# "darkgrey", "darkkhaki", "darkmagenta", "darkolivegreen",
# "darkorange", "darkorchid", "darkred", "darksalmon", "darkseagreen",
# "darkslateblue", "darkslategray", "darkslategrey", "darkturquoise",
# "darkviolet", "deeppink", "deepskyblue", "dimgray", "dimgrey",
# "dodgerblue", "firebrick", "floralwhite", "forestgreen", "fuchsia",
# "gainsboro", "ghostwhite", "gold", "goldenrod", "gray", "green",
# "greenyellow", "grey", "honeydew", "hotpink", "indianred", "indigo",
# "ivory", "khaki", "lavender", "lavenderblush", "lawngreen",
# "lemonchiffon", "lightblue", "lightcoral", "lightcyan",
# "lightgoldenrodyellow", "lightgray", "lightgreen", "lightgrey",
# "lightpink", "lightsalmon", "lightseagreen", "lightskyblue",
# "lightslategray", "lightslategrey", "lightsteelblue", "lightyellow",
# "lime", "limegreen", "linen", "magenta", "maroon",
# "mediumaquamarine", "mediumblue", "mediumorchid", "mediumpurple",
# "mediumseagreen", "mediumslateblue", "mediumspringgreen",
# "mediumturquoise", "mediumvioletred", "midnightblue", "mintcream",
# "mistyrose", "moccasin", "navajowhite", "navy", "oldlace", "olive",
# "olivedrab", "orange", "orangered", "orchid", "palegoldenrod",
# "palegreen", "paleturquoise", "palevioletred", "papayawhip",
# "peachpuff", "peru", "pink", "plum", "powderblue", "purple",
# "rebeccapurple", "red", "rosybrown", "royalblue", "saddlebrown",
# "salmon", "sandybrown", "seagreen", "seashell", "sienna", "silver",
# "skyblue", "slateblue", "slategray", "slategrey", "snow",
# "springgreen", "steelblue", "tan", "teal", "thistle", "tomato",
# "turquoise", "violet", "wheat", "white", "whitesmoke", "yellow",
# "yellowgreen"]
ALL_PLT_COLORS = {k.replace("tab:", ""): v  # RGB colors
                  for k, v in mcolors.CSS4_COLORS.items()}


# pelvis to head 7 items. Then 4 each: right arm, left arm, right leg, left leg
MVNX_COLORS = tuple(ALL_PLT_COLORS[k] for k in
                    ["salmon", "indianred", "firebrick", "maroon",
                     "lightgrey", "darkgrey", "black",
                     "mediumpurple", "blueviolet", "darkviolet", "indigo",
                     "lightsteelblue", "cornflowerblue", "royalblue", "navy",
                     "cyan", "darkturquoise", "lightseagreen", "teal",
                     # "khaki", "gold", "goldenrod", "darkgoldenrod",
                     "greenyellow", "limegreen", "forestgreen", "olivedrab"])

PI = 3.141592653589793


# #############################################################################
# # xx
# #############################################################################
def plot_3d_pose(xyz_pos_values, colors=MVNX_COLORS, diameter=1,
                 title="3D Pose",
                 x_range_mm=(-2000, 2000), y_range_mm=(-2000, 2000),
                 z_range_mm=(0, 4000)):
    """
    :param xyz_pos_frames: A list in the form ``[frame1, frame2, ...]``
      where each frame is a list in the form ``[(x1, y1, z1), ...]``, each
      xyz triple corresponding to the 3D position of a keypoint. All
      keypointsmust always be in the same order, frames must be ordered by
      time.

    3D scatterplot of a specific pose.
    """
    surface = PI * (diameter / 2) ** 2
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    xxx, yyy, zzz = zip(*xyz_pos_values)
    if colors is not None:
        assert len(colors) == len(xyz_pos_values), "#colors must match #(xyz)!"
    ax_scatter = ax.scatter(
        xxx, yyy, zzz, marker="o", depthshade=False, c=colors, s=surface)
    #
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax_title = ax.set_title(title)
    #
    ax.set_xlim3d(x_range_mm)
    ax.set_ylim3d(y_range_mm)
    ax.set_zlim3d(z_range_mm)
    #
    return fig, ax, ax_scatter, ax_title


class PoseAnimation3D:
    """
    Functor providing 3D scatterplot frames in a way that can be animated by
    the PLT animation engine. See ``animation_3d_mvnx`` for an usage example.
    """

    def __init__(self, xyz_pos_frames, scat, title):
        """
        :param xyz_pos_frames: A list in the form ``[frame1, frame2, ...]``
          where each frame is a list in the form ``[(x1, y1, z1), ...]``, each
          xyz triple corresponding to the 3D position of a keypoint. All
          keypointsmust always be in the same order, frames must be ordered by
          time.
        """
        self.xyz_pos_frames = xyz_pos_frames
        self.scat = scat
        self.title = title
        self.ori_title = title.get_text()

    def __call__(self, frame_idx):
        """
        """
        # https://stackoverflow.com/a/41609238/4511978
        xxx, yyy, zzz = zip(*self.xyz_pos_frames[frame_idx])
        self.scat._offsets3d[0][:] = xxx
        self.scat._offsets3d[1][:] = yyy
        self.scat._offsets3d[2][:] = zzz
        #
        self.title.set_text(self.ori_title + f" frame={frame_idx}")
        #
        return self.scat, self.title


def animation_3d_mvnx(mvnx_pos_dataframe, colors=MVNX_COLORS, diameter=10,
                      begin_frame=None, end_frame=None,
                      skip_every=5, repeat=True):
    """
    :param mvnx_pos_dataframe: A Pandas dataframe with columns like
      ``Pelvix_x, Pelvis_y, ...``. Use ``df.loc[:, "ori":"dest"]`` to
      slice columns

    Usage example:

    m = Mvnx(MVNX_PATH, SCHEMA_PATH)
    processor = MvnxToTabular(m)
    dataframes = processor(FIELDS)
    animation_3d_mvnx(dataframes["position"], MVNX_COLORS)
    """
    xyz_pos_values = [
        row.values.reshape(-1, 3) * 1000 for f_idx, row in
        mvnx_pos_dataframe.loc[:, "Pelvis_x":"LeftToe_z"].iterrows()]
    if begin_frame is None:
        begin_frame = 0
    if end_frame is None:
        end_frame = len(xyz_pos_values)
    # frame_range = range(begin_frame, end_frame, skip_every)
    fig, ax, ax_scat, ax_title = plot_3d_pose(xyz_pos_values[0], colors=colors,
                                              diameter=diameter,
                                              z_range_mm=(0, 2500))
    # ani = pla.FuncAnimation(
    #     fig, PoseAnimation3D(xyz_pos_values, ax_scat, ax_title),
    #     frame_range, interval=1, repeat=repeat, blit=False)
    plt.show()
