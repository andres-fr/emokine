#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
This script performs face masking/blurring on a collection of images expected
to contain one person each. It performs the following steps:
1. Loads image
2. Performs human keypoint detection and isolates the most prominent person
3. Isolates selected keypoints and averages their location to find the head
4. Extracts a patch around the head and performs face detection (segmentation)
5. Optionally, transforms the predicted mask into a fitted ellipse
6. Saves output as binary mask or as original image with the blurred mask.

The pipeline includes potential resizing before steps 2 and 4, to ensure that
the neural networks being used get images of appropriate scale.
"""


import os
# for OmegaConf
from dataclasses import dataclass
from typing import Optional, List
#
import numpy as np
import torch
from torchvision.transforms import Resize
from PIL import Image, ImageFilter
import wget
from omegaconf import OmegaConf, MISSING
#
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
#
import face_segmentation_pytorch as fsp
#
from emokine.utils import make_elliptic_mask, resize_crop_bbox
# import matplotlib.pyplot as plt


# ##############################################################################
# # HELPERS
# ##############################################################################
def human_keypoints_setup(
        url="COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml",
        model_dir=os.path.join("output", "model_snapshots", "keypoints")):
    """
    :param url: Model URL from the Detectron2 URL model zoo.
    :param model_dir: Where the pre-trained model will be saved
    :returns: ``(model, cfg, meta)``, where ``model`` is the pre-trained
      pytorch model, ``cfg`` is the detectron2 config, and ``meta`` is

    More info:
    https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md
    """
    # load config
    cfg = model_zoo.get_config(url)
    # sanity checks
    assert cfg.INPUT.FORMAT in {"BGR", "RGB"}, cfg.model.input_format
    model_dir = os.path.join("output", "model_snapshots", "keypoints")
    #
    meta = {"size_range": (cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST)}
    # load model from zoo
    model = build_model(cfg)
    model_ckpt = DetectionCheckpointer(model, save_dir=model_dir)
    try:
        # try to load pre-downloaded snapshot
        ckpt = os.path.join(model_dir, os.listdir(model_dir)[0])
        model_ckpt.load(ckpt)
    except (FileNotFoundError, IndexError):
        os.makedirs(model_dir, exist_ok=True)
        # we assume this is the first time and download the snapshot
        ckpt_url = model_zoo.get_checkpoint_url(url)
        down_path = os.path.join(model_dir, os.path.basename(ckpt_url))
        print("Downloading", ckpt_url, "\nto", down_path)
        wget.download(ckpt_url, down_path)
        # and retry to load snapshot
        ckpt = os.path.join(model_dir, os.listdir(model_dir)[0])
        model_ckpt.load(ckpt)
    #
    return model, cfg, meta


def main_person_inference(img_t, model, threshold=0.5):
    """
    :param img_t: Float tensor of shape ``(3, h, w)``.
    :param model: Detectron2 model that provides ``pred_keypoints``.
    :returns: None if no person with conficence above threshold is found, and
      the ``(K, 3)`` keypoints otherwise (signaling ``x, y, confidence``).
    """
    # predict head position
    _, h, w = img_t.shape
    result = model.inference(
        [{"image": img_t, "height": h, "width": w}])[0]["instances"]
    person_accepted = (result.scores >= threshold)
    if person_accepted.sum() > 0:
        main_person_idx = result.scores.argmax().item()
        main_person_kps = result.pred_keypoints[main_person_idx]
        return main_person_kps
    #
    return None


def thresh_avg_kps(kps, thresh=0.5):
    """
    :param kps: Tensor of shape ``(num_kps, 3)``, where each row contains
      a ``(x, y, score)`` triple.
    :returns: ``(x_avg, y_avg), n``, where ``n`` determines how many points
      were used to obtain the average (i.e. how many had a confidence above
      the threshold). If ``n=0``, returns ``None, 0``.
    """
    kps = kps[kps[:, 2] >= thresh]
    num_kps = len(kps)
    if num_kps == 0:
        return None, 0
    else:
        x_avg, y_avg = kps.mean(dim=0)[:2]
        return (x_avg.item(), y_avg.item()), num_kps


# ##############################################################################
# # CLI
# ##############################################################################
@dataclass
class ConfDef:
    """
    :cvar IMGS_DIR: Path to a directory containing only the images to be
      processed. It is assumed that all images have the same shape and format,
      and come from a sequence containing people on a static background.
    :cvar KP_URL: Detectron2 URL for the keypoint estimation model, e.g.
      ``COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml``
    :cvar KP_MODEL_DIR: Where the keypoint estimation model from detectron2 is
      expected to be downloaded/stored.
    :cvar FM_MODEL_DIR: Where the face masking model is expected to be
      downloaded/stored.
    :cvar MIN_IDX: Once sorted by their name, images will be processed starting
      with index 0, unless a different starting index is given here.
    :cvar MAX_IDX: Once sorted by their name, images will be processed until
      the last one, unless a different max index is given here.
    :cvar SKIP_N: Once sorted by their name, images will be processed one
      by one. If this parameter is N, one out of N will be processed, and the
      rest skipped.
    :cvar DEVICE: PyTorch device, e.g. ``cuda`` or ``cpu``
    :cvar KP_INPUT_SIZE: Given size ``h*w`` for an input image in pixels, a
      rescaling will be applied for the keypoint estimation inference, so that
      the smaller dimension equals this param, and aspect ratio is maintained.
    :cvar FM_INPUT_SIZE: The input to the face masking model will be rescaled
      so that the smaller dimension equals this param. Note that the Nirkin et
      al model works best in the ballpark of 400-500 (in pixels).
    :cvar PERSON_THRESHOLD: Between 0 and 1, any detected person below this
      confidence will not be considered.
    :cvar KP_THRESHOLD: Between 0 and 1, any detected keypoints below this
      confidence will not be considered
    :cvar FM_THRESHOLD: Once the face heatmap is computed as a matrix of values
      between 0 and 1, the mask is extracted by applying this threshold.
    :cvar BBOX_SIZE: Once the ``KP_SELECTION`` is found, a bounding box of
      this size (in pixels) will be drawn around its middlepoint. This will be
      the patch send to the face masking model, after resizing via
      ``FM_INPUT_SIZE``.
    :cvar GAUSS_BLUR_STD: If not given, results will be stored as black/white
      binary face masks. If given, results will be the images, where the region
      covered by the mask is blurred out using the given standard deviation,
      in pixels.
    :cvar BORDERS_BLUR_STD: If ``GAUSS_BLUR_STD`` is active, this parameter
      regulated the sharpness of the transition between the blurred and
      non-blurred regions, in pixels.
    :cvar ELLIPTIC_MASK: If not given, the mask is the actual output from the
      face masking model. If given, the mask is fitted to an ellipse and this
      parameter determines the scale of the resulting ellipse (e.g. 2 means
      the ellipse axes will be 2 times longer). It is a ratio.
    :cvar KP_CLASSES: The keypoint detection model is expected to output K
      triples in the form ``(x_location, y_location, confidence)``, each
      corresponding to a specific keypoint. This list of length K determines
      the name of the corresponding keypoints.
    :cvar KP_SELECTION: To extract the bounding box sent to the face mask
      model, we identify a selection of keypoints and take their average. This
      list determines which keypoints will be taken into account.
    """
    IMGS_DIR: str = MISSING
    KP_URL: str = "COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"
    KP_MODEL_DIR: str = os.path.join("output", "model_snapshots", "keypoints")
    FM_MODEL_DIR: str = os.path.join("output", "model_snapshots", "face_mask")
    #
    MIN_IDX: Optional[int] = None
    MAX_IDX: Optional[int] = None
    SKIP_N: Optional[int] = None
    #
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    KP_INPUT_SIZE: Optional[int] = 800
    FM_INPUT_SIZE: Optional[int] = 450
    #
    PERSON_THRESHOLD: float = 0.5
    KP_THRESHOLD: float = 0.5
    FM_THRESHOLD: float = 0.5
    #
    BBOX_SIZE: int = 100
    GAUSS_BLUR_STD: Optional[float] = None
    BORDERS_BLUR_STD: float = 3
    ELLIPTIC_MASK: Optional[float] = None

    #
    KP_CLASSES: List[str] = ("nose",
                             "left_eye", "right_eye",
                             "left_ear", "right_ear",
                             "left_shoulder", "right_shoulder",
                             "left_elbow", "right_elbow",
                             "left_wrist", "right_wrist",
                             "left_hip", "right_hip",
                             "left_knee", "right_knee",
                             "left_ankle", "right_ankle")
    KP_SELECTION: List[str] = ("nose", "left_eye", "right_eye",
                               "left_ear", "right_ear")


# ##############################################################################
# # MAIN ROUTINE
# ##############################################################################
if __name__ == "__main__":

    CONF = OmegaConf.structured(ConfDef())
    cli_conf = OmegaConf.from_cli()
    CONF = OmegaConf.merge(CONF, cli_conf)
    print("\n\nCONFIGURATION:")
    print(OmegaConf.to_yaml(CONF), end="\n\n\n")

    img_paths = sorted([os.path.join(CONF.IMGS_DIR, p)
                        for p in os.listdir(CONF.IMGS_DIR)])[
                                CONF.MIN_IDX:CONF.MAX_IDX:CONF.SKIP_N]
    assert img_paths, "Empty images dir?"
    num_imgs = len(img_paths)

    # load keypoint estimation model
    kp_model, cfg, meta = human_keypoints_setup(CONF.KP_URL, CONF.KP_MODEL_DIR)
    min_size_range, max_size_range = meta["size_range"]
    assert (min_size_range <= CONF.KP_INPUT_SIZE <= max_size_range), \
        "KP_INPUT_SIZE must be in range {(min_size_range, max_size_range)!}"

    kp_sel_idxs = [CONF.KP_CLASSES.index(k) for k in CONF.KP_SELECTION]
    kp_model = kp_model.to(CONF.DEVICE)
    kp_model.eval()

    # load face mask model
    bbox_half = CONF.BBOX_SIZE / 2
    fm_model = fsp.model.FaceSegmentationNet()
    fsp.utils.load_model_parameters(fm_model, CONF.FM_MODEL_DIR)
    fm_model = fm_model.to(CONF.DEVICE)
    fm_model.eval()
    fm_mean_bgr = torch.tensor(fm_model.MEAN_BGR).type(torch.float32).to(
        CONF.DEVICE)

    # Main loop
    with torch.no_grad():
        for i, ip in enumerate(img_paths):
            print(f"DL inference: [{i}/{num_imgs}]: {ip}")
            # load (h, w, c) image (typically RGB uint8) and optionally resize:
            img = Image.open(ip).convert("RGB")
            arr = np.array(img).astype(np.float32)

            if CONF.KP_INPUT_SIZE is not None:
                img_resized = Resize(CONF.KP_INPUT_SIZE)(img)
            else:
                img_resized = img

            # convert to float32 tensor, permute, swap channels
            t = torch.as_tensor(np.array(img_resized).astype(
                np.float32)).permute(2, 0, 1).to(CONF.DEVICE)
            assert img.mode in {"BGR", "RGB"}, img.mode
            if img.mode != cfg.INPUT.FORMAT:
                t = t.flip(0)  # flip channels

            # mask inference
            face_heatmap = torch.zeros_like(t[0])
            kps = main_person_inference(t, kp_model, CONF.PERSON_THRESHOLD)
            if kps is not None:
                kp_avg, _ = thresh_avg_kps(kps[kp_sel_idxs], CONF.KP_THRESHOLD)
                if kp_avg is not None:
                    # if head found, extract bbox around it and crop tensor
                    x_avg, y_avg = kp_avg
                    h, w = t.shape[-2:]
                    x0, x1, y0, y1 = resize_crop_bbox(
                        x_avg - bbox_half, x_avg + bbox_half,
                        y_avg - bbox_half, y_avg + bbox_half,
                        max_x=w, max_y=h, expansion_ratio=1.0)
                    t_crop = t[:, y0:y1, x0:x1]
                    # normalize and resize
                    t_crop = fsp.utils.normalize_range(
                        t_crop, torch.float32, out_range=(0, 255))
                    t_crop = t_crop.permute(1, 2, 0).sub(
                        fm_mean_bgr).permute(2, 0, 1)
                    if CONF.FM_INPUT_SIZE is not None:
                        t_crop = Resize(CONF.FM_INPUT_SIZE)(t_crop)
                    # perform face segmentation
                    hm = fm_model(t_crop.unsqueeze(0), as_pmap=True)[0]
                    # plt.clf(); plt.imshow(t_crop[0].cpu()); plt.show()
                    # plt.clf(); plt.imshow(hm.cpu()); plt.show()

                    # paste hm on global domain
                    if CONF.FM_INPUT_SIZE is not None:
                        hm_sz = Resize(min(x1-x0, y1-y0))(hm.unsqueeze(0))[0]
                    face_heatmap[y0:y1, x0:x1] = hm_sz
            # extract global mask, possibly convert to elliptic
            face_mask = (face_heatmap >= CONF.FM_THRESHOLD).cpu().numpy()
            if CONF.ELLIPTIC_MASK is not None:
                face_mask = make_elliptic_mask(
                    face_mask, stretch=CONF.ELLIPTIC_MASK)
            # plt.clf(); plt.imshow(face_heatmap.cpu()); plt.show()
            # plt.clf(); plt.imshow(face_mask); plt.show()

            if CONF.GAUSS_BLUR_STD is None:
                # save binary mask
                outpath = ip + "__facemask.png"
                Image.fromarray(face_mask).save(outpath)
            else:
                mask_blur_fn = ImageFilter.GaussianBlur(
                    radius=CONF.BORDERS_BLUR_STD)
                img_blur_fn = ImageFilter.GaussianBlur(
                    radius=CONF.GAUSS_BLUR_STD)
                outpath = ip + "__faceblur.jpg"
                mask = Image.fromarray(face_mask).convert("L").filter(
                    mask_blur_fn)
                blur_img = img.filter(img_blur_fn)
                mix = Image.composite(blur_img, img, mask)
                mix.save(outpath)
            print(f"[{i}/{num_imgs}]", "Saved to", outpath)
