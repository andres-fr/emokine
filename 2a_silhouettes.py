#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Given a folder with N images assumed to belong to a sequence where a human
moves on a fixed background, this script performs DL human segmentation,
saving a binarized version for each image, where the background is black and
the human silhouette is white.
"""


import os
import multiprocessing
from itertools import repeat
# for OmegaConf
from dataclasses import dataclass
from typing import Optional
#
import numpy as np
import torch
from torchvision.transforms import Resize
from PIL import Image
import wget
from omegaconf import OmegaConf, MISSING
#
from detectron2.config import LazyConfig, instantiate
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import ROIMasks
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers.mask_ops import paste_masks_in_image

#
from emokine.utils import get_lab_distance, otsu_hist_median
from emokine.utils import resize_hw
# import matplotlib.pyplot as plt


# ##############################################################################
# # HELPERS
# ##############################################################################
def human_segmentation_setup(
        url="new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py",
        model_dir=os.path.join("output", "model_snapshots", "segmentation")):
    """
    :param url: Model URL from the Detectron2 URL model zoo.
    :param model_dir: Where the pre-trained model will be saved
    :returns: ``(model, cfg, classes)``, where ``model`` is the pre-trained
      pytorch model, ``cfg`` is the detectron2 config, and ``classes`` is
      a list of class names as output by the model.

    More info:
    https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md
    https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html
    """
    # load config and replace every SyncBN with BN
    cfg = LazyConfig.load(model_zoo.get_config_file(url))
    if cfg.model.backbone.norm == "SyncBN":
        cfg.model.backbone.norm = "BN"
    if cfg.model.backbone.bottom_up.norm == "SyncBN":
        cfg.model.backbone.bottom_up.norm = "BN"
    # sanity checks
    assert cfg.model.input_format in {"BGR", "RGB"}, cfg.model.input_format

    # list of class names as output by the model
    classes = MetadataCatalog.get(
        cfg.dataloader.test.dataset.names).thing_classes

    # load model from zoo
    model = instantiate(cfg.model)

    # load model parameters from zoo, download if not existing
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
    return model, cfg, classes


def silhouette_inference(img_t, model):
    """
    :param img_t: Float tensor of shape ``(3, h, w)``.
    :param model: Detectron2 model of type ``GeneralizedRCNN``.

    This function works like ``model.inference`` in detectron2 but omits the
    final thresholding, returning a soft output as uint8.

    https://detectron2.readthedocs.io/en/latest/tutorials/models.html#partially-execute-a-model
    https://github.com/facebookresearch/detectron2/blob/38af375052d3ae7331141bc1a22cfa2713b02987/detectron2/modeling/meta_arch/rcnn.py#L178
    https://github.com/facebookresearch/detectron2/blob/38af375052d3ae7331141bc1a22cfa2713b02987/detectron2/layers/mask_ops.py#L74
    """
    _, h, w = img_t.shape
    result = model.inference(
        [{"image": img_t, "height": h, "width": w}],
        do_postprocess=False)[0]
    # expand predicted masks to HxW heatmaps
    roi_masks = ROIMasks(result.pred_masks[:, 0, :, :]).tensor
    heatmaps = retry_if_cuda_oom(paste_masks_in_image)(
        roi_masks, result.pred_boxes.tensor, (h, w),
        threshold=-1)
    # replace masks with expanded heatmaps and return
    result.pred_masks = heatmaps
    return result


# ##############################################################################
# # CLI
# ##############################################################################
@dataclass
class ConfDef:
    """
    :cvar IMGS_DIR: Path to a directory containing only the images to be
      processed. It is assumed that all images have the same shape and format,
      and come from a sequence containing people on a static background.
    :cvar SEG_URL: Detectron2 URL for the segmentation model, e.g.
      ``new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py``
    :cvar SEG_MODEL_DIR: Where the segmentation model from detectron2 is
      expected to be downloaded/stored.
    :cvar TARGET_CLASS: Name of the class that we want to extract silhouettes
      from. It must be a class supported by the detectron2 model.
    :cvar MIN_IDX: Once sorted by their name, images will be processed starting
      with index 0, unless a different starting index is given here.
    :cvar MAX_IDX: Once sorted by their name, images will be processed until
      the last one, unless a different max index is given here.
    :cvar SKIP_N: Once sorted by their name, images will be processed one
      by one. If this parameter is N, one out of N will be processed, and the
      rest skipped.
    :cvar SIL_INPUT_SIZE: Given size ``h*w`` for all images in pixels, a
      rescaling will be applied for the DL inference, so that the smaller
      dimension equals this parameter, and aspect ratio is maintained.
    :cvar BG_ESTIMATION_RATIO: The lowest ``x`` DL predicted values across the
      sequence will be taken into account and averaged to estimate the
      background color. This parameter determines how many of the lowest will
      be used, as a proportion (ratio) of the whole sequence.
    :cvar MEDIAN_FILT_SIZE: After thresholding, spatial median filter is
      applied. This determines the size of the filter.
    :cvar INCLUDE_DL_ABOVE: If given, DL predictions equal or greater than this
      threshold are guaranteed to be part of the solution.
    :cvar DEVICE: PyTorch device, e.g. ``cuda`` or ``cpu``
    """
    IMGS_DIR: str = MISSING
    SEG_URL: str = "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py"
    SEG_MODEL_DIR: str = os.path.join("output", "model_snapshots",
                                      "segmentation")
    TARGET_CLASS: str = "person"
    #
    MIN_IDX: Optional[int] = None
    MAX_IDX: Optional[int] = None
    SKIP_N: Optional[int] = None
    #
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    SIL_INPUT_SIZE: Optional[int] = 400
    BG_ESTIMATION_RATIO: float = 0.02
    MEDIAN_FILT_SIZE: int = 5
    INCLUDE_DL_ABOVE: Optional[float] = None


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

    model, cfg, classes = human_segmentation_setup(
        CONF.SEG_URL, CONF.SEG_MODEL_DIR)
    target_cls_idx = classes.index(CONF.TARGET_CLASS)
    model = model.to(CONF.DEVICE)
    model.eval()

    # Human DL segmentation
    img_modes = []
    img_arrs = []
    dl_heatmaps = []
    with torch.no_grad():
        for i, ip in enumerate(img_paths):
            print(f"DL inference: [{i}/{num_imgs}]: {ip}")
            # load (h, w, c) image (typically RGB uint8) and optionally resize:
            img = Image.open(ip)
            arr = np.array(img).astype(np.float32)
            img_arrs.append(arr)

            if CONF.SIL_INPUT_SIZE is not None:
                img_resized = Resize(CONF.SIL_INPUT_SIZE)(img)
            else:
                img_resized = img

            # convert to float32 tensor, permute, swap channels
            t = torch.as_tensor(np.array(img_resized).astype(
                np.float32)).permute(2, 0, 1).to(CONF.DEVICE)
            assert img.mode in {"BGR", "RGB"}, img.mode
            if img.mode != cfg.model.input_format:
                t = t.flip(0)  # flip channels
            img_modes.append(img.mode)

            # perform inference
            out = silhouette_inference(t, model)

            # extract masks corresponding to target class
            target_out = out.pred_masks[
                torch.where(out.pred_classes == target_cls_idx)]
            if target_out.numel() >= 1:
                target_out = target_out.max(dim=0)[0]
            else:  # if no masks found, we get all zeros
                target_out = torch.zeros_like(out.pred_masks[0])
            # optionally resize back and gather result
            if CONF.SIL_INPUT_SIZE is not None:
                target_out = resize_hw(target_out, img.size[::-1])
            dl_heatmaps.append(target_out.cpu())
    img_arrs = np.stack(img_arrs)
    dl_heatmaps = torch.stack(dl_heatmaps).numpy()
    # idx=10; plt.clf(); plt.imshow(dl_heatmaps[idx]); plt.show()

    # get background as an average of colors with lowest human detection
    # bg_uncertainty gives away regions where person detection covers the
    # bg and we are not cetain about the color.
    print(f"Extracting background...")
    pick_k = round(CONF.BG_ESTIMATION_RATIO * len(img_arrs))
    k_idxs = np.argpartition(dl_heatmaps, pick_k, axis=0,
                             kind='introselect', order=None)[:pick_k]
    mean_bg = np.take_along_axis(
        img_arrs, k_idxs[:, :, :, None], axis=0).mean(axis=0)
    bg_uncertainty = np.take_along_axis(
        dl_heatmaps, k_idxs, axis=0).mean(axis=0) / 255.0
    assert (bg_uncertainty >= 0).all() and (bg_uncertainty <= 1).all(), \
        "bg_uncertainty should be between 0 and 1!"
    del k_idxs
    # plt.clf(); plt.imshow(mean_bg / 255); plt.show()
    # plt.clf(); plt.imshow(bg_uncertainty); plt.show()

    print(f"Computing LAB residuals...")
    with multiprocessing.Pool() as pool:
        residual_dists = np.stack(
            pool.starmap(get_lab_distance,
                         zip(img_arrs, repeat(mean_bg), repeat(np.float32))))
    residual_dists /= residual_dists.max()
    # idx=10; plt.clf(); plt.imshow(residual_dists[idx]); plt.show()

    print("Fusion and thresholding...")
    residual_dists *= (dl_heatmaps / 255)

    # INCLUDE_DL_ABOVE: Optional[float] = None
    fusion = ((dl_heatmaps / 255) * bg_uncertainty) + (residual_dists *
                                                       (1 - bg_uncertainty))
    with multiprocessing.Pool() as pool:
        fusion_masks = pool.starmap(
            otsu_hist_median,
            zip(fusion, repeat(CONF.MEDIAN_FILT_SIZE)))
    # idx=10; plt.clf(); plt.imshow(fusion[idx]); plt.show()
    # idx=10; plt.clf(); plt.imshow(fusion_masks[idx]); plt.show()

    if CONF.INCLUDE_DL_ABOVE is not None:
        thresh = round(CONF.INCLUDE_DL_ABOVE * 255)
        dl_masks = (dl_heatmaps >= thresh)
        fusion_masks = [fm | dlm for fm, dlm in zip(fusion_masks, dl_masks)]
        # idx=10; plt.clf(); plt.imshow(dl_masks[idx]); plt.show()

    for i, (ip, fm) in enumerate(zip(img_paths, fusion_masks)):
        outpath = ip + "__silhouette.png"
        Image.fromarray(fm).save(outpath)
        print(f"[{i}/{num_imgs}]", "Saved to", outpath)
