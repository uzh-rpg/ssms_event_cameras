import os

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1

from pathlib import Path
import sys

current_filepath = Path(os.path.realpath(__file__))
sys.path.insert(0, str(current_filepath.parent.parent.parent))
from typing import Tuple, Optional

import imageio.v3 as iio
import torch as th
from tqdm import tqdm

from data.utils.types import DataType, DatasetType
from data.genx_utils.sequence_for_streaming import SequenceForIter
from data.genx_utils.labels import ObjectLabels
from utils.evaluation.prophesee.io.box_loading import loaded_label_to_prophesee
from callbacks.viz_base import VizCallbackBase
import cv2
import numpy as np
import bbox_visualizer as bbv
import hdf5plugin

LABELMAP_GEN1 = ("car", "pedestrian")
LABELMAP_GEN4_SHORT = ("pedestrian", "two wheeler", "car")


def draw_bboxes_bbv(
    img, boxes, labelmap=LABELMAP_GEN1, hd_resolution: bool = False
) -> np.ndarray:
    """
    draw bboxes in the image img
    """
    colors = cv2.applyColorMap(np.arange(0, 255).astype(np.uint8), cv2.COLORMAP_HSV)
    colors = [tuple(*item) for item in colors.tolist()]

    if labelmap == LABELMAP_GEN1:
        classid2colors = {
            0: (255, 255, 0),  # car -> yellow (rgb)
            1: (0, 0, 255),  # ped -> blue (rgb)
        }
        scale_multiplier = 4
    else:
        assert labelmap == LABELMAP_GEN4_SHORT
        classid2colors = {
            0: (0, 0, 255),  # ped -> blue (rgb)
            1: (0, 255, 255),  # 2-wheeler cyan (rgb)
            2: (255, 255, 0),  # car -> yellow (rgb)
        }
        scale_multiplier = 1 if hd_resolution else 2

    add_score = True
    ht, wd, ch = img.shape
    dim_new_wh = (int(wd * scale_multiplier), int(ht * scale_multiplier))
    if scale_multiplier != 1:
        img = cv2.resize(img, dim_new_wh, interpolation=cv2.INTER_AREA)
    for i in range(boxes.shape[0]):
        pt1 = (int(boxes["x"][i]), int(boxes["y"][i]))
        size = (int(boxes["w"][i]), int(boxes["h"][i]))
        pt2 = (pt1[0] + size[0], pt1[1] + size[1])
        bbox = (pt1[0], pt1[1], pt2[0], pt2[1])
        bbox = tuple(x * scale_multiplier for x in bbox)

        score = boxes["class_confidence"][i]
        class_id = boxes["class_id"][i]
        class_name = labelmap[class_id % len(labelmap)]
        bbox_txt = class_name
        if add_score:
            bbox_txt += f" {score:.2f}"
        color_tuple_rgb = classid2colors[class_id]
        img = bbv.draw_rectangle(img, bbox, bbox_color=color_tuple_rgb)
        img = bbv.add_label(
            img, bbox_txt, bbox, text_bg_color=color_tuple_rgb, top=True
        )

    return img


def draw_predictions(
    ev_repr: th.Tensor,
    predictions_proph,
    hd_resolution: bool = False,
    labelmap=LABELMAP_GEN4_SHORT,
):
    img = VizCallbackBase.ev_repr_to_img(ev_repr.cpu().numpy())
    if predictions_proph is not None:
        img = draw_bboxes_bbv(
            img, predictions_proph, labelmap=labelmap, hd_resolution=hd_resolution
        )
    return img


def gen_gt_generator(
    seq_path: Path,
    ev_representation_name: str,
    downsample_by_factor_2: bool,
    dataset_type: DatasetType = DatasetType.GEN4,
) -> Tuple[th.Tensor, Optional[ObjectLabels]]:
    sequence_length = 5

    if dataset_type == DatasetType.GEN1:
        map_dataset = SequenceForIter(
            path=seq_path,
            ev_representation_name=ev_representation_name,
            sequence_length=sequence_length,
            dataset_type=DatasetType.GEN1,
            downsample_by_factor_2=downsample_by_factor_2,
        )
    else:
        map_dataset = SequenceForIter(
            path=seq_path,
            ev_representation_name=ev_representation_name,
            sequence_length=sequence_length,
            dataset_type=DatasetType.GEN4,
            downsample_by_factor_2=downsample_by_factor_2,
        )

    iter_dataset = map_dataset.to_iter_datapipe()

    for data in iter_dataset:
        seq_ev_reprs = data[DataType.EV_REPR]
        seq_labels = data[DataType.OBJLABELS_SEQ]

        for idx, ev_repr in enumerate(seq_ev_reprs):
            labels = seq_labels[idx]
            yield ev_repr, labels


if __name__ == "__main__":
    SEQUENCE_PATH = "/data/scratch1/nzubic/datasets/RVT/gen1_frequencies/gen1_40hz/test/17-04-04_11-00-13_cut_15_500000_60500000/"
    OUT_DIR_PATH = "/data/scratch1/nzubic/out_viz/"
    DOWNSAMPLE = False
    EV_REPR_NAME = "stacked_histogram_dt=25_nbins=10"  # dt varies depending on different frequencies
    DATASET_TYPE = DatasetType.GEN1

    seq_path = Path(SEQUENCE_PATH)
    out_dir = Path(OUT_DIR_PATH)
    os.makedirs(out_dir, exist_ok=False)

    if DATASET_TYPE == DatasetType.GEN1:
        labelmap = LABELMAP_GEN1
    else:
        labelmap = LABELMAP_GEN4_SHORT

    viz_at_hd_resolution = None
    prev_img_with_labels = None
    for idx, (ev_repr, labels) in enumerate(
        tqdm(
            gen_gt_generator(
                seq_path=seq_path,
                ev_representation_name=EV_REPR_NAME,
                downsample_by_factor_2=DOWNSAMPLE,
                dataset_type=DATASET_TYPE,
            )
        )
    ):
        if viz_at_hd_resolution is None:
            height, width = ev_repr.shape[-2:]
            viz_at_hd_resolution = height * width > 9e5

        have_labels = labels is not None
        labels_proph = loaded_label_to_prophesee(labels) if have_labels else None
        img = draw_predictions(
            ev_repr=ev_repr,
            predictions_proph=labels_proph,
            hd_resolution=viz_at_hd_resolution,
            labelmap=labelmap,
        )

        filename = f"{idx}".zfill(6) + ".png"
        img_filepath = out_dir / filename

        if have_labels or prev_img_with_labels is None:
            img_to_write = img
        else:
            img_to_write = prev_img_with_labels

        iio.imwrite(str(img_filepath), img_to_write)

        if labels_proph is not None:
            prev_img_with_labels = img
