"""
Wraps Segment Anything (SAM) to pull individual clothing-item crops out of
a single 'closet dump' style photo (bed, floor, rack, whatever).

Ships with a mock grid-segmenter fallback so you can build and test the
rest of the pipeline (attribute extraction, confidence gating, review loop)
today, before you've downloaded a SAM checkpoint. Swap in the real
checkpoint whenever — the rest of the pipeline doesn't change.

Real segmentation setup:
    pip install segment-anything torch torchvision opencv-python-headless
    Download a checkpoint (vit_b is the smallest, ~375MB):
        https://github.com/facebookresearch/segment-anything#model-checkpoints
    Then pass --sam-checkpoint /path/to/sam_vit_b_01ec64.pth to pipeline.py
"""

from __future__ import annotations

import os
import uuid
from typing import List

import numpy as np
from PIL import Image


class ClothingSegmenter:
    def __init__(
        self,
        checkpoint_path: str | None = None,
        model_type: str = "vit_b",
        device: str = "cpu",
    ):
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.device = device
        self._mask_generator = self._try_load_sam()

    def _try_load_sam(self):
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            print(
                "[ClothingSegmenter] No SAM checkpoint found — using mock grid "
                "segmenter. Pass --sam-checkpoint to enable real segmentation."
            )
            return None
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

            sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
            sam.to(device=self.device)
            return SamAutomaticMaskGenerator(
                sam,
                points_per_side=24,
                pred_iou_thresh=0.88,
                stability_score_thresh=0.92,
                min_mask_region_area=2000,
            )
        except ImportError:
            print(
                "[ClothingSegmenter] segment_anything not installed — "
                "using mock grid segmenter."
            )
            return None

    def segment(self, image_path: str, output_dir: str) -> List[str]:
        os.makedirs(output_dir, exist_ok=True)
        image = Image.open(image_path).convert("RGB")

        if self._mask_generator is None:
            return self._mock_segment(image, output_dir)

        arr = np.array(image)
        masks = self._mask_generator.generate(arr)
        masks = self._filter_masks(masks, arr.shape)

        crop_paths = []
        for m in masks:
            crop = self._crop_from_mask(arr, m["segmentation"])
            crop_path = os.path.join(output_dir, f"{uuid.uuid4().hex[:8]}.png")
            Image.fromarray(crop).save(crop_path)
            crop_paths.append(crop_path)
        return crop_paths

    def _filter_masks(self, masks, image_shape):
        """Drop background-sized blobs and specks/slivers unlikely to be a
        single garment. Tune these thresholds against your own test photos —
        they're starting heuristics, not validated numbers."""
        h, w = image_shape[:2]
        image_area = h * w
        keep = []
        for m in masks:
            area_ratio = m["area"] / image_area
            _, _, bw, bh = m["bbox"]
            aspect = bw / max(bh, 1)
            if 0.01 < area_ratio < 0.6 and 0.2 < aspect < 5.0:
                keep.append(m)
        return keep

    def _crop_from_mask(self, arr: np.ndarray, seg_mask: np.ndarray) -> np.ndarray:
        ys, xs = np.where(seg_mask)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        cropped = arr[y0 : y1 + 1, x0 : x1 + 1].copy()
        mask_crop = seg_mask[y0 : y1 + 1, x0 : x1 + 1]
        rgba = np.dstack([cropped, (mask_crop * 255).astype(np.uint8)])
        return rgba

    def _mock_segment(self, image: Image.Image, output_dir: str) -> List[str]:
        """3x3 grid crop — good enough to exercise the full pipeline end to
        end. This is NOT real segmentation; replace before you trust any
        detection-quality numbers."""
        w, h = image.size
        crop_paths = []
        rows, cols = 3, 3
        for r in range(rows):
            for c in range(cols):
                box = (
                    c * w // cols,
                    r * h // rows,
                    (c + 1) * w // cols,
                    (r + 1) * h // rows,
                )
                crop = image.crop(box)
                crop_path = os.path.join(output_dir, f"grid_{r}_{c}.png")
                crop.save(crop_path)
                crop_paths.append(crop_path)
        return crop_paths
