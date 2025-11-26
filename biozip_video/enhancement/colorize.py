from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class Colorizer:
    """
    Thin wrapper around the OpenCV DNN colorization network.

    Expects:
      - colorization_deploy_v2.prototxt
      - colorization_release_v2.caffemodel
      - pts_in_hull.npy

    Typical usage:
        colorizer = Colorizer("models/colorization")
        color_frame = colorizer.colorize_frame(gray_or_bgr_frame)
    """

    def __init__(self, model_dir: Path | str) -> None:
        self.model_dir = Path(model_dir)

        prototxt = self.model_dir / "colorization_deploy_v2.prototxt"
        caffemodel = self.model_dir / "colorization_release_v2.caffemodel"
        pts_path = self.model_dir / "pts_in_hull.npy"

        missing = [p for p in (prototxt, caffemodel, pts_path) if not p.exists()]
        if missing:
            missing_str = ", ".join(str(p) for p in missing)
            raise FileNotFoundError(
                f"Colorization model files not found: {missing_str}. "
                "Make sure you downloaded them into models/colorization/."
            )

        # Load the network
        self.net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))

        # Load cluster centers
        pts_in_hull = np.load(str(pts_path))  # shape (313, 2)
        # Reshape to (2, 313, 1, 1) as expected by the network
        pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)

        # Inject cluster centers into the model
        class8_id = self.net.getLayerId("class8_ab")
        conv8_id = self.net.getLayerId("conv8_313_rh")

        self.net.getLayer(class8_id).blobs = [pts_in_hull.astype(np.float32)]
        self.net.getLayer(conv8_id).blobs = [
            np.full((1, 313, 1, 1), 2.606, dtype=np.float32)
        ]

    def colorize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Colorize a single frame.

        Input:
            - frame:  HxW or HxWx1 (grayscale) OR HxWx3 (BGR)
        Output:
            - HxWx3 uint8 BGR color frame
        """
        if frame is None or frame.size == 0:
            raise ValueError("Empty frame passed to colorize_frame")

        # Ensure 3-channel BGR
        if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
            # grayscale -> BGR
            bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            bgr = frame

        h, w = bgr.shape[:2]

        # Convert to float and then to LAB
        img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        L = img_lab[:, :, 0]  # lightness channel

        # Resize to network input size
        L_rs = cv2.resize(L, (224, 224))
        L_rs -= 50  # mean-centering as in the original paper/code

        # Forward through network
        blob = cv2.dnn.blobFromImage(L_rs)
        self.net.setInput(blob)
        ab_dec = self.net.forward()[0, :, :, :].transpose((1, 2, 0))  # 224x224x2

        # Resize ab to original size
        ab_dec_us = cv2.resize(ab_dec, (w, h))

        # Combine L + ab
        L_orig = L[:, :, np.newaxis]
        lab_out = np.concatenate((L_orig, ab_dec_us), axis=2).astype(np.float32)

        # Back to BGR
        img_bgr_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
        img_bgr_out = np.clip(img_bgr_out, 0, 1)
        img_bgr_out = (img_bgr_out * 255).astype("uint8")

        return img_bgr_out


def colorize_frame(
    frame: np.ndarray,
    colorizer: Optional[Colorizer] = None,
    model_dir: Optional[str | Path] = None,
) -> np.ndarray:
    """
    Convenience function when you don't want to manage the Colorizer instance yourself.

    NOTE: For performance on video, you *should* create a Colorizer once and reuse it
    rather than calling this helper every frame.
    """
    if colorizer is None:
        if model_dir is None:
            model_dir = "models/colorization"
        colorizer = Colorizer(model_dir)

    return colorizer.colorize_frame(frame)
