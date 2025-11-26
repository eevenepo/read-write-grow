"""
Video enhancement utilities:

- AI colorization using OpenCV DNN (Zhang et al. colorization model).
- Optional ESRGAN upscaling (if Real-ESRGAN / PyTorch are installed).
- Optional temporal smoothing between frames.

This module is used by the BioZip video pipeline after the DNA decode
to optionally post-process the reconstructed grayscale video.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Global flags mainly for debugging / logging.
ESRGAN_AVAILABLE: bool = False
ESRGAN_IMPORT_ERROR: Optional[Exception] = None


# -------------------------------------------------------------------------
# Colorization model loading (OpenCV DNN)
# -------------------------------------------------------------------------

def _find_file_with_candidates(
    model_dir: Path,
    candidates: List[str],
    description: str,
) -> Path:
    for name in candidates:
        p = model_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find {description} in {model_dir}. "
        f"Tried: {', '.join(candidates)}"
    )


def load_colorization_model(
    model_dir: str | Path,
) -> Tuple[cv2.dnn_Net, np.ndarray]:
    model_dir = Path(model_dir)

    proto_path = _find_file_with_candidates(
        model_dir,
        ["colorization_deploy_v2.prototxt", "colorization.prototxt"],
        "colorization prototxt",
    )

    caffemodel_path = _find_file_with_candidates(
        model_dir,
        ["colorization_release_v2.caffemodel", "colorization.caffemodel"],
        "colorization caffemodel",
    )

    pts_path = _find_file_with_candidates(
        model_dir,
        ["pts_in_hull.npy", "pts_in_hull.npy"],
        "pts_in_hull cluster centers (.npy)",
    )

    logger.info(f"Loading colorization net from:\n  {proto_path}\n  {caffemodel_path}")
    net = cv2.dnn.readNetFromCaffe(str(proto_path), str(caffemodel_path))

    logger.info(f"Loading pts_in_hull from: {pts_path}")
    pts_in_hull = np.load(str(pts_path))
    pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)

    class8_id = net.getLayerId("class8_ab")
    conv8_id = net.getLayerId("conv8_313_rh")

    net.getLayer(class8_id).blobs = [pts_in_hull.astype("float32")]
    net.getLayer(conv8_id).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    return net, pts_in_hull


def _grayscale_to_bw_photo(gray: np.ndarray) -> np.ndarray:
    """
    Convert a grayscale video frame to look more like a traditional B&W photo.
    
    The Zhang colorization model was trained on black & white photographs,
    which have different characteristics than grayscale video:
    - Higher contrast
    - More distinct tonal regions
    - Histogram closer to full range
    
    This preprocessing helps the colorization network produce better results.
    """
    # 1. Histogram equalization to use full tonal range (like developed film)
    equalized = cv2.equalizeHist(gray)
    
    # 2. Blend with original to not be too harsh (50/50 mix)
    blended = cv2.addWeighted(gray, 0.5, equalized, 0.5, 0)
    
    # 3. Apply slight contrast curve (S-curve) for that photo look
    # This mimics the characteristic curve of photographic film
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        # S-curve: darken shadows slightly, brighten highlights
        normalized = i / 255.0
        # Sigmoid-like curve
        curved = normalized ** 0.9 if normalized < 0.5 else 1 - (1 - normalized) ** 0.9
        curved = 0.3 * normalized + 0.7 * curved  # Blend with linear
        lut[i] = np.clip(int(curved * 255), 0, 255)
    
    result = cv2.LUT(blended, lut)
    
    return result


def colorize_frame(frame_bgr: np.ndarray, net: cv2.dnn_Net, saturation_boost: float = 1.0) -> np.ndarray:
    """
    Colorize a single BGR frame using the Zhang et al. colorization network.
    
    This implementation follows the EXACT reference implementation from:
    https://github.com/AbhilipsaJena/Image_colorization-OpenCV
    
    Args:
        frame_bgr: Input frame in BGR format (grayscale or color)
        net: Loaded colorization network  
        saturation_boost: Factor to boost color saturation (1.0 = no change)
    
    Returns:
        Colorized BGR image (uint8)
    """
    # Handle grayscale input - convert to BGR
    if frame_bgr.ndim == 2:
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
    elif frame_bgr.ndim == 3 and frame_bgr.shape[2] == 1:
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
    
    # Store original dimensions
    H_orig, W_orig = frame_bgr.shape[:2]
    
    # === EXACT Reference Implementation ===
    # Step 1: Scale image to float32 [0, 1]
    scaled = frame_bgr.astype("float32") / 255.0
    
    # Step 2: Convert BGR to LAB (note: BGR2LAB, not RGB2Lab)
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    
    # Step 3: Resize LAB image to network input size (224x224)
    resized = cv2.resize(lab, (224, 224))
    
    # Step 4: Extract L channel from resized image and mean-center
    L = cv2.split(resized)[0]
    L -= 50
    
    # Step 5: Forward pass through network
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    
    # Step 6: Resize ab predictions back to original image size
    ab = cv2.resize(ab, (W_orig, H_orig))
    
    # Step 7: Apply saturation boost if requested
    if saturation_boost != 1.0:
        ab = ab * saturation_boost
    
    # Step 8: Get L channel from ORIGINAL full-size LAB image
    L_orig = cv2.split(lab)[0]
    
    # Step 9: Concatenate L with predicted ab
    colorized = np.concatenate((L_orig[:, :, np.newaxis], ab), axis=2)
    
    # Step 10: Convert LAB back to BGR
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    
    # Step 11: Convert back to uint8
    colorized = (255 * colorized).astype("uint8")
    
    return colorized


def _suppress_color_artifacts(img_bgr: np.ndarray, threshold: float = 0.4) -> np.ndarray:
    """
    Suppress unnatural color artifacts (like random green/magenta patches).
    
    Works by detecting pixels with very high saturation in areas that should
    likely be neutral (gray/brown) and desaturating them.
    """
    # Convert to HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    
    # Normalize saturation
    s_norm = s / 255.0
    
    # Detect overly saturated pixels (likely artifacts)
    # Natural beach/sand scenes shouldn't have extremely saturated colors
    oversaturated_mask = s_norm > threshold
    
    # Reduce saturation of these pixels
    s[oversaturated_mask] = s[oversaturated_mask] * 0.5
    
    # Merge and convert back
    hsv_out = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_out.astype("uint8"), cv2.COLOR_HSV2BGR)


def _apply_color_correction(img_bgr: np.ndarray, saturation: float = 1.0, contrast: float = 1.0) -> np.ndarray:
    """
    Apply color correction adjustments to an image.
    
    Args:
        img_bgr: Input BGR image
        saturation: Saturation multiplier (1.0 = no change)
        contrast: Contrast multiplier (1.0 = no change)
    """
    # Convert to HSV for saturation adjustment
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype("float32")
    
    # Adjust saturation
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    
    # Convert back
    result = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    
    # Adjust contrast
    if contrast != 1.0:
        result = cv2.convertScaleAbs(result, alpha=contrast, beta=0)
    
    return result


def _auto_white_balance(img_bgr: np.ndarray) -> np.ndarray:
    """
    Apply automatic white balance using the gray world assumption.
    This helps correct color casts from the colorization model.
    """
    # Calculate channel means
    b, g, r = cv2.split(img_bgr.astype("float32"))
    
    # Gray world: assume average color should be gray
    avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    
    if avg_gray < 1:
        return img_bgr
    
    # Scale each channel
    b = np.clip(b * (avg_gray / max(avg_b, 1)), 0, 255)
    g = np.clip(g * (avg_gray / max(avg_g, 1)), 0, 255)
    r = np.clip(r * (avg_gray / max(avg_r, 1)), 0, 255)
    
    return cv2.merge([b, g, r]).astype("uint8")

# -------------------------------------------------------------------------
# ESRGAN Upscaling (optional)
# -------------------------------------------------------------------------

class ESRGANUpscaler:
    """
    Wrapper around Real-ESRGAN for 4x upscaling.
    """

    def __init__(self, model_path: str | Path, device: str = "cuda"):
        global ESRGAN_AVAILABLE, ESRGAN_IMPORT_ERROR

        try:
            from biozip_video.rrdbnet_arch import RRDBNet
            import torch
        except Exception as e:
            ESRGAN_AVAILABLE = False
            ESRGAN_IMPORT_ERROR = e
            logger.exception("PyTorch imports failed")
            raise RuntimeError(
                "PyTorch is not available. Check log."
            ) from e

        # Force CPU if CUDA not available
        if device != "cpu" and not torch.cuda.is_available():
            logger.warning("CUDA not available. Falling back to CPU.")
            device = "cpu"

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ESRGAN model file not found: {model_path}")

        # RRDBNet architecture for RealESRGAN_x4plus
        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
        
        try:
            loadnet = torch.load(str(model_path), map_location=torch.device(device), weights_only=True)
            if 'params_ema' in loadnet:
                keyname = 'params_ema'
            elif 'params' in loadnet:
                keyname = 'params'
            else:
                # Direct state dict
                keyname = None
            
            if keyname:
                self.model.load_state_dict(loadnet[keyname], strict=True)
            else:
                self.model.load_state_dict(loadnet, strict=True)
            
            self.model.eval()
            self.model = self.model.to(device)
        except Exception as e:
            raise RuntimeError(f"Failed to load ESRGAN weights: {e}") from e

        self.device = device
        self.torch = torch
        ESRGAN_AVAILABLE = True
        logger.info(f"Loaded ESRGAN model from {model_path} on device={device}")

    def upscale(self, frame_bgr: np.ndarray, sharpen_amount: float = 0.5) -> np.ndarray:
        """
        Upscale 4x with post-processing for sharper, more natural results.
        
        Args:
            frame_bgr: Input frame in BGR format
            sharpen_amount: Amount of sharpening (0.0-1.0, default 0.5)
        """
        # 1. AI Upscale
        # Pre-process: BGR -> RGB, 0-255 -> 0-1, HWC -> CHW
        img = frame_bgr.astype(np.float32) / 255.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        img = img.unsqueeze(0).to(self.device)

        # Inference
        with self.torch.no_grad():
            output = self.model(img)

        # Post-process: CHW -> HWC, 0-1 -> 0-255, RGB -> BGR
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output, (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        out_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        # 2. Multi-scale sharpening for better detail restoration
        if sharpen_amount > 0:
            # Convert to float for processing
            img_float = out_bgr.astype(np.float32)
            
            # Multi-scale unsharp mask: combine fine and coarse details
            # Fine details (small sigma)
            blur_fine = cv2.GaussianBlur(img_float, (0, 0), 1.0)
            detail_fine = img_float - blur_fine
            
            # Medium details
            blur_medium = cv2.GaussianBlur(img_float, (0, 0), 2.0)
            detail_medium = img_float - blur_medium
            
            # Combine: original + weighted details
            # Fine details get more weight for sharpness
            sharpened = img_float + detail_fine * sharpen_amount * 1.5 + detail_medium * sharpen_amount * 0.5
            
            out_bgr = np.clip(sharpened, 0, 255).astype("uint8")
        
        return out_bgr


def upscale_frame_bicubic(frame_bgr: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    Fallback upscaling with Lanczos (better than bicubic for video).
    """
    h, w = frame_bgr.shape[:2]
    # Lanczos4 is sharper than standard Cubic
    return cv2.resize(frame_bgr, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)

# -------------------------------------------------------------------------
# Temporal smoothing
# -------------------------------------------------------------------------

def smooth_frames(prev_frame: Optional[np.ndarray], current_frame: np.ndarray, alpha: float) -> np.ndarray:
    if prev_frame is None:
        return current_frame
    return cv2.addWeighted(current_frame, alpha, prev_frame, 1.0 - alpha, 0)


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------

def enhance_video(
    input_path: str,
    output_path: str,
    *,
    do_colorize: bool = True,
    do_upscale: bool = False,
    do_smooth: bool = True,
    do_white_balance: bool = True,
    smooth_alpha: float = 0.3,
    saturation_boost: float = 1.4,
    color_model_dir: str | Path = "models",
    esrgan_model_path: str | Path | None = None,
    esrgan_device: str = "cuda",
    target_fps: Optional[float] = None,
) -> str:
    """
    Enhance a video with colorization, upscaling, and temporal smoothing.
    
    Args:
        input_path: Path to input video
        output_path: Path for output video
        do_colorize: Whether to apply AI colorization
        do_upscale: Whether to apply AI upscaling (4x with ESRGAN or 2x bicubic)
        do_smooth: Whether to apply temporal smoothing
        do_white_balance: Whether to apply automatic white balance
        smooth_alpha: Smoothing strength (higher = more smoothing)
        saturation_boost: Color saturation multiplier (1.0 = no change)
        color_model_dir: Directory containing colorization model files
        esrgan_model_path: Path to ESRGAN model weights
        esrgan_device: Device for ESRGAN ('cuda' or 'cpu')
        target_fps: Target FPS for output (None = keep original)
    """
    input_path_p = Path(input_path)
    if not input_path_p.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    cap = cv2.VideoCapture(str(input_path_p))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"Input video: {input_path} ({width}x{height} @ {fps:.3f} fps)")

    # Load models
    color_net: Optional[cv2.dnn_Net] = None
    if do_colorize:
        color_net, _ = load_colorization_model(color_model_dir)
        logger.info("Colorization model loaded")

    esrgan: Optional[ESRGANUpscaler] = None
    if do_upscale:
        if esrgan_model_path is None:
            logger.warning("do_upscale=True but no esrgan_model_path provided. Using bicubic.")
        else:
            try:
                esrgan = ESRGANUpscaler(esrgan_model_path, device=esrgan_device)
                logger.info("ESRGAN upscaler initialized")
            except Exception as e:
                logger.warning(f"ESRGAN init failed ({e}). Falling back to bicubic.")
                esrgan = None

    # Determine scale
    upscale_scale = 1
    if do_upscale:
        upscale_scale = 4 if esrgan is not None else 2

    processed_frames: List[np.ndarray] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame is None:
            continue

        # Ensure BGR
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # -------------------------------------------------------
        # STEP 0: ENSURE TRUE GRAYSCALE
        # -------------------------------------------------------
        # The decoded video might have slight color variations from compression
        # Convert to grayscale and back to ensure clean input
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # -------------------------------------------------------
        # STEP 1: UPSCALING FIRST (on grayscale - gives better results)
        # -------------------------------------------------------
        # Upscaling the grayscale first gives the colorizer more detail to work with
        if do_upscale:
            try:
                if esrgan is not None:
                    frame = esrgan.upscale(frame, sharpen_amount=0.5)
                else:
                    frame = upscale_frame_bicubic(frame, scale=upscale_scale)
            except Exception as e:
                logger.error(f"Upscaling failed frame {frame_idx}: {e}")

        # -------------------------------------------------------
        # STEP 2: COLORIZATION (on upscaled frame for better quality)
        # -------------------------------------------------------
        if do_colorize and color_net is not None:
            try:
                frame = colorize_frame(frame, color_net, saturation_boost=saturation_boost)
            except Exception as e:
                logger.error(f"Colorization failed frame {frame_idx}: {e}")
        
        # -------------------------------------------------------
        # STEP 3: WHITE BALANCE (Optional color correction)
        # -------------------------------------------------------
        if do_white_balance and do_colorize:
            frame = _auto_white_balance(frame)
        
        processed_frames.append(frame)

    cap.release()

    if not processed_frames:
        raise RuntimeError("No frames processed from input video.")

    # Frame properties might have changed (upscaling)
    out_h, out_w = processed_frames[0].shape[:2]
    out_fps = float(target_fps) if target_fps is not None else fps

    # Resample FPS if needed
    if target_fps is not None and abs(out_fps - fps) > 1e-3:
        duration = len(processed_frames) / fps
        desired_count = int(round(duration * out_fps))
        if desired_count > 0:
            logger.info(f"Resampling frames: {len(processed_frames)} -> {desired_count}")
            resampled = []
            for i in range(desired_count):
                t = i / out_fps
                src_idx = int(round(t * fps))
                src_idx = max(0, min(src_idx, len(processed_frames) - 1))
                resampled.append(processed_frames[src_idx])
            processed_frames = resampled

    # Temporal smoothing
    if do_smooth and processed_frames:
        logger.info("Applying temporal smoothing (centered sliding window)...")
        smoothed_frames = []
        # Use a centered window to avoid phase lag (ghosting behind the object)
        # Reduced window radius to 1 (3 frames total) to minimize motion blur
        window_radius = 1
        
        # Pre-convert to float to avoid repeated casting if memory allows, 
        # but for video it might be too big. We'll do it on the fly.
        
        count = len(processed_frames)
        for i in range(count):
            start_idx = max(0, i - window_radius)
            end_idx = min(count, i + window_radius + 1)
            
            # Stack frames in the window
            window_stack = processed_frames[start_idx:end_idx]
            
            # Weighted average with Scene Change Detection (Temporal Bilateral-ish)
            # 1. Base temporal weights (centered)
            center_rel_idx = i - start_idx
            weights = np.ones(len(window_stack), dtype=np.float32)
            weights[center_rel_idx] = 3.0 
            
            # 2. Scene similarity weights
            # Compare each frame in window to the center frame
            # If difference is high (scene cut), reduce weight to near zero
            center_frame_small = cv2.resize(window_stack[center_rel_idx], (64, 36)) # Downscale for speed
            
            for w_idx, fr in enumerate(window_stack):
                if w_idx == center_rel_idx:
                    continue
                
                fr_small = cv2.resize(fr, (64, 36))
                # L1 distance (Mean Absolute Difference)
                diff = np.mean(np.abs(center_frame_small.astype(float) - fr_small.astype(float)))
                
                # Threshold: if MAD > 30 (out of 255), it's likely a different scene or massive motion
                # We use a soft decay: weight *= exp(-diff^2 / sigma^2)
                # sigma = 20 seems reasonable for scene cuts
                similarity = np.exp(-(diff**2) / (2 * 20**2))
                weights[w_idx] *= similarity

            # Normalize
            weights /= weights.sum()
            
            # Reshape for broadcasting: (N, 1, 1, 1) if frames are (H, W, C)
            # But window_stack is a list of arrays.
            # Stack them: (N, H, W, C)
            stack_arr = np.stack(window_stack, axis=0).astype(np.float32)
            
            # Weighted sum
            # weights shape: (N,) -> (N, 1, 1, 1)
            w_arr = weights.reshape(-1, 1, 1, 1)
            avg_frame = np.sum(stack_arr * w_arr, axis=0).astype("uint8")
            
            smoothed_frames.append(avg_frame)
            
        processed_frames = smoothed_frames

    # Write output
    # Use avc1 (H.264) if available for better quality/compression than mp4v
    try:
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
    except:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        
    out = cv2.VideoWriter(str(output_path), fourcc, out_fps, (out_w, out_h))
    if not out.isOpened():
        # Fallback if avc1 fails
        logger.warning("Could not open VideoWriter with avc1, falling back to mp4v")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, out_fps, (out_w, out_h))
        
    if not out.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {output_path}")

    for fr in processed_frames:
        out.write(fr)
    out.release()

    logger.info(f"Saved: {output_path} ({out_w}x{out_h})")
    return str(output_path)


def colorize_video_file(input_path: str, output_path: str, model_dir: str | Path = "models") -> str:
    # Backwards compatibility wrapper
    return enhance_video(
        input_path=input_path, output_path=output_path,
        do_colorize=True, do_upscale=False, do_smooth=False,
        color_model_dir=model_dir,
    )

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="enhanced_output.mp4")
    parser.add_argument("--color", action="store_true")
    parser.add_argument("--upscale", action="store_true")
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("--color_model_dir", default="models")
    args = parser.parse_args()

    enhance_video(
        input_path=args.input, output_path=args.output,
        do_colorize=args.color, do_upscale=args.upscale, do_smooth=args.smooth,
        color_model_dir=args.color_model_dir
    )