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


def colorize_frame(frame_bgr: np.ndarray, net: cv2.dnn_Net) -> np.ndarray:
    """
    Colorize a single BGR frame using the loaded DNN net.
    """
    h, w = frame_bgr.shape[:2]

    # 1. Preprocessing: Convert to standard normalized float for the model
    # Ensure 3 channels even if input is effectively gray
    if frame_bgr.ndim == 2:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2RGB)
    else:
        # If already BGR, we convert to RGB for the Lab conversion step
        # because cv2.COLOR_BGR2Lab expects BGR, but we want to be safe with float ranges.
        # Actually, let's stick to standard BGR->Lab flow:
        frame_rgb = frame_bgr

    img_float = frame_rgb.astype("float32") / 255.0
    
    # OpenCV float32 Lab: L is [0..100], a/b are raw (approx -127..127)
    img_lab = cv2.cvtColor(img_float, cv2.COLOR_BGR2Lab)
    L_channel = img_lab[:, :, 0] 

    # 2. Resize L to 224x224 and subtract 50 (model normalization)
    L_input = cv2.resize(L_channel, (224, 224))
    L_input -= 50.0

    # 3. Forward pass
    net.setInput(cv2.dnn.blobFromImage(L_input))
    ab_dec = net.forward()[0, :, :, :].transpose((1, 2, 0))  # (224, 224, 2)

    # 4. Resize ab back to original resolution
    ab_upsampled = cv2.resize(ab_dec, (w, h))

    # --- THE FIX IS HERE ---
    
    # 5. Robust Merge Strategy (Convert to uint8 manually)
    
    # L_channel is 0..100. Scale to 0..255 for uint8
    L_uint8 = np.clip(L_channel * (255.0 / 100.0), 0, 255).astype("uint8")
    
    # Optional: Apply CLAHE to L channel to improve local contrast
    # Reduced clipLimit to 1.0 to avoid "deep fried" look on noisy inputs
    # Increased tileGridSize to 16x16 for smoother, less patchy contrast
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16, 16))
    L_uint8 = clahe.apply(L_uint8)

    # Boost saturation in ab channels
    # ab is approx -128..127. Multiplying by >1.0 increases saturation.
    # Increased to 1.1 to combat the "sepia" look of the model
    saturation_factor = 1.1
    ab_upsampled = ab_upsampled * saturation_factor

    # ab_upsampled is ALREADY in raw range (approx -110 to 110).
    # We DO NOT multiply by 128. We only add 128 to center it at 0 for uint8.
    ab_uint8 = np.clip(ab_upsampled + 128.0, 0, 255).astype("uint8")

    # Merge channels
    lab_final = cv2.merge([L_uint8, ab_uint8])

    # 6. Convert Lab (uint8) -> BGR
    # OpenCV handles uint8 Lab strictly: L:0-255, a:0-255, b:0-255 (with 128 bias)
    bgr_out = cv2.cvtColor(lab_final, cv2.COLOR_Lab2BGR)

    return bgr_out

# -------------------------------------------------------------------------
# ESRGAN Upscaling (optional)
# -------------------------------------------------------------------------

class ESRGANUpscaler:
    """
    Wrapper around Real-ESRGAN with added sharpening to prevent 
    the "watercolor/oil-painting" effect on low-res inputs.
    """

    def __init__(self, model_path: str | Path, device: str = "cuda"):
        global ESRGAN_AVAILABLE, ESRGAN_IMPORT_ERROR

        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            import torch
        except Exception as e:
            ESRGAN_AVAILABLE = False
            ESRGAN_IMPORT_ERROR = e
            logger.exception("Real-ESRGAN imports failed")
            raise RuntimeError(
                "Real-ESRGAN is not available. Check log."
            ) from e

        # Force CPU if CUDA not available
        if device != "cpu" and not torch.cuda.is_available():
            logger.warning("CUDA not available. Falling back to CPU.")
            device = "cpu"

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ESRGAN model file not found: {model_path}")

        # standard RealESRGAN_x4plus architecture
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

        try:
            self.er = RealESRGANer(
                scale=4,
                model_path=str(model_path),
                model=model,
                tile=400,      # Tiling helps reduce memory usage and sometimes artifacts
                tile_pad=10,
                pre_pad=0,
                half=(device == "cuda"), # Use FP16 only on CUDA
                device=device,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ESRGAN: {e}") from e

        self.device = device
        ESRGAN_AVAILABLE = True
        logger.info(f"Loaded ESRGAN model from {model_path} on device={device}")

    def upscale(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Upscale 4x and apply sharpening to reduce 'plastic' look.
        """
        # 1. AI Upscale
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        out_rgb, _ = self.er.enhance(frame_rgb, outscale=4)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

        # 2. Post-Process: Unsharp Mask (Sharpening)
        # The AI tends to over-smooth low-res video. We add clarity back.
        # Increased strength to 1.3 since we are now denoising the input first.
        gaussian = cv2.GaussianBlur(out_bgr, (0, 0), 2.0)
        out_sharpened = cv2.addWeighted(out_bgr, 1.3, gaussian, -0.3, 0)
        
        # Clip to valid range just in case
        return np.clip(out_sharpened, 0, 255).astype("uint8")


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
    smooth_alpha: float = 0.3,
    color_model_dir: str | Path = "models",
    esrgan_model_path: str | Path | None = None,
    esrgan_device: str = "cuda",
    target_fps: Optional[float] = None,
) -> str:
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
        # STEP 0: PRE-PROCESSING (Deblocking / Denoising)
        # -------------------------------------------------------
        # High CRF (40) leaves blocking artifacts. We lightly denoise 
        # the low-res input to help the AI models focus on structure.
        if do_colorize or do_upscale:
            # h=3 is subtle enough to keep details but smooth out block noise
            frame = cv2.fastNlMeansDenoising(frame, None, h=3.0, templateWindowSize=7, searchWindowSize=21)

        # -------------------------------------------------------
        # STEP 1: COLORIZATION (Run on original resolution first)
        # -------------------------------------------------------
        if do_colorize:
            if color_net is not None:
                try:
                    frame = colorize_frame(frame, color_net)
                except Exception as e:
                    logger.error(f"Colorization failed frame {frame_idx}: {e}")

        # -------------------------------------------------------
        # STEP 2: UPSCALING (Run on the colorized frame)
        # -------------------------------------------------------
        if do_upscale:
            try:
                if esrgan is not None:
                    frame = esrgan.upscale(frame)
                else:
                    frame = upscale_frame_bicubic(frame, scale=upscale_scale)
            except Exception as e:
                logger.error(f"Upscaling failed frame {frame_idx}: {e}")
        
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