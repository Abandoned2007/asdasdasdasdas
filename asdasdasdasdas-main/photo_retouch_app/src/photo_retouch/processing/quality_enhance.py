"""AI-based photo quality enhancement using Real-ESRGAN and similar models."""
import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from photo_retouch.core.models import QualityEnhanceSettings

logger = logging.getLogger(__name__)


@dataclass
class EnhancementResult:
    """Result of quality enhancement operation."""
    success: bool
    image: Optional[Image.Image] = None
    error: str = ""
    processing_time_ms: float = 0.0


class QualityEnhancer:
    """AI-powered photo quality enhancer using Real-ESRGAN/BSRGAN models."""

    def __init__(self, model_dir: Optional[Path] = None):
        """
        Initialize quality enhancer.

        Args:
            model_dir: Directory containing model weights. If None, uses default locations.
        """
        self.model_dir = model_dir or Path(__file__).parent.parent.parent / "models"
        self._models = {}
        self._current_model = None

    def _load_model(self, model_name: str) -> Optional[cv2.dnn.Net]:
        """Load a Real-ESRGAN model from ONNX file."""
        if model_name in self._models:
            return self._models[model_name]

        model_path = self.model_dir / f"{model_name}.onnx"
        if not model_path.exists():
            # Try alternative naming
            model_path = self.model_dir / f"{model_name.lower()}.onnx"

        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            return None

        try:
            net = cv2.dnn.readNetFromONNX(str(model_path))
            # Use CPU for compatibility
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self._models[model_name] = net
            logger.info(f"Loaded model: {model_name}")
            return net
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return None

    def _preprocess_image(self, image: Image.Image) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Preprocess image for model input."""
        # Convert to RGB numpy array
        if image.mode == "RGBA":
            # Keep alpha channel separate
            rgb = image.convert("RGB")
            alpha = np.array(image.split()[-1], dtype=np.float32) / 255.0
        else:
            rgb = image.convert("RGB")
            alpha = None

        cv_image = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
        original_size = cv_image.shape[:2][::-1]  # (width, height)

        # Normalize to [0, 1]
        cv_image = cv_image.astype(np.float32) / 255.0

        # Convert to CHW format
        cv_image = cv_image.transpose(2, 0, 1)
        cv_image = np.expand_dims(cv_image, axis=0)  # Add batch dimension

        return cv_image, original_size, alpha

    def _postprocess_image(self, output: np.ndarray, original_size: Tuple[int, int],
                           alpha: Optional[np.ndarray] = None) -> Image.Image:
        """Postprocess model output to PIL Image."""
        # Remove batch dimension and convert to HWC
        output = output.squeeze(0).transpose(1, 2, 0)
        output = np.clip(output * 255, 0, 255).astype(np.uint8)

        # Convert BGR to RGB
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

        # Resize alpha if needed
        if alpha is not None:
            if alpha.shape[:2] != output.shape[:2]:
                alpha = cv2.resize(alpha, (output.shape[1], output.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)
            alpha = (alpha * 255).astype(np.uint8)
            output = np.dstack([output, alpha])
            return Image.fromarray(output, "RGBA")

        return Image.fromarray(output, "RGB")

    def enhance(self, image: Image.Image, settings: QualityEnhanceSettings) -> EnhancementResult:
        """
        Enhance photo quality using AI model.

        Args:
            image: Input PIL Image
            settings: Quality enhancement settings

        Returns:
            EnhancementResult with enhanced image or error
        """
        import time
        start_time = time.time()

        try:
            # Load model
            net = self._load_model(settings.model)
            if net is None:
                # Fallback to traditional enhancement
                return self._traditional_enhance(image, settings, start_time)

            # Preprocess
            input_blob, original_size, alpha = self._preprocess_image(image)

            # Run inference
            net.setInput(input_blob)
            output = net.forward()

            # Postprocess
            enhanced = self._postprocess_image(output, original_size, alpha)

            # Apply additional adjustments based on settings
            enhanced = self._apply_adjustments(enhanced, settings)

            processing_time = (time.time() - start_time) * 1000
            return EnhancementResult(
                success=True,
                image=enhanced,
                processing_time_ms=processing_time
            )

        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
            # Fallback to traditional enhancement
            return self._traditional_enhance(image, settings, start_time)

    def _traditional_enhance(self, image: Image.Image, settings: QualityEnhanceSettings,
                             start_time: float) -> EnhancementResult:
        """Fallback traditional enhancement using OpenCV."""
        try:
            cv_image = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

            # Denoising
            if settings.denoise_strength > 0:
                h = int(10 * settings.denoise_strength)
                cv_image = cv2.fastNlMeansDenoisingColored(cv_image, None, h, h, 7, 21)

            # Sharpening
            if settings.sharpen_strength > 0:
                kernel = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]], dtype=np.float32)
                strength = settings.sharpen_strength
                kernel = kernel * strength + np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32) * (1 - strength)
                cv_image = cv2.filter2D(cv_image, -1, kernel)

            # Face enhancement (simple bilateral filter for skin smoothing)
            if settings.face_enhance:
                cv_image = cv2.bilateralFilter(cv_image, 9, 75, 75)

            # Convert back
            enhanced = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            if image.mode == "RGBA":
                alpha = np.array(image.split()[-1])
                enhanced = np.dstack([enhanced, alpha])
                result = Image.fromarray(enhanced, "RGBA")
            else:
                result = Image.fromarray(enhanced, "RGB")

            processing_time = (time.time() - start_time) * 1000
            return EnhancementResult(
                success=True,
                image=result,
                processing_time_ms=processing_time
            )

        except Exception as e:
            logger.error(f"Traditional enhancement failed: {e}")
            processing_time = (time.time() - start_time) * 1000
            return EnhancementResult(
                success=False,
                error=str(e),
                processing_time_ms=processing_time
            )

    def _apply_adjustments(self, image: Image.Image, settings: QualityEnhanceSettings) -> Image.Image:
        """Apply additional post-processing adjustments."""
        from PIL import ImageEnhance, ImageFilter

        result = image

        # Slight sharpening if enabled
        if settings.sharpen_strength > 0:
            enhancer = ImageEnhance.Sharpness(result)
            result = enhancer.enhance(1.0 + settings.sharpen_strength * 0.5)

        return result


def enhance_photo_quality(image: Image.Image, settings: QualityEnhanceSettings) -> EnhancementResult:
    """
    Convenience function for one-shot quality enhancement.

    Args:
        image: Input PIL Image
        settings: Quality enhancement settings

    Returns:
        EnhancementResult with enhanced image or error
    """
    enhancer = QualityEnhancer()
    return enhancer.enhance(image, settings)