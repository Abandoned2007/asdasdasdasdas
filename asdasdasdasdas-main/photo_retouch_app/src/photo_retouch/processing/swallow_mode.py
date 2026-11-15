"""Swallow mode: AI-powered beard, wart, and imperfection removal with smooth skin transitions."""
import logging
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageFilter

from photo_retouch.core.models import SwallowModeSettings

logger = logging.getLogger(__name__)


@dataclass
class SwallowResult:
    """Result of swallow mode processing."""
    success: bool
    image: Optional[Image.Image] = None
    mask: Optional[np.ndarray] = None  # Debug: shows detected regions
    error: str = ""
    processing_time_ms: float = 0.0


class SwallowProcessor:
    """Processes photos to remove beard, warts, and skin imperfections with smooth transitions."""

    def __init__(self):
        self._face_cascade = None
        self._load_detectors()

    def _load_detectors(self):
        """Load Haar cascades for face and facial feature detection."""
        try:
            # Load OpenCV's built-in Haar cascades
            self._face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self._eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            self._nose_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_mcs_nose.xml'
            )
            self._mouth_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_mcs_mouth.xml'
            )
        except Exception as e:
            logger.warning(f"Could not load all Haar cascades: {e}")

    def _detect_face_regions(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect face regions in the image."""
        if self._face_cascade is None:
            return []

        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        return [(x, y, w, h) for (x, y, w, h) in faces]

    def _detect_beard_region(self, gray: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Detect beard region (lower face area below nose/mouth).
        Returns binary mask of beard area.
        """
        x, y, w, h = face
        face_roi = gray[y:y+h, x:x+w]

        # Beard is typically in lower 40% of face
        beard_start = int(h * 0.55)
        beard_region = face_roi[beard_start:, :]

        # Use adaptive threshold to detect darker hair-like regions
        beard_mask = np.zeros_like(face_roi, dtype=np.uint8)

        # Detect horizontal edges (hair texture)
        sobel_x = cv2.Sobel(beard_region, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(beard_region, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)

        # Threshold for hair-like textures
        _, hair_mask = cv2.threshold(
            gradient_mag.astype(np.uint8),
            0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Also use color-based detection for darker regions
        _, dark_mask = cv2.threshold(beard_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Combine masks
        combined = cv2.bitwise_or(hair_mask, dark_mask)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

        # Place back in face ROI
        beard_mask[beard_start:, :] = combined

        return beard_mask

    def _detect_warts_moles(self, gray: np.ndarray, face: Tuple[int, int, int, int],
                            sensitivity: float) -> np.ndarray:
        """
        Detect warts, moles, and raised skin imperfections.
        Returns binary mask of detected imperfections.
        """
        x, y, w, h = face
        face_roi = gray[y:y+h, x:x+w]

        # Use top-hat transform to find bright spots (moles/warts)
        kernel_size = max(5, int(min(w, h) * 0.03))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Top-hat: original - opening (finds bright spots smaller than kernel)
        tophat = cv2.morphologyEx(face_roi, cv2.MORPH_TOPHAT, kernel)

        # Threshold based on sensitivity
        threshold_val = int(255 * (0.15 + 0.35 * (1 - sensitivity)))
        _, spots_mask = cv2.threshold(tophat, threshold_val, 255, cv2.THRESH_BINARY)

        # Filter by size and circularity
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(spots_mask, 8)
        filtered_mask = np.zeros_like(spots_mask)

        min_area = max(3, int(min(w, h) * 0.002))
        max_area = int(min(w, h) * 0.02)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if min_area <= area <= max_area:
                # Check circularity
                w_comp = stats[i, cv2.CC_STAT_WIDTH]
                h_comp = stats[i, cv2.CC_STAT_HEIGHT]
                aspect_ratio = w_comp / max(1, h_comp)
                if 0.5 <= aspect_ratio <= 2.0:  # Roughly circular
                    filtered_mask[labels == i] = 255

        return filtered_mask

    def _detect_acne(self, gray: np.ndarray, face: Tuple[int, int, int, int],
                     sensitivity: float) -> np.ndarray:
        """
        Detect acne/pimples (small red/inflammatory spots).
        Returns binary mask of detected acne.
        """
        x, y, w, h = face
        face_roi = gray[y:y+h, x:x+w]

        # Use Laplacian to find high-frequency details (small bumps)
        laplacian = cv2.Laplacian(face_roi, cv2.CV_64F, ksize=3)
        laplacian = np.uint8(np.absolute(laplacian))

        # Threshold for high-frequency details
        threshold_val = int(255 * (0.1 + 0.3 * (1 - sensitivity)))
        _, acne_mask = cv2.threshold(laplacian, threshold_val, 255, cv2.THRESH_BINARY)

        # Filter very small components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(acne_mask, 8)
        filtered_mask = np.zeros_like(acne_mask)

        max_area = int(min(w, h) * 0.005)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if 2 <= area <= max_area:
                filtered_mask[labels == i] = 255

        return filtered_mask

    def _create_smooth_transition_mask(self, mask: np.ndarray,
                                       transition_strength: float) -> np.ndarray:
        """
        Create a feathered mask for smooth transitions between processed and unprocessed areas.
        """
        if transition_strength <= 0:
            return mask.astype(np.float32) / 255.0

        # Distance transform for feathering
        dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)

        # Normalize distance to transition width
        transition_width = max(5, int(30 * transition_strength))
        feathered = np.clip(dist / transition_width, 0, 1)

        # Combine with original mask
        mask_float = mask.astype(np.float32) / 255.0
        result = np.maximum(mask_float, feathered * (1 - mask_float))

        return result

    def _preserve_skin_texture(self, original: np.ndarray, processed: np.ndarray,
                               texture_mask: np.ndarray, preserve_strength: float) -> np.ndarray:
        """
        Blend original skin texture back into processed areas.
        """
        if preserve_strength <= 0:
            return processed

        # Extract high-frequency texture from original
        blur = cv2.bilateralFilter(original, 9, 75, 75)
        texture = cv2.subtract(original, blur)
        texture = cv2.add(texture, 128)  # Center at 128

        # Apply texture only where mask indicates
        texture_mask_3ch = cv2.merge([texture_mask, texture_mask, texture_mask])
        texture = cv2.multiply(texture.astype(np.float32) / 255.0,
                               texture_mask_3ch * preserve_strength)
        texture = (texture * 255).astype(np.uint8)
        texture = cv2.subtract(texture, 128 * preserve_strength)

        # Add texture back
        result = cv2.add(processed.astype(np.int16), texture.astype(np.int16))
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def process(self, image: Image.Image, settings: SwallowModeSettings) -> SwallowResult:
        """
        Process image with swallow mode settings.

        Args:
            image: Input PIL Image (RGBA or RGB)
            settings: Swallow mode settings

        Returns:
            SwallowResult with processed image
        """
        import time
        start_time = time.time()

        try:
            if not settings.enabled:
                return SwallowResult(
                    success=True,
                    image=image,
                    processing_time_ms=(time.time() - start_time) * 1000
                )

            # Convert to OpenCV format
            has_alpha = image.mode == "RGBA"
            if has_alpha:
                alpha = np.array(image.split()[-1])
                rgb = image.convert("RGB")
            else:
                rgb = image
                alpha = None

            cv_image = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # Detect faces
            faces = self._detect_face_regions(gray)
            if not faces:
                logger.warning("No faces detected for swallow mode")
                return SwallowResult(
                    success=True,
                    image=image,
                    processing_time_ms=(time.time() - start_time) * 1000
                )

            # Combined mask for all imperfections
            combined_mask = np.zeros((h, w), dtype=np.uint8)

            for face in faces:
                x, y, fw, fh = face

                # Detect beard
                if settings.remove_beard:
                    beard_mask = self._detect_beard_region(gray, face)
                    # Place in full image coordinates
                    full_beard_mask = np.zeros((h, w), dtype=np.uint8)
                    full_beard_mask[y:y+fh, x:x+fw] = beard_mask
                    combined_mask = cv2.bitwise_or(combined_mask, full_beard_mask)

                # Detect warts/moles
                if settings.remove_warts or settings.remove_moles:
                    spots_mask = self._detect_warts_moles(gray, face, settings.detection_sensitivity)
                    full_spots_mask = np.zeros((h, w), dtype=np.uint8)
                    full_spots_mask[y:y+fh, x:x+fw] = spots_mask
                    combined_mask = cv2.bitwise_or(combined_mask, full_spots_mask)

                # Detect acne
                if settings.remove_acne:
                    acne_mask = self._detect_acne(gray, face, settings.detection_sensitivity)
                    full_acne_mask = np.zeros((h, w), dtype=np.uint8)
                    full_acne_mask[y:y+fh, x:x+fw] = acne_mask
                    combined_mask = cv2.bitwise_or(combined_mask, full_acne_mask)

            # Create smooth transition mask
            transition_mask = self._create_smooth_transition_mask(
                combined_mask, settings.smooth_transition
            )

            # Inpaint detected regions
            inpaint_radius = max(3, int(min(w, h) * 0.01))
            processed = cv2.inpaint(cv_image, combined_mask, inpaint_radius, cv2.INPAINT_TELEA)

            # Preserve skin texture in processed areas
            if settings.preserve_skin_texture > 0:
                processed = self._preserve_skin_texture(
                    cv_image, processed, transition_mask, settings.preserve_skin_texture
                )

            # Apply smooth blending at edges using the transition mask
            # Blend original and processed based on transition mask
            transition_3ch = cv2.merge([transition_mask, transition_mask, transition_mask])
            original_float = cv_image.astype(np.float32)
            processed_float = processed.astype(np.float32)

            blended = original_float * (1 - transition_3ch) + processed_float * transition_3ch
            blended = np.clip(blended, 0, 255).astype(np.uint8)

            # Convert back to PIL
            if has_alpha:
                result_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
                result_rgba = np.dstack([result_rgb, alpha])
                result = Image.fromarray(result_rgba, "RGBA")
            else:
                result_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
                result = Image.fromarray(result_rgb, "RGB")

            processing_time = (time.time() - start_time) * 1000
            return SwallowResult(
                success=True,
                image=result,
                mask=combined_mask,
                processing_time_ms=processing_time
            )

        except Exception as e:
            logger.error(f"Swallow mode processing failed: {e}")
            processing_time = (time.time() - start_time) * 1000
            return SwallowResult(
                success=False,
                error=str(e),
                processing_time_ms=processing_time
            )


def apply_swallow_mode(image: Image.Image, settings: SwallowModeSettings) -> SwallowResult:
    """
    Convenience function for one-shot swallow mode processing.

    Args:
        image: Input PIL Image
        settings: Swallow mode settings

    Returns:
        SwallowResult with processed image or error
    """
    processor = SwallowProcessor()
    return processor.process(image, settings)