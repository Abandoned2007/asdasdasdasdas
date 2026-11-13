"""
Asset management system for scanning and managing resource folders.
"""
import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Set, Callable
from dataclasses import dataclass, field
import logging

from photo_retouch.core.models import (
    AssetCategory,
    ASSET_CATEGORIES,
    SUPPORTED_IMAGE_FORMATS,
    SUPPORTED_FONT_FORMATS,
    FRAME_COSTUME_FORMATS
)

logger = logging.getLogger(__name__)


@dataclass
class AssetInfo:
    """Information about an asset file."""
    path: Path
    name: str
    category: AssetCategory
    size: int
    extension: str
    is_valid: bool = True
    validation_error: str = ""
    thumbnail: Optional[bytes] = None  # Cached thumbnail

    @property
    def relative_path(self) -> str:
        """Get path relative to project root."""
        try:
            return str(self.path.relative_to(self.path.parents[2]))
        except ValueError:
            return str(self.path)


@dataclass
class AssetScanResult:
    """Result of asset folder scanning."""
    assets: List[AssetInfo] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    scanned_folders: int = 0
    total_files: int = 0


class AssetValidator:
    """Validates asset files for correctness."""

    @staticmethod
    def validate_image(filepath: Path, require_alpha: bool = False) -> tuple[bool, str]:
        """Validate image file format and readability."""
        try:
            from PIL import Image
            with Image.open(filepath) as img:
                img.verify()
            # Re-open for further checks
            with Image.open(filepath) as img:
                if require_alpha and img.mode not in ("RGBA", "LA", "PA"):
                    return False, "Image must have alpha channel (RGBA/LA/PA)"
                if img.width == 0 or img.height == 0:
                    return False, "Image has zero dimensions"
            return True, ""
        except Exception as e:
            return False, f"Invalid image file: {str(e)}"

    @staticmethod
    def validate_font(filepath: Path) -> tuple[bool, str]:
        """Validate font file."""
        try:
            from fontTools.ttLib import TTFont
            font = TTFont(filepath)
            # Check if it has basic tables
            if "cmap" not in font:
                return False, "Font missing character map"
            return True, ""
        except Exception as e:
            return False, f"Invalid font file: {str(e)}"


class AssetManager:
    """Manages asset scanning, validation, and access."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve()
        self._assets: Dict[AssetCategory, List[AssetInfo]] = {
            cat: [] for cat in AssetCategory
        }
        self._callbacks: List[Callable[[AssetCategory], None]] = []
        self._validator = AssetValidator()

    def register_callback(self, callback: Callable[[AssetCategory], None]) -> None:
        """Register callback for asset changes."""
        self._callbacks.append(callback)

    def _notify_change(self, category: AssetCategory) -> None:
        """Notify all callbacks of asset change."""
        for callback in self._callbacks:
            try:
                callback(category)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def get_category_folder(self, category: AssetCategory) -> Path:
        """Get the folder path for a category."""
        folder_name = ASSET_CATEGORIES.get(category.value, category.value)
        return self.project_root / folder_name

    def scan_all(self) -> AssetScanResult:
        """Scan all asset folders."""
        result = AssetScanResult()
        for category in AssetCategory:
            cat_result = self.scan_category(category)
            result.assets.extend(cat_result.assets)
            result.errors.extend(cat_result.errors)
            result.scanned_folders += cat_result.scanned_folders
            result.total_files += cat_result.total_files
        return result

    def scan_category(self, category: AssetCategory) -> AssetScanResult:
        """Scan a specific asset category folder."""
        result = AssetScanResult()
        folder = self.get_category_folder(category)

        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created missing folder: {folder}")
            result.scanned_folders = 1
            self._assets[category] = []
            self._notify_change(category)
            return result

        result.scanned_folders = 1
        assets = []

        for filepath in folder.iterdir():
            if not filepath.is_file():
                continue

            result.total_files += 1
            ext = filepath.suffix.lower()

            # Determine valid formats based on category
            if category in (AssetCategory.FRAMES, AssetCategory.COSTUMES):
                valid_formats = FRAME_COSTUME_FORMATS
                require_alpha = True
            elif category == AssetCategory.FONTS:
                valid_formats = SUPPORTED_FONT_FORMATS
                require_alpha = False
            else:
                valid_formats = SUPPORTED_IMAGE_FORMATS
                require_alpha = False

            is_valid = ext in valid_formats
            validation_error = ""

            if is_valid:
                if category == AssetCategory.FONTS:
                    is_valid, validation_error = self._validator.validate_font(filepath)
                else:
                    is_valid, validation_error = self._validator.validate_image(filepath, require_alpha)

            asset = AssetInfo(
                path=filepath,
                name=filepath.stem,
                category=category,
                size=filepath.stat().st_size,
                extension=ext,
                is_valid=is_valid,
                validation_error=validation_error
            )
            assets.append(asset)

        self._assets[category] = assets
        self._notify_change(category)
        return result

    def get_assets(self, category: AssetCategory, valid_only: bool = True) -> List[AssetInfo]:
        """Get assets for a category."""
        assets = self._assets.get(category, [])
        if valid_only:
            return [a for a in assets if a.is_valid]
        return assets

    def get_asset_names(self, category: AssetCategory, valid_only: bool = True) -> List[str]:
        """Get asset names for a category."""
        return [a.name for a in self.get_assets(category, valid_only)]

    def get_asset_path(self, category: AssetCategory, name: str) -> Optional[Path]:
        """Get full path for an asset by name."""
        for asset in self.get_assets(category):
            if asset.name == name:
                return asset.path
        return None

    def add_asset(self, category: AssetCategory, source_path: Path, copy: bool = True) -> AssetInfo:
        """Add a new asset to a category folder."""
        folder = self.get_category_folder(category)
        folder.mkdir(parents=True, exist_ok=True)

        dest_path = folder / source_path.name

        # Handle name conflicts
        counter = 1
        original_stem = source_path.stem
        while dest_path.exists():
            dest_path = folder / f"{original_stem}_{counter}{source_path.suffix}"
            counter += 1

        if copy:
            shutil.copy2(source_path, dest_path)
        else:
            shutil.move(str(source_path), str(dest_path))

        # Validate and add
        ext = dest_path.suffix.lower()
        if category in (AssetCategory.FRAMES, AssetCategory.COSTUMES):
            is_valid, error = self._validator.validate_image(dest_path, require_alpha=True)
        elif category == AssetCategory.FONTS:
            is_valid, error = self._validator.validate_font(dest_path)
        else:
            is_valid, error = self._validator.validate_image(dest_path)

        asset = AssetInfo(
            path=dest_path,
            name=dest_path.stem,
            category=category,
            size=dest_path.stat().st_size,
            extension=ext,
            is_valid=is_valid,
            validation_error=error
        )

        self._assets[category].append(asset)
        self._notify_change(category)
        return asset

    def remove_asset(self, category: AssetCategory, name: str, delete_file: bool = False) -> bool:
        """Remove an asset from the category."""
        assets = self._assets[category]
        for i, asset in enumerate(assets):
            if asset.name == name:
                if delete_file and asset.path.exists():
                    asset.path.unlink()
                assets.pop(i)
                self._notify_change(category)
                return True
        return False

    def generate_thumbnail(self, asset: AssetInfo, size: tuple = (128, 128)) -> bytes:
        """Generate thumbnail for an asset."""
        try:
            from PIL import Image
            import io

            if asset.category == AssetCategory.FONTS:
                # Generate font preview
                img = Image.new("RGBA", size, (0, 0, 0, 0))
                # Could render sample text here
            else:
                with Image.open(asset.path) as img:
                    img.thumbnail(size, Image.Resampling.LANCZOS)
                    if img.mode != "RGBA":
                        img = img.convert("RGBA")

            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Thumbnail generation failed: {e}")
            # Return placeholder
            from PIL import Image
            import io
            img = Image.new("RGBA", size, (128, 128, 128, 255))
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return buffer.getvalue()


def create_default_assets(project_root: Path) -> None:
    """Create default/sample assets for testing."""
    # This would create placeholder assets if needed
    pass