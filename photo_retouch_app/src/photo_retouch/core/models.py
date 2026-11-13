"""
Core data models for the Photo Retouch application.
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from enum import Enum
import json
import uuid


class AssetCategory(Enum):
    """Asset category enumeration."""
    INITIAL_PHOTO = "initial_photo"
    RESTORED_PHOTO = "restored_photo"
    FRAMES = "frames"
    BACKGROUNDS = "backgrounds"
    FONTS = "fonts"
    COSTUMES = "costumes"
    FINAL_RETOUCH = "final_retouch"


@dataclass
class Point:
    """2D point with x, y coordinates."""
    x: float = 0.0
    y: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "Point":
        return cls(x=data.get("x", 0.0), y=data.get("y", 0.0))


@dataclass
class Transform:
    """Transform properties for positioning, scaling, rotation."""
    position: Point = field(default_factory=Point)
    scale: float = 1.0
    rotation: float = 0.0  # degrees

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": self.position.to_dict(),
            "scale": self.scale,
            "rotation": self.rotation
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Transform":
        return cls(
            position=Point.from_dict(data.get("position", {})),
            scale=data.get("scale", 1.0),
            rotation=data.get("rotation", 0.0)
        )


@dataclass
class TextStyle:
    """Text styling properties."""
    font_size: int = 48
    color: str = "#FFFFFF"
    stroke_enabled: bool = True
    stroke_width: int = 2
    stroke_color: str = "#000000"
    shadow_enabled: bool = False
    shadow_offset: Point = field(default_factory=lambda: Point(2, 2))
    shadow_color: str = "#000000"
    shadow_blur: int = 4

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fontSize": self.font_size,
            "color": self.color,
            "stroke": {
                "enabled": self.stroke_enabled,
                "width": self.stroke_width,
                "color": self.stroke_color
            },
            "shadow": {
                "enabled": self.shadow_enabled,
                "offset": self.shadow_offset.to_dict(),
                "color": self.shadow_color,
                "blur": self.shadow_blur
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextStyle":
        stroke = data.get("stroke", {})
        shadow = data.get("shadow", {})
        return cls(
            font_size=data.get("fontSize", 48),
            color=data.get("color", "#FFFFFF"),
            stroke_enabled=stroke.get("enabled", True),
            stroke_width=stroke.get("width", 2),
            stroke_color=stroke.get("color", "#000000"),
            shadow_enabled=shadow.get("enabled", False),
            shadow_offset=Point.from_dict(shadow.get("offset", {"x": 2, "y": 2})),
            shadow_color=shadow.get("color", "#000000"),
            shadow_blur=shadow.get("blur", 4)
        )


@dataclass
class Adjustments:
    """Image adjustment parameters."""
    brightness: int = 0      # -100 to 100
    contrast: int = 0        # -100 to 100
    sharpness: int = 0       # 0 to 100
    saturation: int = 0      # -100 to 100
    temperature: int = 0     # -100 to 100 (warm/cool)
    vignette: int = 0        # 0 to 100

    def to_dict(self) -> Dict[str, int]:
        return {
            "brightness": self.brightness,
            "contrast": self.contrast,
            "sharpness": self.sharpness,
            "saturation": self.saturation,
            "temperature": self.temperature,
            "vignette": self.vignette
        }

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "Adjustments":
        return cls(
            brightness=data.get("brightness", 0),
            contrast=data.get("contrast", 0),
            sharpness=data.get("sharpness", 0),
            saturation=data.get("saturation", 0),
            temperature=data.get("temperature", 0),
            vignette=data.get("vignette", 0)
        )

    def clamp(self) -> "Adjustments":
        """Return clamped adjustments within valid ranges."""
        return Adjustments(
            brightness=max(-100, min(100, self.brightness)),
            contrast=max(-100, min(100, self.contrast)),
            sharpness=max(0, min(100, self.sharpness)),
            saturation=max(-100, min(100, self.saturation)),
            temperature=max(-100, min(100, self.temperature)),
            vignette=max(0, min(100, self.vignette))
        )


@dataclass
class MLSettings:
    """Machine Learning model settings for photo restoration."""
    model: str = "GFPGANv1.4"  # GFPGANv1.4, CodeFormer, Real-ESRGAN
    fidelity: float = 0.75     # 0.0 to 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "fidelity": self.fidelity
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MLSettings":
        return cls(
            model=data.get("model", "GFPGANv1.4"),
            fidelity=max(0.0, min(1.0, data.get("fidelity", 0.75)))
        )


@dataclass
class TextOverlay:
    """Text overlay on the canvas."""
    content: str = ""
    position: Point = field(default_factory=Point)
    style: TextStyle = field(default_factory=TextStyle)
    font_path: str = ""
    visible: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "position": self.position.to_dict(),
            "style": self.style.to_dict(),
            "fontPath": self.font_path,
            "visible": self.visible
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextOverlay":
        return cls(
            content=data.get("content", ""),
            position=Point.from_dict(data.get("position", {})),
            style=TextStyle.from_dict(data.get("style", {})),
            font_path=data.get("fontPath", ""),
            visible=data.get("visible", True)
        )


@dataclass
class CostumeItem:
    """Costume/clothing overlay item."""
    asset_path: str = ""
    transform: Transform = field(default_factory=Transform)
    visible: bool = True
    opacity: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assetPath": self.asset_path,
            "transform": self.transform.to_dict(),
            "visible": self.visible,
            "opacity": self.opacity
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CostumeItem":
        return cls(
            asset_path=data.get("assetPath", ""),
            transform=Transform.from_dict(data.get("transform", {})),
            visible=data.get("visible", True),
            opacity=data.get("opacity", 1.0)
        )


@dataclass
class Project:
    """Main project data structure."""
    version: str = "1.0"
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    modified: str = field(default_factory=lambda: datetime.now().isoformat())

    # Asset paths (relative to project root)
    initial_photo: str = ""
    restored_photo: str = ""
    background: str = ""
    frame: str = ""
    font: str = ""
    costume: str = ""

    # Adjustments and settings
    adjustments: Adjustments = field(default_factory=Adjustments)
    costume_transform: Transform = field(default_factory=Transform)
    text_overlay: TextOverlay = field(default_factory=TextOverlay)
    ml_settings: MLSettings = field(default_factory=MLSettings)

    # Canvas state
    canvas_zoom: float = 1.0
    canvas_pan: Point = field(default_factory=Point)

    # Layer visibility and order
    layer_visibility: Dict[str, bool] = field(default_factory=lambda: {
        "background": True,
        "photo": True,
        "costume": True,
        "frame": True,
        "text": True
    })

    def to_dict(self) -> Dict[str, Any]:
        """Serialize project to dictionary."""
        self.modified = datetime.now().isoformat()
        return {
            "version": self.version,
            "id": self.id,
            "created": self.created,
            "modified": self.modified,
            "initialPhoto": self.initial_photo,
            "restoredPhoto": self.restored_photo,
            "background": self.background,
            "frame": self.frame,
            "font": self.font,
            "costume": self.costume,
            "adjustments": self.adjustments.to_dict(),
            "costumeTransform": self.costume_transform.to_dict(),
            "textOverlay": self.text_overlay.to_dict(),
            "mlSettings": self.ml_settings.to_dict(),
            "canvasZoom": self.canvas_zoom,
            "canvasPan": self.canvas_pan.to_dict(),
            "layerVisibility": self.layer_visibility
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Project":
        """Deserialize project from dictionary."""
        project = cls(
            version=data.get("version", "1.0"),
            id=data.get("id", str(uuid.uuid4())[:8]),
            created=data.get("created", datetime.now().isoformat()),
            modified=data.get("modified", datetime.now().isoformat()),
            initial_photo=data.get("initialPhoto", ""),
            restored_photo=data.get("restoredPhoto", ""),
            background=data.get("background", ""),
            frame=data.get("frame", ""),
            font=data.get("font", ""),
            costume=data.get("costume", ""),
            adjustments=Adjustments.from_dict(data.get("adjustments", {})),
            costume_transform=Transform.from_dict(data.get("costumeTransform", {})),
            text_overlay=TextOverlay.from_dict(data.get("textOverlay", {})),
            ml_settings=MLSettings.from_dict(data.get("mlSettings", {})),
            canvas_zoom=data.get("canvasZoom", 1.0),
            canvas_pan=Point.from_dict(data.get("canvasPan", {})),
            layer_visibility=data.get("layerVisibility", {
                "background": True, "photo": True, "costume": True,
                "frame": True, "text": True
            })
        )
        return project

    def to_json(self, indent: int = 2) -> str:
        """Serialize project to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "Project":
        """Deserialize project from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def save(self, filepath: Path) -> None:
        """Save project to file."""
        filepath.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, filepath: Path) -> "Project":
        """Load project from file."""
        return cls.from_json(filepath.read_text(encoding="utf-8"))


# Asset categories mapping
ASSET_CATEGORIES = {
    "initial_photo": "01_InitialPhoto",
    "restored_photo": "02_RestoredPhoto",
    "frames": "03_Frames",
    "backgrounds": "04_Backgrounds",
    "fonts": "05_Fonts",
    "costumes": "06_Costumes",
    "final_retouch": "07_FinalRetouch"
}

SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
SUPPORTED_FONT_FORMATS = {".ttf", ".otf"}
FRAME_COSTUME_FORMATS = {".png"}  # Must have alpha channel