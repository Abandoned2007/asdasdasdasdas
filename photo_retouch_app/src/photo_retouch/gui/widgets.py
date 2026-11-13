"""
Custom widgets for the Photo Retouch GUI.
"""
import logging
from pathlib import Path
from typing import Optional, List, Callable
from enum import Enum

from PyQt6.QtCore import (Qt, QSize, QSizeF, QRect, QPoint, QPointF, QTimer, pyqtSignal,
                          QThread, QObject, QEvent, QRectF)
from PyQt6.QtGui import (QPixmap, QImage, QPainter, QPen, QColor, QBrush,
                         QFont, QFontDatabase, QMouseEvent, QWheelEvent,
                         QKeyEvent, QDragEnterEvent, QDropEvent, QCursor,
                         QTransform, QPolygonF, QAction, QIcon)
from PyQt6.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout,
                             QSlider, QPushButton, QComboBox, QFileDialog,
                             QListWidget, QListWidgetItem, QScrollArea,
                             QFrame, QSizePolicy, QMenu, QToolButton,
                             QStyle, QStyleOptionSlider, QProgressBar,
                             QMessageBox, QInputDialog, QDialog, QFormLayout,
                             QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox,
                             QTabWidget, QSplitter, QToolBar, QStatusBar,
                             QDockWidget, QMainWindow, QApplication,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                             QGraphicsRectItem, QGraphicsTextItem,
                             QGraphicsItem, QStyleOptionGraphicsItem)

from photo_retouch.core.models import (AssetCategory, ASSET_CATEGORIES, Project,
                            Adjustments, Transform, Point, TextOverlay, TextStyle)
from photo_retouch.assets.manager import AssetManager, AssetInfo

logger = logging.getLogger(__name__)


class ImageViewer(QGraphicsView):
    """Main canvas widget for viewing and editing the composite image."""

    # Signals
    zoomChanged = pyqtSignal(float)
    panChanged = pyqtSignal(QPointF)
    itemSelected = pyqtSignal(object)
    itemMoved = pyqtSignal(object, QPointF)
    itemResized = pyqtSignal(object, QSizeF)
    costumeTransformChanged = pyqtSignal(object)  # Transform for costume
    textPositionChanged = pyqtSignal(object)  # Point for text position

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # View settings
        self.setRenderHints(QPainter.RenderHint.Antialiasing |
                            QPainter.RenderHint.SmoothPixmapTransform |
                            QPainter.RenderHint.TextAntialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # State
        self._zoom = 1.0
        self._min_zoom = 0.1
        self._max_zoom = 10.0
        self._pan_mode = False
        self._composite_item: Optional[QGraphicsPixmapItem] = None
        self._overlay_items: List[QGraphicsItem] = []
        self._costume_item: Optional[QGraphicsPixmapItem] = None
        self._text_item: Optional[QGraphicsTextItem] = None
        self._selected_item: Optional[QGraphicsItem] = None
        self._handles: List[QGraphicsRectItem] = []

        # Background pattern
        self._checkerboard = self._create_checkerboard()

        # Tool mode
        self._tool_mode = "select"  # select, pan, crop
        self._crop_rect: Optional[QGraphicsRectItem] = None
        self._crop_start: Optional[QPointF] = None

    def _create_checkerboard(self, size=20) -> QPixmap:
        """Create checkerboard pattern for transparency visualization."""
        pixmap = QPixmap(size * 2, size * 2)
        painter = QPainter(pixmap)
        light = QColor(200, 200, 200)
        dark = QColor(160, 160, 160)
        painter.fillRect(0, 0, size, size, light)
        painter.fillRect(size, 0, size, size, dark)
        painter.fillRect(0, size, size, size, dark)
        painter.fillRect(size, size, size, size, light)
        painter.end()
        return pixmap

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        """Draw checkerboard background."""
        painter.fillRect(rect, self.palette().window())
        if self._composite_item:
            # Draw checkerboard only where image is transparent
            pass
        else:
            # Fill entire background with checkerboard
            painter.drawTiledPixmap(rect, self._checkerboard)

    def set_composite_image(self, image: QImage) -> None:
        """Set the main composite image."""
        if self._composite_item:
            self._scene.removeItem(self._composite_item)

        pixmap = QPixmap.fromImage(image)
        self._composite_item = QGraphicsPixmapItem(pixmap)
        self._composite_item.setZValue(0)
        self._scene.addItem(self._composite_item)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def add_overlay_item(self, item: QGraphicsItem, z_value: float = 10) -> None:
        """Add an overlay item (text, costume, etc.)."""
        item.setZValue(z_value)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self._scene.addItem(item)
        self._overlay_items.append(item)

    def remove_overlay_item(self, item: QGraphicsItem) -> None:
        """Remove an overlay item."""
        if item in self._overlay_items:
            self._overlay_items.remove(item)
        self._scene.removeItem(item)

    def clear_overlays(self) -> None:
        """Clear all overlay items."""
        for item in self._overlay_items:
            self._scene.removeItem(item)
        self._overlay_items.clear()
        self._clear_handles()

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handle zoom with mouse wheel."""
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Zoom
            delta = event.angleDelta().y()
            factor = 1.15 if delta > 0 else 1/1.15
            self.zoom_by(factor, event.position())
        else:
            # Pan
            super().wheelEvent(event)

    def zoom_by(self, factor: float, center: QPointF = None) -> None:
        """Zoom by factor around center point."""
        new_zoom = self._zoom * factor
        new_zoom = max(self._min_zoom, min(self._max_zoom, new_zoom))
        factor = new_zoom / self._zoom

        if factor != 1.0:
            if center is None:
                center = QPointF(self.viewport().width() / 2, self.viewport().height() / 2)

            scene_center = self.mapToScene(center.toPoint())
            self.scale(factor, factor)
            self._zoom = new_zoom
            self.zoomChanged.emit(self._zoom)

            # Re-center
            new_scene_center = self.mapToScene(center.toPoint())
            delta = new_scene_center - scene_center
            self.translate(delta.x(), delta.y())

    def reset_zoom(self) -> None:
        """Reset zoom to 100%."""
        self._zoom = 1.0
        self.resetTransform()
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.zoomChanged.emit(1.0)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle key presses."""
        if event.key() == Qt.Key.Key_Space:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self._pan_mode = True
        elif event.key() == Qt.Key.Key_Delete and self._selected_item:
            self.remove_overlay_item(self._selected_item)
            self._selected_item = None
            self.itemSelected.emit(None)
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        """Handle key releases."""
        if event.key() == Qt.Key.Key_Space:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self._pan_mode = False
        else:
            super().keyReleaseEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press."""
        if event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.pos())
            if item and item in self._overlay_items:
                self._select_item(item)
            else:
                self._clear_selection()
        super().mousePressEvent(event)

    def _select_item(self, item: QGraphicsItem) -> None:
        """Select an overlay item and show handles."""
        self._clear_selection()
        self._selected_item = item
        item.setSelected(True)
        self._show_handles(item)
        self.itemSelected.emit(item)

    def _clear_selection(self) -> None:
        """Clear selection and hide handles."""
        if self._selected_item:
            self._selected_item.setSelected(False)
            self._selected_item = None
        self._clear_handles()
        self.itemSelected.emit(None)

    def _show_handles(self, item: QGraphicsItem) -> None:
        """Show resize/rotate handles around item."""
        self._clear_handles()
        rect = item.boundingRect()
        if rect.isNull():
            return

        handle_size = 10
        positions = [
            QPointF(rect.left(), rect.top()),      # Top-left
            QPointF(rect.center().x(), rect.top()), # Top-center
            QPointF(rect.right(), rect.top()),      # Top-right
            QPointF(rect.right(), rect.center().y()), # Right-center
            QPointF(rect.right(), rect.bottom()),   # Bottom-right
            QPointF(rect.center().x(), rect.bottom()), # Bottom-center
            QPointF(rect.left(), rect.bottom()),    # Bottom-left
            QPointF(rect.left(), rect.center().y()),  # Left-center
        ]

        for i, pos in enumerate(positions):
            handle = QGraphicsRectItem(-handle_size/2, -handle_size/2, handle_size, handle_size)
            handle.setBrush(QBrush(QColor(0, 120, 215)))
            handle.setPen(QPen(Qt.GlobalColor.white, 1))
            handle.setZValue(100)
            handle.setPos(item.mapToScene(pos))
            handle.setData(0, i)  # Handle index
            handle.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
            handle.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
            self._scene.addItem(handle)
            self._handles.append(handle)

    def _clear_handles(self) -> None:
        """Remove all handles."""
        for handle in self._handles:
            self._scene.removeItem(handle)
        self._handles.clear()

    def get_composite_image(self) -> QImage:
        """Get the current composite as QImage."""
        if not self._composite_item:
            return QImage()

        rect = self._scene.sceneRect().toRect()
        image = QImage(rect.size(), QImage.Format.Format_RGBA8888)
        image.fill(Qt.GlobalColor.transparent)

        painter = QPainter(image)
        self._scene.render(painter, QRectF(image.rect()), self._scene.sceneRect())
        painter.end()

        return image

    # Costume overlay methods
    def set_costume_image(self, image: QImage, transform: Transform = None) -> None:
        """Set costume overlay image with transform."""
        if self._costume_item:
            self._scene.removeItem(self._costume_item)
            self._costume_item = None

        if image is None or image.isNull():
            return

        pixmap = QPixmap.fromImage(image)
        self._costume_item = QGraphicsPixmapItem(pixmap)
        self._costume_item.setZValue(20)  # Above photo, below frame/text
        self._costume_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self._costume_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self._costume_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self._scene.addItem(self._costume_item)
        self._overlay_items.append(self._costume_item)

        # Apply transform if provided
        if transform:
            self._apply_transform_to_costume(transform)

    def _apply_transform_to_costume(self, transform: Transform) -> None:
        """Apply transform to costume item."""
        if not self._costume_item:
            return

        # Apply scale
        self._costume_item.setScale(transform.scale)

        # Apply rotation
        self._costume_item.setRotation(transform.rotation)

        # Apply position (center of item at transform position)
        self._costume_item.setPos(transform.position.x, transform.position.y)

    def get_costume_transform(self) -> Transform:
        """Get current costume transform from item."""
        if not self._costume_item:
            return Transform()

        transform = Transform()
        transform.scale = self._costume_item.scale()
        transform.rotation = self._costume_item.rotation()
        pos = self._costume_item.pos()
        transform.position = Point(pos.x(), pos.y())
        return transform

    def clear_costume(self) -> None:
        """Clear costume overlay."""
        if self._costume_item:
            self._scene.removeItem(self._costume_item)
            if self._costume_item in self._overlay_items:
                self._overlay_items.remove(self._costume_item)
            self._costume_item = None

    # Text overlay methods
    def set_text_overlay(self, text: str, position: Point, style: TextStyle, font_path: str = "") -> None:
        """Set or update text overlay with drag-to-move capability."""
        if self._text_item:
            self._scene.removeItem(self._text_item)
            if self._text_item in self._overlay_items:
                self._overlay_items.remove(self._text_item)
            self._text_item = None

        if not text:
            return

        # Load font
        try:
            if font_path and Path(font_path).exists():
                font = QFont()
                font_id = QFontDatabase.addApplicationFont(font_path)
                if font_id != -1:
                    families = QFontDatabase.applicationFontFamilies(font_id)
                    if families:
                        font.setFamily(families[0])
                    else:
                        font.setFamily("Arial")
                else:
                    font.setFamily("Arial")
            else:
                font = QFont("Arial")
        except Exception:
            font = QFont("Arial")

        font.setPointSize(style.font_size)

        # Create text item with custom itemChange handler
        self._text_item = MovableTextItem(text, self)
        self._text_item.setFont(font)
        self._text_item.setDefaultTextColor(QColor(style.color))
        self._text_item.setZValue(30)  # Above costume, below frame
        self._text_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self._text_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self._text_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)

        # Add stroke effect using HTML if enabled
        if style.stroke_enabled:
            # Use HTML to create stroke effect
            html = f'<span style="color: {style.color}; -webkit-text-stroke: {style.stroke_width}px {style.stroke_color};">{text}</span>'
            self._text_item.setHtml(html)

        # Set position (center of text at position point)
        self._text_item.setPos(position.x, position.y)
        self._scene.addItem(self._text_item)
        self._overlay_items.append(self._text_item)

    def clear_text_overlay(self) -> None:
        """Clear text overlay."""
        if self._text_item:
            self._scene.removeItem(self._text_item)
            if self._text_item in self._overlay_items:
                self._overlay_items.remove(self._text_item)
            self._text_item = None

    def get_text_position(self) -> Point:
        """Get current text position."""
        if self._text_item:
            pos = self._text_item.pos()
            return Point(pos.x(), pos.y())
        return Point()

    def _on_text_position_changed(self, position: Point) -> None:
        """Handle text position change from MovableTextItem."""
        self.textPositionChanged.emit(position)


class MovableTextItem(QGraphicsTextItem):
    """QGraphicsTextItem that emits signal on position change."""

    def __init__(self, text: str, viewer: 'ImageViewer'):
        super().__init__(text)
        self._viewer = viewer

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            pos = value
            if hasattr(pos, 'x') and hasattr(pos, 'y'):
                self._viewer._on_text_position_changed(Point(pos.x(), pos.y()))
        return super().itemChange(change, value)

    # Crop tool methods
    def set_tool_mode(self, mode: str) -> None:
        """Set current tool mode: select, pan, crop."""
        self._tool_mode = mode
        if mode == "pan":
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        elif mode == "select":
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        elif mode == "crop":
            self.setDragMode(QGraphicsView.DragMode.NoDrag)

    def start_crop(self, pos: QPointF) -> None:
        """Start crop rectangle."""
        if self._tool_mode != "crop":
            return
        self._crop_start = self.mapToScene(pos.toPoint())
        self._crop_rect = QGraphicsRectItem()
        self._crop_rect.setPen(QPen(QColor(0, 120, 215), 2, Qt.PenStyle.DashLine))
        self._crop_rect.setZValue(50)
        self._scene.addItem(self._crop_rect)

    def update_crop(self, pos: QPointF) -> None:
        """Update crop rectangle."""
        if not self._crop_rect or not self._crop_start:
            return
        current = self.mapToScene(pos.toPoint())
        rect = QRectF(self._crop_start, current).normalized()
        self._crop_rect.setRect(rect)

    def finish_crop(self) -> QRectF:
        """Finish crop and return rectangle in scene coordinates."""
        if not self._crop_rect or not self._crop_start:
            return QRectF()
        rect = self._crop_rect.rect()
        self._scene.removeItem(self._crop_rect)
        self._crop_rect = None
        self._crop_start = None
        self.set_tool_mode("select")
        return rect

    def cancel_crop(self) -> None:
        """Cancel crop operation."""
        if self._crop_rect:
            self._scene.removeItem(self._crop_rect)
            self._crop_rect = None
        self._crop_start = None
        self.set_tool_mode("select")


class AssetGallery(QWidget):
    """Gallery widget for browsing assets (frames, backgrounds, costumes)."""

    assetSelected = pyqtSignal(str, AssetCategory)  # name, category
    assetDoubleClicked = pyqtSignal(str, AssetCategory)

    def __init__(self, category: AssetCategory, parent=None):
        super().__init__(parent)
        self.category = category
        self._asset_manager: Optional[AssetManager] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header
        header = QHBoxLayout()
        label = QLabel(self.category.value.replace("_", " ").title())
        label.setStyleSheet("font-weight: bold; font-size: 12px;")
        header.addWidget(label)
        header.addStretch()

        refresh_btn = QToolButton()
        refresh_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        refresh_btn.setToolTip("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        header.addWidget(refresh_btn)

        layout.addLayout(header)

        # List widget with icon view
        self.list_widget = QListWidget()
        self.list_widget.setViewMode(QListWidget.ViewMode.IconMode)
        self.list_widget.setIconSize(QSize(96, 96))
        self.list_widget.setGridSize(QSize(110, 130))
        self.list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list_widget.setMovement(QListWidget.Movement.Static)
        self.list_widget.setSpacing(4)
        self.list_widget.setUniformItemSizes(True)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._show_context_menu)

        layout.addWidget(self.list_widget)

    def set_asset_manager(self, manager: AssetManager) -> None:
        """Set the asset manager and refresh."""
        self._asset_manager = manager
        if manager:
            manager.register_callback(self._on_assets_changed)
        self.refresh()

    def _on_assets_changed(self, category: AssetCategory) -> None:
        if category == self.category:
            self.refresh()

    def refresh(self) -> None:
        """Refresh the gallery from asset manager."""
        if not self._asset_manager:
            return

        self.list_widget.clear()
        assets = self._asset_manager.get_assets(self.category, valid_only=True)

        for asset in assets:
            item = QListWidgetItem(asset.name)
            item.setToolTip(f"{asset.name}{asset.extension}\n{asset.size // 1024} KB")
            item.setData(Qt.ItemDataRole.UserRole, asset.name)

            # Generate thumbnail
            try:
                thumb_data = self._asset_manager.generate_thumbnail(asset, (96, 96))
                pixmap = QPixmap()
                pixmap.loadFromData(thumb_data)
                item.setIcon(QIcon(pixmap))
            except Exception as e:
                logger.warning(f"Failed to generate thumbnail for {asset.name}: {e}")
                # Use default icon
                item.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon))

            self.list_widget.addItem(item)

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        name = item.data(Qt.ItemDataRole.UserRole)
        self.assetSelected.emit(name, self.category)

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        name = item.data(Qt.ItemDataRole.UserRole)
        self.assetDoubleClicked.emit(name, self.category)

    def _show_context_menu(self, pos) -> None:
        item = self.list_widget.itemAt(pos)
        if not item:
            return

        menu = QMenu(self)
        name = item.data(Qt.ItemDataRole.UserRole)

        select_action = menu.addAction("Select")
        select_action.triggered.connect(lambda: self.assetSelected.emit(name, self.category))

        add_action = menu.addAction("Add to Project")
        add_action.triggered.connect(lambda: self.assetDoubleClicked.emit(name, self.category))

        if self._asset_manager:
            remove_action = menu.addAction("Remove from Library")
            remove_action.triggered.connect(lambda: self._remove_asset(name))

        menu.exec(self.list_widget.mapToGlobal(pos))

    def _remove_asset(self, name: str) -> None:
        if self._asset_manager:
            reply = QMessageBox.question(
                self, "Remove Asset",
                f"Remove '{name}' from library?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._asset_manager.remove_asset(self.category, name, delete_file=True)
                self.refresh()


class AdjustmentSlider(QWidget):
    """Custom slider widget for image adjustments."""

    valueChanged = pyqtSignal(int)

    def __init__(self, label: str, minimum: int, maximum: int, default: int = 0,
                 suffix: str = "", parent=None):
        super().__init__(parent)
        self.minimum = minimum
        self.maximum = maximum
        self._setup_ui(label, suffix)
        self.set_value(default)

    def _setup_ui(self, label: str, suffix: str) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.label = QLabel(label)
        self.label.setMinimumWidth(80)
        layout.addWidget(self.label)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(self.minimum, self.maximum)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(10)
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider)

        self.value_label = QLabel()
        self.value_label.setMinimumWidth(50)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.suffix = suffix
        layout.addWidget(self.value_label)

    def _on_slider_changed(self, value: int) -> None:
        self.value_label.setText(f"{value}{self.suffix}")
        self.valueChanged.emit(value)

    def value(self) -> int:
        return self.slider.value()

    def set_value(self, value: int) -> None:
        self.slider.setValue(value)

    def setEnabled(self, enabled: bool) -> None:
        self.slider.setEnabled(enabled)
        self.label.setEnabled(enabled)
        self.value_label.setEnabled(enabled)


class LayerControlWidget(QWidget):
    """Widget for controlling layer visibility and order."""

    layerVisibilityChanged = pyqtSignal(str, bool)
    layerOrderChanged = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layers = ["background", "photo", "costume", "frame", "text"]
        self._layer_labels = {
            "background": "Фон",
            "photo": "Фото",
            "costume": "Костюм",
            "frame": "Рамка",
            "text": "Текст"
        }
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Layer list with checkboxes
        self.layer_widgets = {}
        for layer_id in self._layers:
            widget = QWidget()
            hlayout = QHBoxLayout(widget)
            hlayout.setContentsMargins(0, 0, 0, 0)

            checkbox = QCheckBox(self._layer_labels[layer_id])
            checkbox.setChecked(True)
            checkbox.toggled.connect(lambda checked, lid=layer_id:
                                     self.layerVisibilityChanged.emit(lid, checked))
            hlayout.addWidget(checkbox)

            # Up/Down buttons for reordering
            up_btn = QToolButton()
            up_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowUp))
            up_btn.setFixedSize(24, 24)
            up_btn.clicked.connect(lambda _, lid=layer_id: self._move_layer(lid, -1))
            hlayout.addWidget(up_btn)

            down_btn = QToolButton()
            down_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowDown))
            down_btn.setFixedSize(24, 24)
            down_btn.clicked.connect(lambda _, lid=layer_id: self._move_layer(lid, 1))
            hlayout.addWidget(down_btn)

            hlayout.addStretch()
            layout.addWidget(widget)

            self.layer_widgets[layer_id] = {"checkbox": checkbox, "widget": widget}

        layout.addStretch()

    def _move_layer(self, layer_id: str, direction: int) -> None:
        """Move layer up or down in the stack."""
        idx = self._layers.index(layer_id)
        new_idx = idx + direction
        if 0 <= new_idx < len(self._layers):
            self._layers[idx], self._layers[new_idx] = self._layers[new_idx], self._layers[idx]
            self._rebuild_layout()
            self.layerOrderChanged.emit(self._layers.copy())

    def _rebuild_layout(self) -> None:
        """Rebuild the layout in new order."""
        layout = self.layout()
        # Remove all widgets
        for layer_id in self._layers:
            widget = self.layer_widgets[layer_id]["widget"]
            layout.removeWidget(widget)
        # Re-add in new order
        for layer_id in self._layers:
            widget = self.layer_widgets[layer_id]["widget"]
            layout.insertWidget(layout.count() - 1, widget)  # Before stretch

    def set_visibility(self, layer_id: str, visible: bool) -> None:
        """Set layer visibility."""
        if layer_id in self.layer_widgets:
            self.layer_widgets[layer_id]["checkbox"].setChecked(visible)

    def get_order(self) -> List[str]:
        return self._layers.copy()


class TextEditDialog(QDialog):
    """Dialog for editing text overlay."""

    def __init__(self, overlay: TextOverlay, font_path: str = "", parent=None):
        super().__init__(parent)
        self.overlay = overlay
        self.font_path = font_path
        self.setWindowTitle("Редактировать текст")
        self.setModal(True)
        self.resize(400, 500)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Text content
        form = QFormLayout()
        self.text_edit = QInputDialog.getMultiLineText
        self.text_content = QTextEdit()
        self.text_content.setPlainText(self.overlay.content)
        self.text_content.setMaximumHeight(100)
        form.addRow("Текст:", self.text_content)

        # Font size
        self.font_size = QSpinBox()
        self.font_size.setRange(8, 200)
        self.font_size.setValue(self.overlay.style.font_size)
        form.addRow("Размер шрифта:", self.font_size)

        # Color
        self.color_btn = QPushButton()
        self.color_btn.setFixedSize(60, 30)
        self.color_btn.setStyleSheet(f"background-color: {self.overlay.style.color}")
        self.color_btn.clicked.connect(self._pick_color)
        form.addRow("Цвет:", self.color_btn)

        # Stroke
        self.stroke_enabled = QCheckBox("Включить контур")
        self.stroke_enabled.setChecked(self.overlay.style.stroke_enabled)
        form.addRow(self.stroke_enabled)

        self.stroke_width = QSpinBox()
        self.stroke_width.setRange(1, 20)
        self.stroke_width.setValue(self.overlay.style.stroke_width)
        form.addRow("Толщина контура:", self.stroke_width)

        self.stroke_color_btn = QPushButton()
        self.stroke_color_btn.setFixedSize(60, 30)
        self.stroke_color_btn.setStyleSheet(f"background-color: {self.overlay.style.stroke_color}")
        self.stroke_color_btn.clicked.connect(self._pick_stroke_color)
        form.addRow("Цвет контура:", self.stroke_color_btn)

        # Shadow
        self.shadow_enabled = QCheckBox("Включить тень")
        self.shadow_enabled.setChecked(self.overlay.style.shadow_enabled)
        form.addRow(self.shadow_enabled)

        self.shadow_blur = QSpinBox()
        self.shadow_blur.setRange(0, 50)
        self.shadow_blur.setValue(self.overlay.style.shadow_blur)
        form.addRow("Размытие тени:", self.shadow_blur)

        layout.addLayout(form)

        # Buttons
        buttons = QHBoxLayout()
        buttons.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Отмена")
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(ok_btn)
        buttons.addWidget(cancel_btn)
        layout.addLayout(buttons)

    def _pick_color(self) -> None:
        from PyQt6.QtWidgets import QColorDialog
        color = QColorDialog.getColor(QColor(self.overlay.style.color), self)
        if color.isValid():
            self.overlay.style.color = color.name()
            self.color_btn.setStyleSheet(f"background-color: {color.name()}")

    def _pick_stroke_color(self) -> None:
        from PyQt6.QtWidgets import QColorDialog
        color = QColorDialog.getColor(QColor(self.overlay.style.stroke_color), self)
        if color.isValid():
            self.overlay.style.stroke_color = color.name()
            self.stroke_color_btn.setStyleSheet(f"background-color: {color.name()}")

    def get_result(self) -> TextOverlay:
        """Get the updated overlay."""
        self.overlay.content = self.text_content.toPlainText()
        self.overlay.style.font_size = self.font_size.value()
        self.overlay.style.stroke_enabled = self.stroke_enabled.isChecked()
        self.overlay.style.stroke_width = self.stroke_width.value()
        self.overlay.style.shadow_enabled = self.shadow_enabled.isChecked()
        self.overlay.style.shadow_blur = self.shadow_blur.value()
        return self.overlay


class ProgressDialog(QDialog):
    """Progress dialog for long operations."""

    def __init__(self, title: str, message: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setFixedSize(400, 120)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)

        layout = QVBoxLayout(self)

        self.label = QLabel(message)
        layout.addWidget(self.label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)

        self.detail_label = QLabel("")
        layout.addWidget(self.detail_label)

        self.cancel_btn = QPushButton("Отмена")
        self.cancel_btn.clicked.connect(self.reject)
        layout.addWidget(self.cancel_btn)

    def set_progress(self, value: int, detail: str = "") -> None:
        self.progress.setValue(value)
        if detail:
            self.detail_label.setText(detail)

    def closeEvent(self, event) -> None:
        self.reject()
        event.accept()


# Need to import QTextEdit
from PyQt6.QtWidgets import QTextEdit