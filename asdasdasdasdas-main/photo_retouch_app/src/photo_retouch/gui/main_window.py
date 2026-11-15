"""
Main application window for Photo Retouch.
"""
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime
from copy import deepcopy

from PyQt6.QtCore import (Qt, QSize, QTimer, QSettings, QThread, pyqtSignal,
                          QPoint, QRect)
from PyQt6.QtGui import (QIcon, QAction, QKeySequence, QImage, QPixmap,
                         QFont, QFontDatabase, QColor, QDesktopServices, QShortcut)
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QSplitter, QTabWidget, QToolBar, QStatusBar,
                             QDockWidget, QFileDialog, QMessageBox, QMenu,
                             QProgressBar, QLabel, QApplication, QStyle,
                             QInputDialog, QProgressDialog, QPushButton, QTextEdit,
                             QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
                             QGroupBox, QFormLayout, QDialog)

from PIL import ImageOps

from photo_retouch.core.models import (Project, AssetCategory, ASSET_CATEGORIES,
                            Adjustments, Transform, Point, TextOverlay, MLSettings,
                            QualityEnhanceSettings, SwallowModeSettings)
from photo_retouch.assets.manager import AssetManager
from photo_retouch.processing.engine import (Compositor, ImageProcessor, TextRenderer,
                                  load_image_safe, save_image, export_bmp_8bit)
from photo_retouch.processing.quality_enhance import QualityEnhancer, QualityEnhanceSettings as QESettings
from photo_retouch.processing.swallow_mode import SwallowProcessor, SwallowModeSettings as SMSettings
from photo_retouch.ml.pipeline import MLPipeline, RestorationTask, RestorationModel, MockMLPipeline
from photo_retouch.gui.widgets import (ImageViewer, AssetGallery, AdjustmentSlider,
                      LayerControlWidget, TextEditDialog, ProgressDialog)

logger = logging.getLogger(__name__)


class RestorationWorker(QThread):
    """Worker thread for photo restoration."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str, str)  # success, output_path, error

    def __init__(self, ml_pipeline: MLPipeline, task: RestorationTask):
        super().__init__()
        self.ml_pipeline = ml_pipeline
        self.task = task

    def run(self) -> None:
        def progress_cb(value: int, detail: str):
            self.progress.emit(value, detail)

        result = self.ml_pipeline.restore_sync(self.task)
        self.finished.emit(result.success, str(result.output_path) if result.output_path else "", result.error)


class QualityEnhancementWorker(QThread):
    """Worker thread for quality enhancement (super-resolution + denoise + sharpen)."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, object, str)  # success, result_image, error

    def __init__(self, input_path: Path, output_path: Path, settings: QESettings, progress_callback=None):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.settings = settings
        self.progress_callback = progress_callback

    def run(self) -> None:
        try:
            from PIL import Image
            from photo_retouch.processing.quality_enhance import QualityEnhancer

            # Load image
            self.progress.emit(10, "Загрузка изображения...")
            img = Image.open(self.input_path)
            img.load()

            # Apply enhancement
            self.progress.emit(30, "Супер-резолюция + шумоподавление + резкость...")
            enhancer = QualityEnhancer()
            result = enhancer.enhance(img, self.settings)

            self.progress.emit(90, "Сохранение результата...")
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            result.image.save(self.output_path)

            self.progress.emit(100, "Готово!")
            self.finished.emit(True, result.image, "")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(False, None, str(e))


class SwallowModeWorker(QThread):
    """Worker thread for Swallow mode (beard/wart removal)."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, object, str)  # success, result_image, error

    def __init__(self, input_path: Path, output_path: Path, settings: SMSettings, progress_callback=None):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.settings = settings
        self.progress_callback = progress_callback

    def run(self) -> None:
        try:
            from PIL import Image
            from photo_retouch.processing.swallow_mode import SwallowProcessor

            # Load image
            self.progress.emit(10, "Загрузка изображения...")
            img = Image.open(self.input_path)
            img.load()

            # Apply swallow mode
            self.progress.emit(30, "Обнаружение бороды, бородавок и дефектов...")
            processor = SwallowProcessor()
            result = processor.process(img, self.settings)

            self.progress.emit(90, "Сохранение результата...")
            if result.success:
                self.finished.emit(True, result.image, "")
            else:
                self.finished.emit(False, None, result.error)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(False, None, str(e))


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self, project_root: Path):
        super().__init__()
        self.project_root = Path(project_root).resolve()
        self.project: Optional[Project] = None
        self.asset_manager = AssetManager(self.project_root)
        self.ml_pipeline = MockMLPipeline()  # Use mock for now
        self.compositor = Compositor()
        self._auto_save_timer = QTimer()
        self._dirty = False

        # Undo/Redo history
        self._history: List[Dict[str, Any]] = []
        self._history_index = -1
        self._max_history = 50
        self._history_enabled = True

        self._setup_ui()
        self._setup_shortcuts()
        self._setup_connections()
        self._load_settings()
        self._scan_assets()
        self.new_project()

    def _setup_ui(self) -> None:
        """Setup the main UI."""
        self.setWindowTitle("Photo Retouch Pro")
        self.resize(1400, 900)
        self.setMinimumSize(1024, 768)

        # Central widget - Image Viewer
        self.image_viewer = ImageViewer()
        self.setCentralWidget(self.image_viewer)

        # Create dock widgets
        self._create_dock_widgets()

        # Toolbar
        self._create_toolbar()

        # Menubar
        self._create_menubar()

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Готово")

        # Progress bar in status bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

        # Zoom label
        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(60)
        self.status_bar.addPermanentWidget(self.zoom_label)

    def _create_dock_widgets(self) -> None:
        """Create dockable panels."""
        # Left panel - Assets
        self.assets_dock = QDockWidget("Ресурсы", self)
        self.assets_dock.setObjectName("assetsDock")
        self.assets_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea |
                                          Qt.DockWidgetArea.RightDockWidgetArea)
        self.assets_widget = QWidget()
        self.assets_layout = QVBoxLayout(self.assets_widget)
        self.assets_layout.setContentsMargins(0, 0, 0, 0)

        self.asset_tabs = QTabWidget()
        self.asset_galleries = {}

        for category in [AssetCategory.FRAMES, AssetCategory.BACKGROUNDS,
                         AssetCategory.COSTUMES, AssetCategory.FONTS,
                         AssetCategory.ENHANCED_QUALITY_PHOTO]:
            gallery = AssetGallery(category)
            gallery.set_asset_manager(self.asset_manager)
            gallery.assetDoubleClicked.connect(self._on_asset_double_clicked)
            self.asset_tabs.addTab(gallery, category.value.replace("_", " ").title())
            self.asset_galleries[category] = gallery

        self.assets_layout.addWidget(self.asset_tabs)
        self.assets_dock.setWidget(self.assets_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.assets_dock)

        # Right panel - Controls
        self.controls_dock = QDockWidget("Управление", self)
        self.controls_dock.setObjectName("controlsDock")
        self.controls_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea |
                                            Qt.DockWidgetArea.RightDockWidgetArea)
        self.controls_widget = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_widget)
        self.controls_layout.setContentsMargins(4, 4, 4, 4)

        self.control_tabs = QTabWidget()

        # Adjustments tab
        self.adjustments_widget = QWidget()
        self.adjustments_layout = QVBoxLayout(self.adjustments_widget)

        self.brightness_slider = AdjustmentSlider("Яркость", -100, 100, 0)
        self.contrast_slider = AdjustmentSlider("Контраст", -100, 100, 0)
        self.sharpness_slider = AdjustmentSlider("Резкость", 0, 100, 0)
        self.saturation_slider = AdjustmentSlider("Насыщенность", -100, 100, 0)
        self.temperature_slider = AdjustmentSlider("Температура", -100, 100, 0)
        self.vignette_slider = AdjustmentSlider("Виньетка", 0, 100, 0)

        self.adjustments_layout.addWidget(self.brightness_slider)
        self.adjustments_layout.addWidget(self.contrast_slider)
        self.adjustments_layout.addWidget(self.sharpness_slider)
        self.adjustments_layout.addWidget(self.saturation_slider)
        self.adjustments_layout.addWidget(self.temperature_slider)
        self.adjustments_layout.addWidget(self.vignette_slider)

        # Reset button
        self.reset_adjustments_btn = QPushButton("Сбросить коррекцию")
        self.reset_adjustments_btn.clicked.connect(self._reset_adjustments)
        self.adjustments_layout.addWidget(self.reset_adjustments_btn)
        self.adjustments_layout.addStretch()

        self.control_tabs.addTab(self.adjustments_widget, "Коррекция")

        # Layers tab
        self.layer_control = LayerControlWidget()
        self.control_tabs.addTab(self.layer_control, "Слои")

        # Text tab
        self.text_widget = QWidget()
        self.text_layout = QVBoxLayout(self.text_widget)
        self.add_text_btn = QPushButton("Добавить текст")
        self.add_text_btn.clicked.connect(self._add_text_overlay)
        self.edit_text_btn = QPushButton("Редактировать текст")
        self.edit_text_btn.clicked.connect(self._edit_text_overlay)
        self.edit_text_btn.setEnabled(False)
        self.remove_text_btn = QPushButton("Удалить текст")
        self.remove_text_btn.clicked.connect(self._remove_text_overlay)
        self.remove_text_btn.setEnabled(False)
        self.text_layout.addWidget(self.add_text_btn)
        self.text_layout.addWidget(self.edit_text_btn)
        self.text_layout.addWidget(self.remove_text_btn)
        self.text_layout.addStretch()
        self.control_tabs.addTab(self.text_widget, "Текст")

        # ML Restoration tab
        self.ml_widget = QWidget()
        self.ml_layout = QVBoxLayout(self.ml_widget)
        self.restore_btn = QPushButton("Восстановить фото")
        self.restore_btn.clicked.connect(self._restore_photo)

        # Enhance Quality button
        self.enhance_quality_btn = QPushButton("Улучшить качество")
        self.enhance_quality_btn.clicked.connect(self._enhance_quality)
        self.enhance_quality_btn.setToolTip("AI-супер-резолюция + шумоподавление + повышение резкости")

        self.ml_model_combo = QComboBox()
        self.ml_model_combo.addItems([m.value for m in RestorationModel])
        self.ml_fidelity_slider = AdjustmentSlider("Сила восстановления", 0, 100, 75, "%")
        self.ml_layout.addWidget(QLabel("Модель:"))
        self.ml_layout.addWidget(self.ml_model_combo)
        self.ml_layout.addWidget(self.ml_fidelity_slider)
        self.ml_layout.addWidget(self.restore_btn)
        self.ml_layout.addWidget(self.enhance_quality_btn)
        self.ml_layout.addStretch()
        self.control_tabs.addTab(self.ml_widget, "ML Восстановление")

        # Costume tab
        self.costume_widget = QWidget()
        self.costume_layout = QVBoxLayout(self.costume_widget)
        self.costume_scale_slider = AdjustmentSlider("Масштаб", 10, 300, 100, "%")
        self.costume_rotation_slider = AdjustmentSlider("Поворот", -180, 180, 0, "°")
        self.costume_opacity_slider = AdjustmentSlider("Прозрачность", 0, 100, 100, "%")
        self.costume_flip_h_btn = QPushButton("Отразить по горизонтали")
        self.costume_flip_h_btn.clicked.connect(lambda: self._flip_costume("horizontal"))
        self.costume_flip_v_btn = QPushButton("Отразить по вертикали")
        self.costume_flip_v_btn.clicked.connect(lambda: self._flip_costume("vertical"))
        self.remove_costume_btn = QPushButton("Удалить костюм")
        self.remove_costume_btn.clicked.connect(self._remove_costume)
        self.remove_costume_btn.setEnabled(False)

        self.costume_layout.addWidget(QLabel("Настройки костюма:"))
        self.costume_layout.addWidget(self.costume_scale_slider)
        self.costume_layout.addWidget(self.costume_rotation_slider)
        self.costume_layout.addWidget(self.costume_opacity_slider)
        self.costume_layout.addWidget(self.costume_flip_h_btn)
        self.costume_layout.addWidget(self.costume_flip_v_btn)
        self.costume_layout.addWidget(self.remove_costume_btn)
        self.costume_layout.addStretch()
        self.control_tabs.addTab(self.costume_widget, "Костюм")

        # Quality Enhancement tab
        self.quality_widget = QWidget()
        self.quality_layout = QVBoxLayout(self.quality_widget)

        self.quality_model_combo = QComboBox()
        self.quality_model_combo.addItems(["RealESRGAN_x4plus", "RealESRGAN_x2plus", "BSRGAN", "Traditional"])
        self.quality_denoise_slider = AdjustmentSlider("Шумоподавление", 0, 100, 50, "%")
        self.quality_sharpen_slider = AdjustmentSlider("Резкость", 0, 100, 50, "%")
        self.quality_face_enhance_cb = QCheckBox("Улучшение лица")
        self.quality_face_enhance_cb.setChecked(True)

        self.quality_layout.addWidget(QLabel("Модель:"))
        self.quality_layout.addWidget(self.quality_model_combo)
        self.quality_layout.addWidget(self.quality_denoise_slider)
        self.quality_layout.addWidget(self.quality_sharpen_slider)
        self.quality_layout.addWidget(self.quality_face_enhance_cb)
        self.quality_layout.addWidget(self.enhance_quality_btn)
        self.quality_layout.addStretch()
        self.control_tabs.addTab(self.quality_widget, "Качество")

        # Swallow Mode tab (Режим ластки)
        self.swallow_widget = QWidget()
        self.swallow_layout = QVBoxLayout(self.swallow_widget)

        self.swallow_enabled_cb = QCheckBox("Включить режим ластки")
        self.swallow_enabled_cb.setChecked(False)
        self.swallow_enabled_cb.toggled.connect(self._on_swallow_enabled_changed)

        self.swallow_beard_cb = QCheckBox("Убирать бороду")
        self.swallow_beard_cb.setChecked(True)
        self.swallow_warts_cb = QCheckBox("Убирать бородавки")
        self.swallow_warts_cb.setChecked(True)
        self.swallow_moles_cb = QCheckBox("Убирать родинки")
        self.swallow_moles_cb.setChecked(False)
        self.swallow_acne_cb = QCheckBox("Убирать акне/прыщи")
        self.swallow_acne_cb.setChecked(True)

        self.swallow_transition_slider = AdjustmentSlider("Плавность перехода", 0, 100, 80, "%")
        self.swallow_transition_slider.setToolTip("Насколько плавно соединять обработанную зону с необработанной")
        self.swallow_texture_slider = AdjustmentSlider("Сохранение текстуры кожи", 0, 100, 70, "%")
        self.swallow_texture_slider.setToolTip("Сколько оригинальной текстуры кожи сохранить")
        self.swallow_sensitivity_slider = AdjustmentSlider("Чувствительность обнаружения", 0, 100, 60, "%")
        self.swallow_sensitivity_slider.setToolTip("Насколько агрессивно находить дефекты")

        self.swallow_apply_btn = QPushButton("Применить режим ластки")
        self.swallow_apply_btn.clicked.connect(self._apply_swallow_mode)
        self.swallow_apply_btn.setEnabled(False)

        self.swallow_layout.addWidget(self.swallow_enabled_cb)
        self.swallow_layout.addWidget(self.swallow_beard_cb)
        self.swallow_layout.addWidget(self.swallow_warts_cb)
        self.swallow_layout.addWidget(self.swallow_moles_cb)
        self.swallow_layout.addWidget(self.swallow_acne_cb)
        self.swallow_layout.addWidget(QLabel("Плавность перехода к необработанной зоне:"))
        self.swallow_layout.addWidget(self.swallow_transition_slider)
        self.swallow_layout.addWidget(QLabel("Сохранение текстуры кожи:"))
        self.swallow_layout.addWidget(self.swallow_texture_slider)
        self.swallow_layout.addWidget(QLabel("Чувствительность обнаружения:"))
        self.swallow_layout.addWidget(self.swallow_sensitivity_slider)
        self.swallow_layout.addWidget(self.swallow_apply_btn)
        self.swallow_layout.addStretch()
        self.control_tabs.addTab(self.swallow_widget, "Режим ластки")

        self.controls_layout.addWidget(self.control_tabs)

        # Photo selection buttons
        photo_group = QWidget()
        photo_layout = QVBoxLayout(photo_group)
        self.load_initial_btn = QPushButton("Загрузить начальное фото")
        self.load_initial_btn.clicked.connect(self._load_initial_photo)
        self.load_restored_btn = QPushButton("Загрузить восстановленное фото")
        self.load_restored_btn.clicked.connect(self._load_restored_photo)
        photo_layout.addWidget(self.load_initial_btn)
        photo_layout.addWidget(self.load_restored_btn)
        self.controls_layout.addWidget(photo_group)

        self.controls_dock.setWidget(self.controls_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.controls_dock)

        # Bottom panel - Initial/Restored/Enhanced photo pickers
        self.photos_dock = QDockWidget("Фотографии", self)
        self.photos_dock.setObjectName("photosDock")
        self.photos_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self.photos_widget = QWidget()
        self.photos_layout = QHBoxLayout(self.photos_widget)

        self.initial_gallery = AssetGallery(AssetCategory.INITIAL_PHOTO)
        self.initial_gallery.set_asset_manager(self.asset_manager)
        self.initial_gallery.assetDoubleClicked.connect(
            lambda name, cat: self._set_initial_photo(name))

        self.restored_gallery = AssetGallery(AssetCategory.RESTORED_PHOTO)
        self.restored_gallery.set_asset_manager(self.asset_manager)
        self.restored_gallery.assetDoubleClicked.connect(
            lambda name, cat: self._set_restored_photo(name))

        self.enhanced_gallery = AssetGallery(AssetCategory.ENHANCED_QUALITY_PHOTO)
        self.enhanced_gallery.set_asset_manager(self.asset_manager)
        self.enhanced_gallery.assetDoubleClicked.connect(
            lambda name, cat: self._set_enhanced_quality_photo(name))

        self.photos_layout.addWidget(self.initial_gallery)
        self.photos_layout.addWidget(self.restored_gallery)
        self.photos_layout.addWidget(self.enhanced_gallery)
        self.photos_dock.setWidget(self.photos_widget)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.photos_dock)

        # Tabify docks
        self.tabifyDockWidget(self.controls_dock, self.assets_dock)
        self.controls_dock.raise_()

    def _create_toolbar(self) -> None:
        """Create main toolbar."""
        toolbar = QToolBar("Основная")
        toolbar.setObjectName("mainToolBar")  # Required for saveState/restoreState
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # File actions
        new_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon), "Новый", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self.new_project)
        toolbar.addAction(new_action)

        open_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon), "Открыть", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_project)
        toolbar.addAction(open_action)

        save_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton), "Сохранить", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_project)
        toolbar.addAction(save_action)

        toolbar.addSeparator()

        # Export
        export_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton), "Экспорт", self)
        export_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        export_action.triggered.connect(self.export_final)
        toolbar.addAction(export_action)

        toolbar.addSeparator()

        # Undo/Redo
        undo_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowLeft), "Отменить", self)
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        undo_action.triggered.connect(self.undo)
        toolbar.addAction(undo_action)

        redo_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowRight), "Повторить", self)
        redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        redo_action.triggered.connect(self.redo)
        toolbar.addAction(redo_action)

        toolbar.addSeparator()

        # Zoom
        zoom_in_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarMaxButton), "Увеличить", self)
        zoom_in_action.setShortcut(QKeySequence.StandardKey.ZoomIn)
        zoom_in_action.triggered.connect(lambda: self.image_viewer.zoom_by(1.2))
        toolbar.addAction(zoom_in_action)

        zoom_out_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarMinButton), "Уменьшить", self)
        zoom_out_action.setShortcut(QKeySequence.StandardKey.ZoomOut)
        zoom_out_action.triggered.connect(lambda: self.image_viewer.zoom_by(1/1.2))
        toolbar.addAction(zoom_out_action)

        zoom_reset_action = QAction("100%", self)
        zoom_reset_action.setShortcut(QKeySequence("Ctrl+0"))
        zoom_reset_action.triggered.connect(self.image_viewer.reset_zoom)
        toolbar.addAction(zoom_reset_action)

        toolbar.addSeparator()

        # Transform tools
        self.crop_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView), "Обрезка (C)", self)
        self.crop_action.setShortcut(QKeySequence("C"))
        self.crop_action.setCheckable(True)
        self.crop_action.toggled.connect(self._toggle_crop_tool)
        toolbar.addAction(self.crop_action)

        rotate_left_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowLeft), "Повернуть влево", self)
        rotate_left_action.setShortcut(QKeySequence("Ctrl+L"))
        rotate_left_action.triggered.connect(lambda: self._rotate_photo(-90))
        toolbar.addAction(rotate_left_action)

        rotate_right_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowRight), "Повернуть вправо", self)
        rotate_right_action.setShortcut(QKeySequence("Ctrl+R"))
        rotate_right_action.triggered.connect(lambda: self._rotate_photo(90))
        toolbar.addAction(rotate_right_action)

        flip_h_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogListView), "Отразить по горизонтали", self)
        flip_h_action.setShortcut(QKeySequence("Ctrl+H"))
        flip_h_action.triggered.connect(lambda: self._flip_photo("horizontal"))
        toolbar.addAction(flip_h_action)

        flip_v_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogListView), "Отразить по вертикали", self)
        flip_v_action.setShortcut(QKeySequence("Ctrl+Shift+H"))
        flip_v_action.triggered.connect(lambda: self._flip_photo("vertical"))
        toolbar.addAction(flip_v_action)

        toolbar.addSeparator()

        # Tool mode buttons
        select_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowUp), "Выделение (V)", self)
        select_action.setShortcut(QKeySequence("V"))
        select_action.setCheckable(True)
        select_action.setChecked(True)
        select_action.triggered.connect(lambda: self._set_tool_mode("select"))
        toolbar.addAction(select_action)

        pan_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowRight), "Перемещение (H)", self)
        pan_action.setShortcut(QKeySequence("H"))
        pan_action.setCheckable(True)
        pan_action.triggered.connect(lambda: self._set_tool_mode("pan"))
        toolbar.addAction(pan_action)

    def _create_menubar(self) -> None:
        """Create menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("Файл")
        new_action = QAction("Новый проект", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self.new_project)
        file_menu.addAction(new_action)

        open_action = QAction("Открыть проект...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_project)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        save_action = QAction("Сохранить проект", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)

        save_as_action = QAction("Сохранить проект как...", self)
        save_as_action.triggered.connect(self.save_project_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        export_action = QAction("Экспорт готовой ретуши...", self)
        export_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        export_action.triggered.connect(self.export_final)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        quit_action = QAction("Выход", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Edit menu
        edit_menu = menubar.addMenu("Правка")
        undo_action = QAction("Отменить", self)
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        edit_menu.addAction(undo_action)

        redo_action = QAction("Повторить", self)
        redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        edit_menu.addAction(redo_action)

        # View menu
        view_menu = menubar.addMenu("Вид")
        view_menu.addAction(self.assets_dock.toggleViewAction())
        view_menu.addAction(self.controls_dock.toggleViewAction())
        view_menu.addAction(self.photos_dock.toggleViewAction())

        # Tools menu
        tools_menu = menubar.addMenu("Инструменты")
        batch_action = QAction("Пакетная обработка...", self)
        batch_action.triggered.connect(self._batch_process)
        tools_menu.addAction(batch_action)

        export_settings_action = QAction("Настройки экспорта...", self)
        export_settings_action.triggered.connect(self._export_settings)
        tools_menu.addAction(export_settings_action)

        # Help menu
        help_menu = menubar.addMenu("Справка")
        about_action = QAction("О программе", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_shortcuts(self) -> None:
        """Setup additional keyboard shortcuts."""
        # Delete key for removing selected item
        delete_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Delete), self)
        delete_shortcut.activated.connect(self._delete_selected)

        # Escape to clear selection
        escape_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        escape_shortcut.activated.connect(self.image_viewer._clear_selection)

    def _setup_connections(self) -> None:
        """Connect signals."""
        # Adjustment sliders - valueChanged for real-time preview
        self.brightness_slider.valueChanged.connect(self._on_adjustment_changed)
        self.contrast_slider.valueChanged.connect(self._on_adjustment_changed)
        self.sharpness_slider.valueChanged.connect(self._on_adjustment_changed)
        self.saturation_slider.valueChanged.connect(self._on_adjustment_changed)
        self.temperature_slider.valueChanged.connect(self._on_adjustment_changed)
        self.vignette_slider.valueChanged.connect(self._on_adjustment_changed)

        # Adjustment sliders - sliderReleased for history
        self.brightness_slider.slider.sliderReleased.connect(self._push_history)
        self.contrast_slider.slider.sliderReleased.connect(self._push_history)
        self.sharpness_slider.slider.sliderReleased.connect(self._push_history)
        self.saturation_slider.slider.sliderReleased.connect(self._push_history)
        self.temperature_slider.slider.sliderReleased.connect(self._push_history)
        self.vignette_slider.slider.sliderReleased.connect(self._push_history)

        # Costume sliders
        self.costume_scale_slider.valueChanged.connect(self._on_costume_transform_changed)
        self.costume_rotation_slider.valueChanged.connect(self._on_costume_transform_changed)
        self.costume_opacity_slider.valueChanged.connect(self._on_costume_opacity_changed)

        # Costume sliders - sliderReleased for history
        self.costume_scale_slider.slider.sliderReleased.connect(self._push_history)
        self.costume_rotation_slider.slider.sliderReleased.connect(self._push_history)
        self.costume_opacity_slider.slider.sliderReleased.connect(self._push_history)

        # Layer control
        self.layer_control.layerVisibilityChanged.connect(self._on_layer_visibility_changed)
        self.layer_control.layerOrderChanged.connect(self._on_layer_order_changed)

        # Image viewer
        self.image_viewer.zoomChanged.connect(self._on_zoom_changed)
        self.image_viewer.itemSelected.connect(self._on_item_selected)
        self.image_viewer.costumeTransformChanged.connect(self._on_costume_transform_from_viewer)
        self.image_viewer.textPositionChanged.connect(self._on_text_position_changed)

        # ML fidelity slider
        self.ml_fidelity_slider.valueChanged.connect(self._on_ml_fidelity_changed)

        # Quality enhancement connections
        self.quality_model_combo.currentTextChanged.connect(self._on_quality_model_changed)
        self.quality_denoise_slider.valueChanged.connect(self._on_quality_settings_changed)
        self.quality_sharpen_slider.valueChanged.connect(self._on_quality_settings_changed)
        self.quality_face_enhance_cb.toggled.connect(self._on_quality_settings_changed)
        self.quality_denoise_slider.slider.sliderReleased.connect(self._push_history)
        self.quality_sharpen_slider.slider.sliderReleased.connect(self._push_history)

        # Swallow mode connections
        self.swallow_beard_cb.toggled.connect(self._on_swallow_settings_changed)
        self.swallow_warts_cb.toggled.connect(self._on_swallow_settings_changed)
        self.swallow_moles_cb.toggled.connect(self._on_swallow_settings_changed)
        self.swallow_acne_cb.toggled.connect(self._on_swallow_settings_changed)
        self.swallow_transition_slider.valueChanged.connect(self._on_swallow_settings_changed)
        self.swallow_texture_slider.valueChanged.connect(self._on_swallow_settings_changed)
        self.swallow_sensitivity_slider.valueChanged.connect(self._on_swallow_settings_changed)
        self.swallow_transition_slider.slider.sliderReleased.connect(self._push_history)
        self.swallow_texture_slider.slider.sliderReleased.connect(self._push_history)
        self.swallow_sensitivity_slider.slider.sliderReleased.connect(self._push_history)

        # Auto-save timer
        self._auto_save_timer.timeout.connect(self._auto_save)
        self._auto_save_timer.start(30000)  # 30 seconds

    def _scan_assets(self) -> None:
        """Scan all asset folders."""
        result = self.asset_manager.scan_all()
        logger.info(f"Scanned assets: {result.total_files} files, {len(result.errors)} errors")
        for error in result.errors:
            logger.warning(f"Asset scan error: {error}")

    def _update_composite(self) -> None:
        """Update the composite image in viewer."""
        if not self.project:
            logger.debug("No project, skipping composite update")
            return

        logger.debug("Updating composite...")
        # Update compositor with current project state
        self._apply_project_to_compositor()

        # Get composite image
        composite = self.compositor.composite()
        logger.debug(f"Composite size: {composite.size}, mode: {composite.mode}")

        # Convert to QImage
        qimage = self._pil_to_qimage(composite)
        logger.debug(f"QImage size: {qimage.width()}x{qimage.height()}")
        self.image_viewer.set_composite_image(qimage)

        self._dirty = True
        logger.debug("Composite updated")

    def _apply_project_to_compositor(self) -> None:
        """Apply project state to compositor."""
        p = self.project

        # Background
        if p.background:
            bg_path = self.project_root / p.background
            if bg_path.exists():
                img = load_image_safe(bg_path)
                if img:
                    self.compositor.set_layer_image("background", img)
                    logger.debug(f"Background loaded: {bg_path}")
            else:
                logger.warning(f"Background not found: {bg_path}")

        # Photo (restored or initial)
        photo_path = None
        if p.restored_photo:
            photo_path = self.project_root / p.restored_photo
        elif p.initial_photo:
            photo_path = self.project_root / p.initial_photo

        if photo_path and photo_path.exists():
            img = load_image_safe(photo_path)
            if img:
                # Apply adjustments
                img = ImageProcessor.apply_adjustments(img, p.adjustments)
                self.compositor.set_layer_image("photo", img)
                logger.debug(f"Photo loaded: {photo_path}")
            else:
                logger.warning(f"Failed to load photo: {photo_path}")
                self.compositor.set_layer_image("photo", None)
        else:
            # No photo set yet - this is normal for new projects
            logger.debug(f"No photo set (photo_path={photo_path})")
            # Clear photo layer
            self.compositor.set_layer_image("photo", None)

        # Costume
        if p.costume:
            costume_path = self.project_root / p.costume
            if costume_path.exists():
                img = load_image_safe(costume_path)
                if img:
                    self.compositor.set_layer_image("costume", img, p.costume_transform)
                    # Also set in image viewer for interactive editing
                    qimage = self._pil_to_qimage(img)
                    self.image_viewer.set_costume_image(qimage, p.costume_transform)
                    # Enable costume controls
                    self.remove_costume_btn.setEnabled(True)
                    self.costume_scale_slider.set_value(int(p.costume_transform.scale * 100))
                    self.costume_rotation_slider.set_value(int(p.costume_transform.rotation))
                    self.costume_opacity_slider.set_value(100)
                    logger.debug(f"Costume loaded: {costume_path}")
            else:
                logger.warning(f"Costume not found: {costume_path}")
                self.compositor.set_layer_image("costume", None)
                self.image_viewer.clear_costume()
        else:
            self.image_viewer.clear_costume()
            self.remove_costume_btn.setEnabled(False)

        # Frame
        if p.frame:
            frame_path = self.project_root / p.frame
            if frame_path.exists():
                img = load_image_safe(frame_path)
                if img:
                    self.compositor.set_layer_image("frame", img)
                    logger.debug(f"Frame loaded: {frame_path}")
            else:
                logger.warning(f"Frame not found: {frame_path}")
                self.compositor.set_layer_image("frame", None)

        # Text
        if p.text_overlay.content and p.text_overlay.visible:
            font_path = ""
            if p.font:
                font_path = str(self.project_root / p.font)
            elif p.text_overlay.font_path:
                font_path = p.text_overlay.font_path

            text_img = TextRenderer.render_text(p.text_overlay, self.compositor.canvas_size, font_path)
            self.compositor.set_layer_image("text", text_img)
            # Also set in image viewer for interactive editing (drag to move)
            self.image_viewer.set_text_overlay(
                p.text_overlay.content,
                p.text_overlay.position,
                p.text_overlay.style,
                font_path
            )
            self.edit_text_btn.setEnabled(True)
            self.remove_text_btn.setEnabled(True)
            logger.debug(f"Text rendered: {p.text_overlay.content[:20]}...")
        else:
            self.compositor.set_layer_image("text", None)
            self.image_viewer.clear_text_overlay()
            self.edit_text_btn.setEnabled(False)
            self.remove_text_btn.setEnabled(False)

        # Layer visibility
        for layer_name, visible in p.layer_visibility.items():
            self.compositor.set_layer_visibility(layer_name, visible)

    def _pil_to_qimage(self, pil_image) -> QImage:
        """Convert PIL Image to QImage."""
        if pil_image.mode == "RGBA":
            data = pil_image.tobytes("raw", "RGBA")
            return QImage(data, pil_image.width, pil_image.height,
                          pil_image.width * 4, QImage.Format.Format_RGBA8888)
        elif pil_image.mode == "RGB":
            data = pil_image.tobytes("raw", "RGB")
            return QImage(data, pil_image.width, pil_image.height,
                          pil_image.width * 3, QImage.Format.Format_RGB888)
        else:
            pil_image = pil_image.convert("RGBA")
            return self._pil_to_qimage(pil_image)

    def new_project(self) -> None:
        """Create a new project."""
        if self._dirty and not self._confirm_discard():
            return

        self._history_enabled = False
        try:
            self.project = Project()
            self.project.created = datetime.now().isoformat()
            self.project.modified = datetime.now().isoformat()

            # Reset UI
            self.brightness_slider.set_value(0)
            self.contrast_slider.set_value(0)
            self.sharpness_slider.set_value(0)
            self.saturation_slider.set_value(0)
            self.temperature_slider.set_value(0)
            self.vignette_slider.set_value(0)
            self.ml_fidelity_slider.set_value(75)
            self.ml_model_combo.setCurrentIndex(0)

            self.image_viewer.clear_overlays()
            self.image_viewer.clear_costume()
            self.image_viewer.clear_text_overlay()
            self.compositor = Compositor()
            self._update_composite()
            self._dirty = False
            self._history = []
            self._history_index = -1
            self.status_bar.showMessage("Новый проект создан")
        finally:
            self._history_enabled = True

    def open_project(self) -> None:
        """Open a project file."""
        if self._dirty and not self._confirm_discard():
            return

        filepath, _ = QFileDialog.getOpenFileName(
            self, "Открыть проект", str(self.project_root),
            "Project Files (*.proj *.json);;All Files (*)"
        )
        if filepath:
            self._load_project(Path(filepath))

    def _load_project(self, filepath: Path) -> None:
        """Load project from file."""
        self._history_enabled = False
        try:
            self.project = Project.load(filepath)
            self._apply_project_to_ui()
            self._update_composite()
            self._dirty = False
            self._history = []
            self._history_index = -1
            self.status_bar.showMessage(f"Проект загружен: {filepath.name}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить проект:\n{e}")
            logger.error(f"Project load failed: {e}")
        finally:
            self._history_enabled = True

    def _apply_project_to_ui(self) -> None:
        """Apply project state to UI controls."""
        if not self.project:
            return

        p = self.project
        self.brightness_slider.set_value(p.adjustments.brightness)
        self.contrast_slider.set_value(p.adjustments.contrast)
        self.sharpness_slider.set_value(p.adjustments.sharpness)
        self.saturation_slider.set_value(p.adjustments.saturation if hasattr(p.adjustments, 'saturation') else 0)
        self.temperature_slider.set_value(p.adjustments.temperature if hasattr(p.adjustments, 'temperature') else 0)
        self.vignette_slider.set_value(p.adjustments.vignette if hasattr(p.adjustments, 'vignette') else 0)
        self.ml_fidelity_slider.set_value(int(p.ml_settings.fidelity * 100))

        model_idx = self.ml_model_combo.findText(p.ml_settings.model)
        if model_idx >= 0:
            self.ml_model_combo.setCurrentIndex(model_idx)

        # Quality enhancement settings
        self.quality_model_combo.setCurrentText(p.quality_enhance_settings.model)
        self.quality_denoise_slider.set_value(int(p.quality_enhance_settings.denoise_strength * 100))
        self.quality_sharpen_slider.set_value(int(p.quality_enhance_settings.sharpen_strength * 100))
        self.quality_face_enhance_cb.setChecked(p.quality_enhance_settings.face_enhance)

        # Swallow mode settings
        self.swallow_enabled_cb.setChecked(p.swallow_mode_settings.enabled)
        self.swallow_beard_cb.setChecked(p.swallow_mode_settings.remove_beard)
        self.swallow_warts_cb.setChecked(p.swallow_mode_settings.remove_warts)
        self.swallow_moles_cb.setChecked(p.swallow_mode_settings.remove_moles)
        self.swallow_acne_cb.setChecked(p.swallow_mode_settings.remove_acne)
        self.swallow_transition_slider.set_value(int(p.swallow_mode_settings.smooth_transition * 100))
        self.swallow_texture_slider.set_value(int(p.swallow_mode_settings.preserve_skin_texture * 100))
        self.swallow_sensitivity_slider.set_value(int(p.swallow_mode_settings.detection_sensitivity * 100))
        self.swallow_apply_btn.setEnabled(p.swallow_mode_settings.enabled)

        # Layer visibility
        for layer_name, visible in p.layer_visibility.items():
            self.layer_control.set_visibility(layer_name, visible)

        # Text buttons
        has_text = bool(p.text_overlay.content)
        self.edit_text_btn.setEnabled(has_text)
        self.remove_text_btn.setEnabled(has_text)

        # Costume controls
        has_costume = bool(p.costume)
        self.remove_costume_btn.setEnabled(has_costume)
        if has_costume:
            self.costume_scale_slider.set_value(int(p.costume_transform.scale * 100))
            self.costume_rotation_slider.set_value(int(p.costume_transform.rotation))
            self.costume_opacity_slider.set_value(100)

        # Refresh galleries
        for gallery in self.asset_galleries.values():
            gallery.refresh()
        self.initial_gallery.refresh()
        self.restored_gallery.refresh()
        self.enhanced_gallery.refresh()

    def save_project(self) -> None:
        """Save project to file."""
        if not self.project:
            return

        if not hasattr(self, '_project_filepath') or not self._project_filepath:
            self.save_project_as()
        else:
            self._save_project_to(self._project_filepath)

    def save_project_as(self) -> None:
        """Save project with new filename."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Сохранить проект", str(self.project_root / "project.proj"),
            "Project Files (*.proj);;JSON Files (*.json);;All Files (*)"
        )
        if filepath:
            self._save_project_to(Path(filepath))

    def _save_project_to(self, filepath: Path) -> None:
        """Save project to specific path."""
        try:
            self._gather_project_from_ui()
            self.project.save(filepath)
            self._project_filepath = filepath
            self._dirty = False
            self.status_bar.showMessage(f"Проект сохранен: {filepath.name}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить проект:\n{e}")
            logger.error(f"Project save failed: {e}")

    def _gather_project_from_ui(self) -> None:
        """Gather current UI state into project."""
        if not self.project:
            return

        p = self.project
        p.adjustments = Adjustments(
            brightness=self.brightness_slider.value(),
            contrast=self.contrast_slider.value(),
            sharpness=self.sharpness_slider.value(),
            saturation=self.saturation_slider.value() if hasattr(self, 'saturation_slider') else 0,
            temperature=self.temperature_slider.value() if hasattr(self, 'temperature_slider') else 0,
            vignette=self.vignette_slider.value() if hasattr(self, 'vignette_slider') else 0
        )
        p.ml_settings = MLSettings(
            model=self.ml_model_combo.currentText(),
            fidelity=self.ml_fidelity_slider.value() / 100.0
        )
        p.quality_enhance_settings = QualityEnhanceSettings(
            model=self.quality_model_combo.currentText(),
            denoise_strength=self.quality_denoise_slider.value() / 100.0,
            sharpen_strength=self.quality_sharpen_slider.value() / 100.0,
            face_enhance=self.quality_face_enhance_cb.isChecked()
        )
        p.swallow_mode_settings = SwallowModeSettings(
            enabled=self.swallow_enabled_cb.isChecked(),
            remove_beard=self.swallow_beard_cb.isChecked(),
            remove_warts=self.swallow_warts_cb.isChecked(),
            remove_moles=self.swallow_moles_cb.isChecked(),
            remove_acne=self.swallow_acne_cb.isChecked(),
            smooth_transition=self.swallow_transition_slider.value() / 100.0,
            preserve_skin_texture=self.swallow_texture_slider.value() / 100.0,
            detection_sensitivity=self.swallow_sensitivity_slider.value() / 100.0
        )
        p.modified = datetime.now().isoformat()

    def export_final(self) -> None:
        """Export final retouched image."""
        if not self.project:
            QMessageBox.warning(self, "Предупреждение", "Нет открытого проекта")
            return

        filepath, selected_filter = QFileDialog.getSaveFileName(
            self, "Экспорт готовой ретуши",
            str(self.project_root / "07_FinalRetouch" / "retouched.png"),
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;TIFF (*.tiff);;BMP 8-bit (256 colors) (*.bmp);;All Files (*)"
        )
        if not filepath:
            return

        self._gather_project_from_ui()
        self._apply_project_to_compositor()
        composite = self.compositor.composite()

        # Save
        fmt = Path(filepath).suffix[1:].upper()
        if fmt == "JPG":
            fmt = "JPEG"

        # Check if BMP 8-bit is selected
        bmp_8bit = "BMP 8-bit" in selected_filter or fmt == "BMP"

        if save_image(composite, Path(filepath), format=fmt, bmp_8bit=bmp_8bit):
            self.status_bar.showMessage(f"Экспортировано: {Path(filepath).name}")
            QMessageBox.information(self, "Успех", "Изображение успешно экспортировано")
        else:
            QMessageBox.critical(self, "Ошибка", "Не удалось сохранить изображение")

    def _load_initial_photo(self) -> None:
        """Load initial photo from file dialog."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Выбрать начальное фото", str(self.project_root / "01_InitialPhoto"),
            "Images (*.jpg *.jpeg *.png *.tiff *.bmp *.webp);;All Files (*)"
        )
        if filepath:
            self._set_initial_photo_from_path(Path(filepath))

    def _set_initial_photo_from_path(self, filepath: Path) -> None:
        """Set initial photo from path, copying to project folder if needed."""
        if not self.project:
            self.new_project()

        # Copy to project folder if external
        try:
            dest_folder = self.project_root / "01_InitialPhoto"
            dest_folder.mkdir(parents=True, exist_ok=True)
            dest_path = dest_folder / filepath.name

            if filepath != dest_path:
                import shutil
                shutil.copy2(filepath, dest_path)
                filepath = dest_path

            # Add to asset manager
            asset = self.asset_manager.add_asset(AssetCategory.INITIAL_PHOTO, filepath)
            if asset.is_valid:
                self.project.initial_photo = f"01_InitialPhoto/{filepath.name}"
                # Clear restored photo when setting new initial photo
                self.project.restored_photo = ""
                self.initial_gallery.refresh()
                self._update_composite()
                self._push_history()
                self.status_bar.showMessage(f"Загружено начальное фото: {filepath.name}")
            else:
                QMessageBox.warning(self, "Ошибка", f"Неверный формат файла: {asset.validation_error}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить фото:\n{e}")

    def _set_initial_photo(self, name: str) -> None:
        """Set initial photo from asset gallery."""
        path = self.asset_manager.get_asset_path(AssetCategory.INITIAL_PHOTO, name)
        if path:
            self.project.initial_photo = f"01_InitialPhoto/{path.name}"
            # Clear restored photo when setting new initial photo
            self.project.restored_photo = ""
            self._update_composite()
            self.status_bar.showMessage(f"Начальное фото установлено: {name}")

    def _load_restored_photo(self) -> None:
        """Load restored photo from file dialog."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Выбрать восстановленное фото", str(self.project_root / "02_RestoredPhoto"),
            "Images (*.jpg *.jpeg *.png *.tiff *.bmp *.webp);;All Files (*)"
        )
        if filepath:
            self._set_restored_photo_from_path(Path(filepath))

    def _set_restored_photo_from_path(self, filepath: Path) -> None:
        """Set restored photo from path."""
        if not self.project:
            self.new_project()

        try:
            dest_folder = self.project_root / "02_RestoredPhoto"
            dest_folder.mkdir(parents=True, exist_ok=True)
            dest_path = dest_folder / filepath.name

            if filepath != dest_path:
                import shutil
                shutil.copy2(filepath, dest_path)
                filepath = dest_path

            asset = self.asset_manager.add_asset(AssetCategory.RESTORED_PHOTO, filepath)
            if asset.is_valid:
                self.project.restored_photo = f"02_RestoredPhoto/{filepath.name}"
                self.restored_gallery.refresh()
                self._update_composite()
                self._push_history()
                self.status_bar.showMessage(f"Загружено восстановленное фото: {filepath.name}")
            else:
                QMessageBox.warning(self, "Ошибка", f"Неверный формат файла: {asset.validation_error}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить фото:\n{e}")

    def _set_restored_photo(self, name: str) -> None:
        """Set restored photo from asset gallery."""
        path = self.asset_manager.get_asset_path(AssetCategory.RESTORED_PHOTO, name)
        if path:
            self.project.restored_photo = f"02_RestoredPhoto/{path.name}"
            self._update_composite()
            self.status_bar.showMessage(f"Восстановленное фото установлено: {name}")

    def _set_enhanced_quality_photo(self, name: str) -> None:
        """Set enhanced quality photo as restored photo."""
        path = self.asset_manager.get_asset_path(AssetCategory.ENHANCED_QUALITY_PHOTO, name)
        if path:
            self.project.restored_photo = f"08_EnhancedQuality/{path.name}"
            self._update_composite()
            self.status_bar.showMessage(f"Улучшенное фото установлено: {name}")

    def _on_asset_double_clicked(self, name: str, category: AssetCategory) -> None:
        """Handle asset double-click in gallery."""
        if not self.project:
            self.new_project()

        path = self.asset_manager.get_asset_path(category, name)
        if not path:
            return

        rel_path = f"{ASSET_CATEGORIES[category.value]}/{path.name}"

        if category == AssetCategory.FRAMES:
            self.project.frame = rel_path
        elif category == AssetCategory.BACKGROUNDS:
            self.project.background = rel_path
        elif category == AssetCategory.COSTUMES:
            self.project.costume = rel_path
        elif category == AssetCategory.FONTS:
            self.project.font = rel_path
            self.project.text_overlay.font_path = rel_path
        elif category == AssetCategory.ENHANCED_QUALITY_PHOTO:
            # Use enhanced quality photo as restored photo
            self.project.restored_photo = rel_path

        self._update_composite()
        self._push_history()
        self.status_bar.showMessage(f"Применен {category.value}: {name}")

    def _on_adjustment_changed(self, value: int) -> None:
        """Handle adjustment slider change."""
        if self.project:
            self._update_composite()

    def _reset_adjustments(self) -> None:
        """Reset all adjustment sliders to zero."""
        self.brightness_slider.set_value(0)
        self.contrast_slider.set_value(0)
        self.sharpness_slider.set_value(0)
        self.saturation_slider.set_value(0)
        self.temperature_slider.set_value(0)
        self.vignette_slider.set_value(0)
        if self.project:
            self.project.adjustments = Adjustments()
            self._update_composite()
        self.status_bar.showMessage("Коррекция сброшена")

    def _on_ml_fidelity_changed(self, value: int) -> None:
        """Handle ML fidelity slider change."""
        if self.project:
            self.project.ml_settings.fidelity = value / 100.0

    def _on_costume_transform_changed(self, value: int) -> None:
        """Handle costume transform slider change."""
        if self.project and self.project.costume:
            # Update costume transform in project
            transform = self.project.costume_transform
            if self.sender() == self.costume_scale_slider:
                transform.scale = value / 100.0
            elif self.sender() == self.costume_rotation_slider:
                transform.rotation = value
            # Apply to image viewer
            self.image_viewer._apply_transform_to_costume(transform)
            self._update_composite()

    def _on_costume_opacity_changed(self, value: int) -> None:
        """Handle costume opacity slider change."""
        if self.project:
            self.compositor.set_layer_opacity("costume", value / 100.0)
            self._update_composite()

    def _on_costume_transform_from_viewer(self, transform: Transform) -> None:
        """Handle costume transform changed from viewer (drag/resize)."""
        if self.project:
            self.project.costume_transform = transform
            # Update sliders to match
            self.costume_scale_slider.set_value(int(transform.scale * 100))
            self.costume_rotation_slider.set_value(int(transform.rotation))
            self._update_composite()

    def _on_text_position_changed(self, position: Point) -> None:
        """Handle text position changed from viewer (drag to move)."""
        if self.project:
            self.project.text_overlay.position = position
            self._update_composite()

    def _on_layer_visibility_changed(self, layer_name: str, visible: bool) -> None:
        """Handle layer visibility change."""
        if self.project:
            self.project.layer_visibility[layer_name] = visible
            self.compositor.set_layer_visibility(layer_name, visible)
            self._update_composite()

    def _on_layer_order_changed(self, order: List[str]) -> None:
        """Handle layer order change."""
        # Would need to update compositor layer order
        pass

    def _on_zoom_changed(self, zoom: float) -> None:
        """Handle zoom change."""
        self.zoom_label.setText(f"{int(zoom * 100)}%")

    def _on_item_selected(self, item) -> None:
        """Handle overlay item selection."""
        # Enable/disable text edit buttons based on selection
        pass

    def _restore_photo(self) -> None:
        """Restore photo using ML."""
        if not self.project or not self.project.initial_photo:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите начальное фото")
            return

        input_path = self.project_root / self.project.initial_photo
        if not input_path.exists():
            QMessageBox.warning(self, "Предупреждение", "Файл исходного фото не найден")
            return

        # Prepare output path
        output_folder = self.project_root / "02_RestoredPhoto"
        output_folder.mkdir(parents=True, exist_ok=True)
        stem = input_path.stem
        output_path = output_folder / f"{stem}_restored.png"

        # Create task
        task = RestorationTask(
            input_path=input_path,
            output_path=output_path,
            model=RestorationModel(self.ml_model_combo.currentText()),
            fidelity=self.ml_fidelity_slider.value() / 100.0,
            upscale=2,
            task_id=f"restore_{int(datetime.now().timestamp())}"
        )

        # Show progress dialog
        progress = ProgressDialog("Восстановление фото", "Обработка...", self)
        progress.set_progress(0, "Инициализация...")

        def on_progress(value: int, detail: str):
            progress.set_progress(value, detail)

        def on_finished(success: str, output: str, error: str):
            progress.accept()
            if success:
                # Add to assets
                asset = self.asset_manager.add_asset(AssetCategory.RESTORED_PHOTO, Path(output))
                if asset.is_valid:
                    self.project.restored_photo = f"02_RestoredPhoto/{Path(output).name}"
                    self.restored_gallery.refresh()
                    self._update_composite()
                    QMessageBox.information(self, "Успех", "Фото успешно восстановлено")
                else:
                    QMessageBox.warning(self, "Предупреждение", f"Результат невалиден: {asset.validation_error}")
            else:
                QMessageBox.critical(self, "Ошибка", f"Восстановление не удалось:\n{error}")

        # Run in worker thread
        self.worker = RestorationWorker(self.ml_pipeline, task)
        self.worker.progress.connect(on_progress)
        self.worker.finished.connect(on_finished)
        self.worker.start()
        progress.exec()

    def _enhance_quality(self) -> None:
        """Enhance photo quality using AI super-resolution + denoising + sharpening."""
        if not self.project or not (self.project.initial_photo or self.project.restored_photo):
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите фото для улучшения")
            return

        # Use restored photo if available, otherwise initial
        photo_path = None
        if self.project.restored_photo:
            photo_path = self.project_root / self.project.restored_photo
        elif self.project.initial_photo:
            photo_path = self.project_root / self.project.initial_photo

        if not photo_path or not photo_path.exists():
            QMessageBox.warning(self, "Предупреждение", "Файл фото не найден")
            return

        # Prepare output path
        output_folder = self.project_root / "08_EnhancedQuality"
        output_folder.mkdir(parents=True, exist_ok=True)
        stem = photo_path.stem
        output_path = output_folder / f"{stem}_enhanced.png"

        # Get settings from project
        settings = QESettings(
            model=self.project.quality_enhance_settings.model,
            denoise_strength=self.project.quality_enhance_settings.denoise_strength,
            sharpen_strength=self.project.quality_enhance_settings.sharpen_strength,
            face_enhance=self.project.quality_enhance_settings.face_enhance
        )

        # Show progress dialog
        progress = ProgressDialog("Улучшение качества", "Обработка...", self)
        progress.set_progress(0, "Инициализация...")

        def on_progress(value: int, detail: str):
            progress.set_progress(value, detail)

        def on_finished(success: bool, result_image: Image.Image = None, error: str = ""):
            progress.accept()
            if success and result_image:
                # Save result
                try:
                    result_image.save(output_path)
                    # Add to assets
                    asset = self.asset_manager.add_asset(AssetCategory.ENHANCED_QUALITY_PHOTO, output_path)
                    if asset.is_valid:
                        # Optionally set as restored photo
                        reply = QMessageBox.question(
                            self, "Успех",
                            "Качество фото улучшено. Использовать как восстановленное фото?",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                        )
                        if reply == QMessageBox.StandardButton.Yes:
                            self.project.restored_photo = f"08_EnhancedQuality/{output_path.name}"
                            self.restored_gallery.refresh()
                            self._update_composite()
                        self.enhanced_gallery.refresh()
                        QMessageBox.information(self, "Успех", "Качество фото успешно улучшено")
                    else:
                        QMessageBox.warning(self, "Предупреждение", f"Результат невалиден: {asset.validation_error}")
                except Exception as e:
                    QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить результат:\n{e}")
            else:
                QMessageBox.critical(self, "Ошибка", f"Улучшение не удалось:\n{error}")

        # Run in worker thread
        worker = QualityEnhancementWorker(photo_path, output_path, settings, on_progress)
        worker.finished.connect(on_finished)
        worker.start()
        progress.exec()

    def _on_quality_model_changed(self, model_name: str) -> None:
        """Handle quality model change."""
        if self.project:
            self.project.quality_enhance_settings.model = model_name

    def _on_quality_settings_changed(self, value) -> None:
        """Handle quality settings change."""
        if self.project:
            sender = self.sender()
            if sender == self.quality_denoise_slider:
                self.project.quality_enhance_settings.denoise_strength = value / 100.0
            elif sender == self.quality_sharpen_slider:
                self.project.quality_enhance_settings.sharpen_strength = value / 100.0
            elif sender == self.quality_face_enhance_cb:
                self.project.quality_enhance_settings.face_enhance = value
            self._update_composite()

    def _set_enhanced_quality_photo(self, name: str) -> None:
        """Set enhanced quality photo from asset gallery."""
        path = self.asset_manager.get_asset_path(AssetCategory.ENHANCED_QUALITY_PHOTO, name)
        if path:
            self.project.enhanced_quality_photo = f"08_EnhancedQuality/{path.name}"
            self._update_composite()
            self.status_bar.showMessage(f"Улучшенное фото установлено: {name}")

    def _set_enhanced_quality_photo_from_path(self, filepath: Path) -> None:
        """Set enhanced quality photo from path."""
        if not self.project:
            self.new_project()

        try:
            dest_folder = self.project_root / "08_EnhancedQuality"
            dest_folder.mkdir(parents=True, exist_ok=True)
            dest_path = dest_folder / filepath.name

            if filepath != dest_path:
                import shutil
                shutil.copy2(filepath, dest_path)
                filepath = dest_path

            asset = self.asset_manager.add_asset(AssetCategory.ENHANCED_QUALITY_PHOTO, filepath)
            if asset.is_valid:
                self.project.enhanced_quality_photo = f"08_EnhancedQuality/{filepath.name}"
                self.enhanced_gallery.refresh()
                self._update_composite()
                self._push_history()
                self.status_bar.showMessage(f"Загружено улучшенное фото: {filepath.name}")
            else:
                QMessageBox.warning(self, "Ошибка", f"Неверный формат файла: {asset.validation_error}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить фото:\n{e}")

    def _on_swallow_enabled_changed(self, enabled: bool) -> None:
        """Handle swallow mode enable/disable."""
        self.swallow_apply_btn.setEnabled(enabled)
        if self.project:
            self.project.swallow_mode_settings.enabled = enabled
            if enabled:
                self._apply_swallow_mode()

    def _on_swallow_settings_changed(self, value) -> None:
        """Handle swallow mode settings change."""
        if not self.project:
            return

        sender = self.sender()
        if sender == self.swallow_beard_cb:
            self.project.swallow_mode_settings.remove_beard = value
        elif sender == self.swallow_warts_cb:
            self.project.swallow_mode_settings.remove_warts = value
        elif sender == self.swallow_moles_cb:
            self.project.swallow_mode_settings.remove_moles = value
        elif sender == self.swallow_acne_cb:
            self.project.swallow_mode_settings.remove_acne = value
        elif sender == self.swallow_transition_slider:
            self.project.swallow_mode_settings.smooth_transition = value / 100.0
        elif sender == self.swallow_texture_slider:
            self.project.swallow_mode_settings.preserve_skin_texture = value / 100.0
        elif sender == self.swallow_sensitivity_slider:
            self.project.swallow_mode_settings.detection_sensitivity = value / 100.0

        if self.project.swallow_mode_settings.enabled:
            self._apply_swallow_mode()

    def _apply_swallow_mode(self) -> None:
        """Apply swallow mode to current photo."""
        if not self.project or not (self.project.initial_photo or self.project.restored_photo):
            return

        if not self.project.swallow_mode_settings.enabled:
            return

        # Use restored photo if available, otherwise initial
        photo_path = None
        if self.project.restored_photo:
            photo_path = self.project_root / self.project.restored_photo
        elif self.project.initial_photo:
            photo_path = self.project_root / self.project.initial_photo

        if not photo_path or not photo_path.exists():
            return

        # Prepare output path
        output_folder = self.project_root / "02_RestoredPhoto"
        output_folder.mkdir(parents=True, exist_ok=True)
        stem = photo_path.stem
        output_path = output_folder / f"{stem}_swallow.png"

        # Show progress dialog
        progress = ProgressDialog("Режим ластки", "Обработка...", self)
        progress.set_progress(0, "Инициализация...")

        def on_progress(value: int, detail: str):
            progress.set_progress(value, detail)

        def on_finished(success: bool, result_image: Image.Image = None, error: str = ""):
            progress.accept()
            if success and result_image:
                try:
                    result_image.save(output_path)
                    asset = self.asset_manager.add_asset(AssetCategory.RESTORED_PHOTO, output_path)
                    if asset.is_valid:
                        self.project.restored_photo = f"02_RestoredPhoto/{output_path.name}"
                        self.restored_gallery.refresh()
                        self._update_composite()
                        QMessageBox.information(self, "Успех", "Режим ластки применен успешно")
                    else:
                        QMessageBox.warning(self, "Предупреждение", f"Результат невалиден: {asset.validation_error}")
                except Exception as e:
                    QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить результат:\n{e}")
            else:
                QMessageBox.critical(self, "Ошибка", f"Обработка не удалась:\n{error}")

        # Run in worker thread
        worker = SwallowModeWorker(photo_path, output_path,
                                   self.project.swallow_mode_settings, on_progress)
        worker.finished.connect(on_finished)
        worker.start()
        progress.exec()

    def _add_text_overlay(self) -> None:
        """Add text overlay."""
        if not self.project:
            self.new_project()

        dialog = TextEditDialog(self.project.text_overlay, self.project.font, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            overlay = dialog.get_result()
            self.project.text_overlay = overlay
            self.project.text_overlay.visible = True
            self.edit_text_btn.setEnabled(True)
            self.remove_text_btn.setEnabled(True)
            self._update_composite()
            self._push_history()

    def _edit_text_overlay(self) -> None:
        """Edit existing text overlay."""
        if not self.project or not self.project.text_overlay.content:
            return

        dialog = TextEditDialog(self.project.text_overlay, self.project.font, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            overlay = dialog.get_result()
            self.project.text_overlay = overlay
            self._update_composite()
            self._push_history()

    def _remove_text_overlay(self) -> None:
        """Remove text overlay."""
        if self.project:
            self.project.text_overlay = TextOverlay()
            self.edit_text_btn.setEnabled(False)
            self.remove_text_btn.setEnabled(False)
            self._update_composite()
            self._push_history()

    def _flip_costume(self, direction: str) -> None:
        """Flip costume horizontally or vertically."""
        if not self.project or not self.project.costume:
            return

        costume_path = self.project_root / self.project.costume
        if not costume_path.exists():
            return

        img = load_image_safe(costume_path)
        if img:
            if direction == "horizontal":
                img = ImageOps.mirror(img)
            elif direction == "vertical":
                img = ImageOps.flip(img)

            # Save flipped version
            flipped_path = costume_path.with_stem(f"{costume_path.stem}_flipped")
            img.save(flipped_path)

            # Update project
            self.project.costume = str(flipped_path.relative_to(self.project_root))
            self._update_composite()
            self._push_history()
            self.status_bar.showMessage(f"Костюм отражен по {direction}")

    def _remove_costume(self) -> None:
        """Remove costume overlay."""
        if self.project:
            self.project.costume = ""
            self.project.costume_transform = Transform()
            self.remove_costume_btn.setEnabled(False)
            self.image_viewer.clear_costume()
            self._update_composite()
            self._push_history()

    def _toggle_crop_tool(self, checked: bool) -> None:
        """Toggle crop tool mode."""
        if checked:
            self.image_viewer.set_tool_mode("crop")
            self.status_bar.showMessage("Режим обрезки: кликните и перетащите для выбора области")
        else:
            self.image_viewer.cancel_crop()
            self.image_viewer.set_tool_mode("select")
            self.status_bar.showMessage("Готово")

    def _set_tool_mode(self, mode: str) -> None:
        """Set the current tool mode."""
        self.image_viewer.set_tool_mode(mode)
        if mode == "select":
            self.crop_action.setChecked(False)
        self.status_bar.showMessage(f"Инструмент: {mode}")

    def _rotate_photo(self, angle: int) -> None:
        """Rotate the main photo by angle degrees."""
        if not self.project or not (self.project.initial_photo or self.project.restored_photo):
            QMessageBox.warning(self, "Предупреждение", "Нет загруженного фото для поворота")
            return

        photo_path = None
        if self.project.restored_photo:
            photo_path = self.project_root / self.project.restored_photo
        elif self.project.initial_photo:
            photo_path = self.project_root / self.project.initial_photo

        if photo_path and photo_path.exists():
            img = load_image_safe(photo_path)
            if img:
                img = img.rotate(angle, expand=True)
                img.save(photo_path)
                self._update_composite()
                self._push_history()
                self.status_bar.showMessage(f"Фото повернуто на {angle}°")

    def _flip_photo(self, direction: str) -> None:
        """Flip the main photo horizontally or vertically."""
        if not self.project or not (self.project.initial_photo or self.project.restored_photo):
            QMessageBox.warning(self, "Предупреждение", "Нет загруженного фото для отражения")
            return

        photo_path = None
        if self.project.restored_photo:
            photo_path = self.project_root / self.project.restored_photo
        elif self.project.initial_photo:
            photo_path = self.project_root / self.project.initial_photo

        if photo_path and photo_path.exists():
            img = load_image_safe(photo_path)
            if img:
                if direction == "horizontal":
                    img = ImageOps.mirror(img)
                elif direction == "vertical":
                    img = ImageOps.flip(img)
                img.save(photo_path)
                self._update_composite()
                self._push_history()
                self.status_bar.showMessage(f"Фото отражено по {direction}")

    def _delete_selected(self) -> None:
        """Delete selected overlay item."""
        # Could be used for costume/text items
        pass

    def _push_history(self) -> None:
        """Push current project state to history stack."""
        if not self._history_enabled or not self.project:
            return

        # Serialize current project state
        state = self.project.to_dict()
        # Remove any history after current index (for redo)
        self._history = self._history[:self._history_index + 1]
        self._history.append(state)
        self._history_index = len(self._history) - 1

        # Limit history size
        if len(self._history) > self._max_history:
            self._history.pop(0)
            self._history_index -= 1

    def _restore_history(self, index: int) -> None:
        """Restore project state from history index."""
        if not self.project or index < 0 or index >= len(self._history):
            return

        self._history_enabled = False  # Prevent pushing to history during restore
        try:
            state = self._history[index]
            self.project = Project.from_dict(state)
            self._apply_project_to_ui()
            self._update_composite()
            self._history_index = index
        finally:
            self._history_enabled = True

    def undo(self) -> None:
        """Undo last action."""
        if self._history_index > 0:
            self._history_index -= 1
            self._restore_history(self._history_index)
            self.status_bar.showMessage("Отменено")

    def redo(self) -> None:
        """Redo last undone action."""
        if self._history_index < len(self._history) - 1:
            self._history_index += 1
            self._restore_history(self._history_index)
            self.status_bar.showMessage("Повторено")

    
    def _batch_process(self) -> None:
        """Batch process multiple photos."""
        QMessageBox.information(self, "В разработке", "Пакетная обработка будет доступна в следующей версии")

    def _export_settings(self) -> None:
        """Show export settings dialog."""
        QMessageBox.information(self, "В разработке", "Настройки экспорта будут доступны в следующей версии")

    def _show_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(self, "О Photo Retouch Pro",
                          "Photo Retouch Pro v1.0\n\n"
                          "Профессиональный инструмент для ретуши фотографий\n"
                          "с поддержкой ML-восстановления.\n\n"
                          "© 2026 Photo Retouch Team")

    def _confirm_discard(self) -> bool:
        """Confirm discarding unsaved changes."""
        reply = QMessageBox.question(
            self, "Несохраненные изменения",
            "Есть несохраненные изменения. Отменить?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        return reply == QMessageBox.StandardButton.Yes

    def _auto_save(self) -> None:
        """Auto-save draft."""
        if self._dirty and self.project:
            try:
                draft_path = self.project_root / "draft.proj"
                self._gather_project_from_ui()
                self.project.save(draft_path)
                logger.debug("Auto-saved draft")
            except Exception as e:
                logger.error(f"Auto-save failed: {e}")

    def _load_settings(self) -> None:
        """Load application settings."""
        settings = QSettings("PhotoRetouch", "PhotoRetouchPro")
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        state = settings.value("windowState")
        if state:
            self.restoreState(state)

    def _save_settings(self) -> None:
        """Save application settings."""
        settings = QSettings("PhotoRetouch", "PhotoRetouchPro")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())

    def closeEvent(self, event) -> None:
        """Handle close event."""
        if self._dirty and not self._confirm_discard():
            event.ignore()
            return

        self._auto_save_timer.stop()
        self._save_settings()
        event.accept()


