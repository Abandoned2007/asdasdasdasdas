#!/usr/bin/env python
"""Test core modules without GUI."""
import sys
from pathlib import Path
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_models():
    """Test core models."""
    print("Testing core models...")
    from photo_retouch.core.models import (
        Project, Adjustments, Transform, Point, TextOverlay, TextStyle,
        MLSettings, CostumeItem, AssetCategory, ASSET_CATEGORIES
    )

    # Test Project
    project = Project()
    project.initial_photo = "01_InitialPhoto/test.jpg"
    project.adjustments = Adjustments(brightness=10, contrast=5, sharpness=20)
    project.ml_settings = MLSettings(model="GFPGANv1.4", fidelity=0.75)

    # Test serialization
    json_str = project.to_json()
    print(f"  Project JSON: {len(json_str)} chars")

    # Test deserialization
    project2 = Project.from_json(json_str)
    assert project2.initial_photo == project.initial_photo
    assert project2.adjustments.brightness == 10
    print("  ✓ Project serialization works")

    # Test Adjustments clamping
    adj = Adjustments(brightness=150, contrast=-150, sharpness=200)
    clamped = adj.clamp()
    assert clamped.brightness == 100
    assert clamped.contrast == -100
    assert clamped.sharpness == 100
    print("  ✓ Adjustments clamping works")

    # Test Transform
    transform = Transform(position=Point(100, 200), scale=1.5, rotation=45)
    d = transform.to_dict()
    transform2 = Transform.from_dict(d)
    assert transform2.position.x == 100
    assert transform2.scale == 1.5
    print("  ✓ Transform works")

    # Test TextOverlay
    overlay = TextOverlay(content="Test", position=Point(50, 50))
    d = overlay.to_dict()
    overlay2 = TextOverlay.from_dict(d)
    assert overlay2.content == "Test"
    print("  ✓ TextOverlay works")

    print("✓ All model tests passed\n")


def test_asset_manager():
    """Test asset manager."""
    print("Testing asset manager...")
    from photo_retouch.assets.manager import AssetManager, AssetCategory

    project_root = Path.cwd() / "PhotoRetouchProject"
    manager = AssetManager(project_root)

    # Scan all
    result = manager.scan_all()
    print(f"  Scanned {result.total_files} files in {result.scanned_folders} folders")

    # Test getting assets
    for category in AssetCategory:
        assets = manager.get_assets(category)
        print(f"  {category.value}: {len(assets)} assets")

    print("✓ Asset manager tests passed\n")


def test_processing_engine():
    """Test processing engine."""
    print("Testing processing engine...")
    from photo_retouch.processing.engine import (
        ImageProcessor, Compositor, TextRenderer, load_image_safe
    )
    from photo_retouch.core.models import Adjustments, Transform, Point, TextOverlay, TextStyle
    from PIL import Image

    # Create test image
    test_img = Image.new("RGBA", (100, 100), (255, 0, 0, 255))

    # Test adjustments
    adj = Adjustments(brightness=20, contrast=10, sharpness=30)
    adjusted = ImageProcessor.apply_adjustments(test_img, adj)
    assert adjusted.size == test_img.size
    print("  ✓ Adjustments work")

    # Test compositor
    compositor = Compositor((200, 200))
    compositor.set_layer_image("background", test_img)
    composite = compositor.composite()
    assert composite.size == (200, 200)
    print("  ✓ Compositor works")

    # Test text rendering
    overlay = TextOverlay(content="Test", position=Point(100, 100))
    text_img = TextRenderer.render_text(overlay, (200, 200))
    assert text_img.size == (200, 200)
    print("  ✓ Text rendering works")

    print("✓ Processing engine tests passed\n")


def test_ml_pipeline():
    """Test ML pipeline (mock)."""
    print("Testing ML pipeline (mock)...")
    from photo_retouch.ml.pipeline import MockMLPipeline, RestorationTask, RestorationModel
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / 'test.jpg'
        output_path = Path(tmpdir) / 'output.jpg'

        test_img = Image.new('RGB', (100, 100), (255, 0, 0))
        test_img.save(input_path)

        # Test mock pipeline
        pipeline = MockMLPipeline()
        task = RestorationTask(
            input_path=input_path,
            output_path=output_path,
            model=RestorationModel.GFPGAN_V14,
            fidelity=0.75,
            upscale=2,
            task_id='test_1'
        )
        result = pipeline.restore_sync(task)
        assert result.success, f"Restoration failed: {result.error}"
        assert output_path.exists(), "Output file not created"

        out_img = Image.open(output_path)
        assert out_img.size == (200, 200), f"Expected 200x200, got {out_img.size}"
        out_img.close()
        print("  ✓ Mock ML pipeline works")

    print("✓ ML pipeline tests passed\n")


def test_gui_imports():
    """Test GUI component imports."""
    print("Testing GUI imports...")
    from photo_retouch.gui.main_window import MainWindow
    from photo_retouch.gui.widgets import (
        ImageViewer, AssetGallery, AdjustmentSlider,
        LayerControlWidget, TextEditDialog, ProgressDialog
    )
    print("  ✓ All GUI imports work")
    print("✓ GUI import tests passed\n")


def main():
    """Run all tests."""
    print("=" * 50)
    print("Photo Retouch Pro - Core Module Tests")
    print("=" * 50 + "\n")

    test_models()
    test_asset_manager()
    test_processing_engine()
    test_ml_pipeline()
    test_gui_imports()

    print("=" * 50)
    print("ALL TESTS PASSED ✓")
    print("=" * 50)


if __name__ == "__main__":
    main()