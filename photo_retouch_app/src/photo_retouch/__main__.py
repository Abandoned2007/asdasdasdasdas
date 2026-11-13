"""
Main entry point for Photo Retouch application.
"""
import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from photo_retouch.gui.main_window import MainWindow


def setup_logging() -> None:
    """Configure application logging."""
    log_dir = Path.home() / ".photo_retouch" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "photo_retouch.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Reduce noise from some libraries
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("PyQt6").setLevel(logging.WARNING)


def main() -> int:
    """Main application entry point."""
    setup_logging()
    logger = logging.getLogger(__name__)

    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Photo Retouch Pro")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("PhotoRetouch")
    app.setOrganizationDomain("photoretouch.local")

    # Set application style
    app.setStyle("Fusion")

    # Determine project root
    project_root = Path.cwd() / "PhotoRetouchProject"
    project_root.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting Photo Retouch Pro with project root: {project_root}")

    # Create and show main window
    window = MainWindow(project_root)
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())