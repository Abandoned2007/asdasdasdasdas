#!/usr/bin/env python
"""Run the PyQt GUI version of Photo Retouch."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from PyQt6.QtWidgets import QApplication
from photo_retouch.gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Photo Retouch Pro")
    app.setOrganizationName("PhotoRetouch")

    # Get project root
    project_root = Path(__file__).parent / "PhotoRetouchProject"

    window = MainWindow(project_root)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()