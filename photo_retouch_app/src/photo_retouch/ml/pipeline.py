"""
ML Pipeline for photo restoration (GFPGAN, CodeFormer, Real-ESRGAN).
Runs in a separate process for isolation.
"""
import logging
import subprocess
import sys
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from enum import Enum
import threading
import queue

logger = logging.getLogger(__name__)


class RestorationModel(Enum):
    GFPGAN_V14 = "GFPGANv1.4"
    CODEFORMER = "CodeFormer"
    REAL_ESRGAN = "Real-ESRGAN"


@dataclass
class RestorationTask:
    """A photo restoration task."""
    input_path: Path
    output_path: Path
    model: RestorationModel = RestorationModel.GFPGAN_V14
    fidelity: float = 0.75
    upscale: int = 2
    task_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_path": str(self.input_path),
            "output_path": str(self.output_path),
            "model": self.model.value,
            "fidelity": self.fidelity,
            "upscale": self.upscale,
            "task_id": self.task_id
        }


@dataclass
class RestorationResult:
    """Result of a restoration task."""
    success: bool = False
    output_path: Optional[Path] = None
    error: str = ""
    processing_time: float = 0.0
    task_id: str = ""


class MLPipeline:
    """
    ML Pipeline for photo restoration.
    Uses subprocess to run Python restoration scripts for process isolation.
    """

    def __init__(self, models_dir: Path = None, python_exe: str = None):
        self.models_dir = models_dir or Path.home() / ".photo_retouch" / "models"
        self.python_exe = python_exe or sys.executable
        self._process: Optional[subprocess.Popen] = None
        self._task_queue: queue.Queue = queue.Queue()
        self._result_callbacks: Dict[str, Callable[[RestorationResult], None]] = {}
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._available_models: Dict[RestorationModel, bool] = {}

    def check_models(self) -> Dict[RestorationModel, bool]:
        """Check which models are available."""
        self._available_models = {}
        for model in RestorationModel:
            model_path = self.models_dir / model.value
            self._available_models[model] = model_path.exists()
        return self._available_models

    def download_model(self, model: RestorationModel, progress_callback: Callable[[float], None] = None) -> bool:
        """Download a model (placeholder - would use actual download logic)."""
        logger.info(f"Downloading model {model.value}...")
        # In real implementation, this would download from GitHub/HuggingFace
        # For now, just create a placeholder
        model_path = self.models_dir / model.value
        model_path.mkdir(parents=True, exist_ok=True)
        (model_path / "README.txt").write_text(f"Model {model.value} placeholder")
        self._available_models[model] = True
        return True

    def start_worker(self) -> None:
        """Start the background worker thread."""
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def stop_worker(self) -> None:
        """Stop the background worker thread."""
        self._running = False
        self._task_queue.put(None)  # Sentinel
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)

    def _worker_loop(self) -> None:
        """Background worker loop."""
        while self._running:
            try:
                task = self._task_queue.get(timeout=1.0)
                if task is None:
                    break
                self._process_task(task)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")

    def _process_task(self, task: RestorationTask) -> None:
        """Process a single restoration task."""
        start_time = time.time()
        result = RestorationResult(task_id=task.task_id)

        try:
            # Run restoration via subprocess
            script_path = Path(__file__).parent / "restore_script.py"
            if not script_path.exists():
                self._create_restore_script(script_path)

            cmd = [
                self.python_exe, str(script_path),
                "--input", str(task.input_path),
                "--output", str(task.output_path),
                "--model", task.model.value,
                "--fidelity", str(task.fidelity),
                "--upscale", str(task.upscale)
            ]

            logger.info(f"Running restoration: {task.model.value} on {task.input_path}")

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=str(Path(__file__).parent)
            )

            if proc.returncode == 0:
                result.success = True
                result.output_path = task.output_path
                logger.info(f"Restoration completed in {time.time() - start_time:.1f}s")
            else:
                result.success = False
                result.error = proc.stderr or proc.stdout or "Unknown error"
                logger.error(f"Restoration failed: {result.error}")

        except subprocess.TimeoutExpired:
            result.success = False
            result.error = "Restoration timed out (5 minutes)"
        except Exception as e:
            result.success = False
            result.error = str(e)
            logger.error(f"Restoration error: {e}")

        result.processing_time = time.time() - start_time

        # Call callback if registered
        if task.task_id in self._result_callbacks:
            try:
                self._result_callbacks[task.task_id](result)
            except Exception as e:
                logger.error(f"Callback error: {e}")
            finally:
                del self._result_callbacks[task.task_id]

    def restore_async(self, task: RestorationTask,
                      callback: Callable[[RestorationResult], None]) -> str:
        """Queue a restoration task for async processing."""
        if not task.task_id:
            task.task_id = f"task_{int(time.time() * 1000)}"

        self._result_callbacks[task.task_id] = callback
        self._task_queue.put(task)
        return task.task_id

    def restore_sync(self, task: RestorationTask) -> RestorationResult:
        """Run restoration synchronously (blocking)."""
        result = RestorationResult(task_id=task.task_id or f"sync_{int(time.time() * 1000)}")
        done = threading.Event()

        def callback(r: RestorationResult):
            result.success = r.success
            result.output_path = r.output_path
            result.error = r.error
            result.processing_time = r.processing_time
            done.set()

        self.restore_async(task, callback)
        done.wait(timeout=300)
        return result

    def _create_restore_script(self, script_path: Path) -> None:
        """Create the restoration script that runs in subprocess."""
        script_content = '''
import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def restore_gfpgan(input_path, output_path, fidelity, upscale):
    """Restore using GFPGAN."""
    try:
        # Try to import GFPGAN
        sys.path.insert(0, str(Path.home() / ".photo_retouch" / "models" / "GFPGANv1.4"))
        from gfpgan import GFPGANer
        import cv2

        restorer = GFPGANer(
            model_path=str(Path.home() / ".photo_retouch" / "models" / "GFPGANv1.4" / "GFPGANv1.4.pth"),
            upscale=upscale,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None
        )

        img = cv2.imread(str(input_path))
        if img is None:
            raise ValueError(f"Cannot read image: {input_path}")

        _, _, restored = restorer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)

        cv2.imwrite(str(output_path), restored)
        return True
    except ImportError:
        logger.warning("GFPGAN not installed, using fallback")
        return fallback_restore(input_path, output_path, upscale)
    except Exception as e:
        logger.error(f"GFPGAN error: {e}")
        return False

def restore_codeformer(input_path, output_path, fidelity, upscale):
    """Restore using CodeFormer."""
    try:
        sys.path.insert(0, str(Path.home() / ".photo_retouch" / "models" / "CodeFormer"))
        from basicsr.utils import imwrite
        import cv2
        import torch

        # CodeFormer inference would go here
        # For now, use fallback
        return fallback_restore(input_path, output_path, upscale)
    except Exception as e:
        logger.error(f"CodeFormer error: {e}")
        return fallback_restore(input_path, output_path, upscale)

def restore_realesrgan(input_path, output_path, fidelity, upscale):
    """Restore using Real-ESRGAN."""
    try:
        from realesrgan import RealESRGANer
        import cv2

        restorer = RealESRGANer(
            scale=upscale,
            model_path=str(Path.home() / ".photo_retouch" / "models" / "Real-ESRGAN" / "RealESRGAN_x4plus.pth"),
            model=None,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False
        )

        img = cv2.imread(str(input_path))
        if img is None:
            raise ValueError(f"Cannot read image: {input_path}")

        output, _ = restorer.enhance(img, outscale=upscale)
        cv2.imwrite(str(output_path), output)
        return True
    except Exception as e:
        logger.error(f"Real-ESRGAN error: {e}")
        return fallback_restore(input_path, output_path, upscale)

def fallback_restore(input_path, output_path, upscale):
    """Fallback: simple upscale with OpenCV."""
    import cv2
    import numpy as np

    img = cv2.imread(str(input_path))
    if img is None:
        return False

    # Simple upscale using Lanczos
    h, w = img.shape[:2]
    new_h, new_w = h * upscale, w * upscale
    upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Apply some basic enhancement
    # Denoise
    denoised = cv2.fastNlMeansDenoisingColored(upscaled, None, 10, 10, 7, 21)

    # Sharpen
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    cv2.imwrite(str(output_path), sharpened)
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="GFPGANv1.4")
    parser.add_argument("--fidelity", type=float, default=0.75)
    parser.add_argument("--upscale", type=int, default=2)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = False
    if args.model == "GFPGANv1.4":
        success = restore_gfpgan(input_path, output_path, args.fidelity, args.upscale)
    elif args.model == "CodeFormer":
        success = restore_codeformer(input_path, output_path, args.fidelity, args.upscale)
    elif args.model == "Real-ESRGAN":
        success = restore_realesrgan(input_path, output_path, args.fidelity, args.upscale)
    else:
        logger.error(f"Unknown model: {args.model}")
        success = fallback_restore(input_path, output_path, args.upscale)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
'''
        script_path.write_text(script_content, encoding="utf-8")


class MockMLPipeline(MLPipeline):
    """Mock ML pipeline for testing without actual models."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Mock all models as available
        for model in RestorationModel:
            self._available_models[model] = True

    def restore_sync(self, task: RestorationTask) -> RestorationResult:
        """Override to process synchronously without worker thread."""
        return self._process_task_sync(task)

    def _process_task_sync(self, task: RestorationTask) -> RestorationResult:
        """Process task synchronously and return result directly."""
        import time
        from PIL import Image

        start_time = time.time()
        result = RestorationResult(task_id=task.task_id)

        try:
            # Simulate processing time
            time.sleep(0.1)  # Reduced for faster tests

            # Open and upscale image
            img = Image.open(task.input_path)
            w, h = img.size
            img = img.resize((w * task.upscale, h * task.upscale), Image.Resampling.LANCZOS)
            img.save(task.output_path)

            result.success = True
            result.output_path = task.output_path
        except Exception as e:
            result.success = False
            result.error = str(e)

        result.processing_time = time.time() - start_time
        return result

    def _process_task(self, task: RestorationTask) -> None:
        """Mock processing - just copy and upscale."""
        import time
        from PIL import Image

        start_time = time.time()
        result = RestorationResult(task_id=task.task_id)

        try:
            # Simulate processing time
            time.sleep(1.0)

            # Open and upscale image
            img = Image.open(task.input_path)
            w, h = img.size
            img = img.resize((w * task.upscale, h * task.upscale), Image.Resampling.LANCZOS)
            img.save(task.output_path)

            result.success = True
            result.output_path = task.output_path
        except Exception as e:
            result.success = False
            result.error = str(e)

        result.processing_time = time.time() - start_time

        if task.task_id in self._result_callbacks:
            try:
                self._result_callbacks[task.task_id](result)
            finally:
                del self._result_callbacks[task.task_id]