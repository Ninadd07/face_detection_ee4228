from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import psutil
import torch

from ..config import load_config
from ..pipeline import build_pipeline
from .enroll_gallery import build_gallery_from_manifest


def _opencv_gui_backend() -> str:
    for line in cv2.getBuildInformation().splitlines():
        stripped = line.strip()
        if stripped.startswith("GUI:"):
            return stripped.split(":", 1)[1].strip()
    return "unknown"


def _ensure_gui_support() -> None:
    backend = _opencv_gui_backend()
    if backend.upper() != "NONE":
        return
    project_python = Path(__file__).resolve().parents[3] / "venv" / "Scripts" / "python.exe"
    raise RuntimeError(
        "OpenCV in the current Python environment was built without window support, "
        f"so live preview cannot open. Interpreter: {sys.executable}. GUI backend: {backend}. "
        f"Run the demo with the project-local interpreter instead: {project_python} -m mobilefacenet_face.cli.live_demo. "
        "If you intended to use this environment, remove opencv-python-headless and reinstall opencv-python."
    )


def _bytes_to_mb(value: int) -> float:
    return value / (1024.0 * 1024.0)


class ResourceMonitor:
    def __init__(self, device: torch.device, sample_interval_s: float = 0.5) -> None:
        self.device = device
        self.sample_interval_s = sample_interval_s
        self.process = psutil.Process()
        self.nvidia_smi = shutil.which("nvidia-smi")
        self.last_sample_at = 0.0
        self.last_snapshot: dict[str, object] | None = None
        psutil.cpu_percent(interval=None)
        self.process.cpu_percent(interval=None)
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except RuntimeError:
                pass

    def snapshot(self) -> dict[str, object]:
        now = time.perf_counter()
        if self.last_snapshot is not None and now - self.last_sample_at < self.sample_interval_s:
            return self.last_snapshot
        snapshot: dict[str, object] = {
            "device": str(self.device),
            "process_cpu_percent": self.process.cpu_percent(interval=None),
            "system_cpu_percent": psutil.cpu_percent(interval=None),
            "process_rss_bytes": self.process.memory_info().rss,
        }
        snapshot.update(self._gpu_snapshot())
        self.last_sample_at = now
        self.last_snapshot = snapshot
        return snapshot

    def _gpu_snapshot(self) -> dict[str, object]:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return {
                "gpu_available": False,
                "gpu_name": None,
                "gpu_utilization_percent": None,
                "gpu_memory_used_mb": None,
                "gpu_memory_total_mb": None,
                "cuda_memory_allocated_bytes": 0,
                "cuda_memory_reserved_bytes": 0,
                "cuda_max_memory_allocated_bytes": 0,
            }
        device_index = self._cuda_device_index()
        snapshot: dict[str, object] = {
            "gpu_available": True,
            "gpu_name": torch.cuda.get_device_name(device_index),
            "gpu_utilization_percent": None,
            "gpu_memory_used_mb": None,
            "gpu_memory_total_mb": None,
            "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated(device_index)),
            "cuda_memory_reserved_bytes": int(torch.cuda.memory_reserved(device_index)),
            "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device_index)),
        }
        if not self.nvidia_smi:
            return snapshot
        try:
            result = subprocess.run(
                [
                    self.nvidia_smi,
                    f"--id={device_index}",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=2,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return snapshot
        line = result.stdout.strip().splitlines()
        if not line:
            return snapshot
        try:
            utilization, used_mb, total_mb = [item.strip() for item in line[0].split(",", maxsplit=2)]
            snapshot["gpu_utilization_percent"] = float(utilization)
            snapshot["gpu_memory_used_mb"] = float(used_mb)
            snapshot["gpu_memory_total_mb"] = float(total_mb)
        except ValueError:
            return snapshot
        return snapshot

    def _cuda_device_index(self) -> int:
        return self.device.index if self.device.index is not None else torch.cuda.current_device()


def _resource_overlay_lines(resources: dict[str, object], timings_ms: dict[str, float]) -> list[str]:
    lines = [
        (
            f"frame {timings_ms['total_ms']:.1f} ms | "
            f"cpu proc {float(resources['process_cpu_percent']):.1f}% | "
            f"cpu sys {float(resources['system_cpu_percent']):.1f}% | "
            f"ram {_bytes_to_mb(int(resources['process_rss_bytes'])):.0f} MB"
        )
    ]
    if not resources.get("gpu_available"):
        return lines
    if resources.get("gpu_utilization_percent") is not None and resources.get("gpu_memory_total_mb") is not None:
        lines.append(
            (
                f"gpu {resources['device']} {float(resources['gpu_utilization_percent']):.0f}% | "
                f"vram {float(resources['gpu_memory_used_mb']):.0f}/{float(resources['gpu_memory_total_mb']):.0f} MB | "
                f"torch {_bytes_to_mb(int(resources['cuda_memory_allocated_bytes'])):.0f} MB"
            )
        )
        return lines
    lines.append(
        (
            f"gpu {resources['device']} | "
            f"torch alloc {_bytes_to_mb(int(resources['cuda_memory_allocated_bytes'])):.0f} MB | "
            f"reserved {_bytes_to_mb(int(resources['cuda_memory_reserved_bytes'])):.0f} MB"
        )
    )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--checkpoint-path", default=None)
    args = parser.parse_args()
    _ensure_gui_support()
    config = load_config(args.config)
    pipeline = build_pipeline(args.config, gallery_path=config["outputs"]["gallery_path"], checkpoint_path=args.checkpoint_path)
    if args.checkpoint_path:
        _, gallery = build_gallery_from_manifest(args.config, checkpoint_path=args.checkpoint_path)
        pipeline.gallery = gallery
    resource_monitor = ResourceMonitor(getattr(pipeline.embedder, "device", torch.device("cpu")))
    log_path = Path(config["outputs"]["prediction_log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam.")
    frame_index = 0
    with log_path.open("a", encoding="utf-8") as log_handle:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = pipeline.run(frame)
            resources = resource_monitor.snapshot()
            frame_predictions: list[dict[str, object]] = []
            for prediction in result.predictions:
                x1, y1, x2, y2 = prediction.detection.box.astype(int)
                label = prediction.match.identity or "unknown"
                text = f"{label} {prediction.match.score:.3f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, text, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                frame_predictions.append(
                    {
                        "identity": label,
                        "score": prediction.match.score,
                        "accepted": prediction.match.accepted,
                        "confidence": prediction.detection.confidence,
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                    }
                )
            log_handle.write(
                json.dumps(
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "frame_index": frame_index,
                        "predictions": frame_predictions,
                        "timings_ms": result.timings_ms,
                        "resources": resources,
                    }
                )
                + "\n"
            )
            log_handle.flush()
            for line_index, text in enumerate(_resource_overlay_lines(resources, result.timings_ms)):
                cv2.putText(
                    frame,
                    text,
                    (10, 25 + (line_index * 24)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
            try:
                cv2.imshow("Compact Face Recognition", frame)
            except cv2.error as exc:
                raise RuntimeError(
                    "OpenCV could not create a preview window for mobilefacenet-live. "
                    f"Interpreter: {sys.executable}. GUI backend: {_opencv_gui_backend()}."
                ) from exc
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_index += 1
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
