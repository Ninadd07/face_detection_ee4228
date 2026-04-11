from __future__ import annotations

import time
from pathlib import Path

import psutil
import torch

from .evaluation import load_image
from .pipeline import RecognitionPipeline


def benchmark_pipeline(
    pipeline: RecognitionPipeline,
    image_paths: list[Path],
) -> dict:
    process = psutil.Process()
    latencies: list[float] = []
    face_counts: list[int] = []
    for path in image_paths:
        frame = load_image(path)
        start = time.perf_counter()
        result = pipeline.run(frame)
        latencies.append((time.perf_counter() - start) * 1000.0)
        face_counts.append(len(result.predictions))
    avg_latency = sum(latencies) / max(1, len(latencies))
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    return {
        "samples": len(image_paths),
        "average_frame_latency_ms": avg_latency,
        "fps": fps,
        "average_faces_per_frame": sum(face_counts) / max(1, len(face_counts)),
        "rss_bytes": process.memory_info().rss,
        "cuda_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "latencies_ms": latencies,
    }
