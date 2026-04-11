"""Compact face recognition benchmark package."""

from .config import load_config
from .pipeline import RecognitionPipeline, build_pipeline

__all__ = ["RecognitionPipeline", "build_pipeline", "load_config"]
