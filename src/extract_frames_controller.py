# -*- coding: utf-8 -*-
import logging
from pathlib import Path

from src.service import FrameExtractService

logger = logging.getLogger(__name__)

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
_OUTPUT_DIR = Path(__file__).parents[1] / "output" / "image"


def extract_frames_controller(
    input_dir: str,
    interval_sec: int,
    search_window_sec: float,
) -> None:
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        raise NotADirectoryError(f"Directory not found: {input_dir}")

    video_files = [
        f
        for f in input_path.rglob("*")
        if f.is_file() and f.suffix.lower() in _VIDEO_EXTENSIONS
    ]

    if not video_files:
        logger.warning(f"動画ファイルが見つかりませんでした: {input_dir}")
        return

    for idx, video_file in enumerate(video_files, start=1):
        output_dir = _OUTPUT_DIR
        logger.info(f"[{idx}/{len(video_files)}] {video_file.name} -> {output_dir}")
        saved = FrameExtractService.extract_frames(
            input_movie_path=str(video_file),
            output_dir=output_dir,
            interval_sec=interval_sec,
            search_window_sec=search_window_sec,
        )
        logger.info(f"  {len(saved)} フレーム保存完了")
