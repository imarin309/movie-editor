# -*- coding: utf-8 -*-
import logging
from pathlib import Path

from src.service import FrameExtractService

logger = logging.getLogger(__name__)

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}


def extract_frames_controller(
    input_dir: str,
    interval_sec: int,
    search_window_sec: float,
) -> None:
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        raise NotADirectoryError(f"Directory not found: {input_dir}")

    output_root = input_path / "output"
    output_dir = output_root / "image"
    video_files = [
        f
        for f in input_path.rglob("*")
        if f.is_file()
        and f.suffix.lower() in _VIDEO_EXTENSIONS
        # output/ 配下の生成物を除外
        and not f.is_relative_to(output_root)
    ]

    if not video_files:
        logger.warning(f"動画ファイルが見つかりませんでした: {input_dir}")
        return

    for idx, video_file in enumerate(video_files, start=1):
        logger.info(f"[{idx}/{len(video_files)}] {video_file.name} -> {output_dir}")
        saved = FrameExtractService.extract_frames(
            input_movie_path=str(video_file),
            output_dir=output_dir,
            interval_sec=interval_sec,
            search_window_sec=search_window_sec,
        )
        logger.info(f"  {len(saved)} フレーム保存完了")
