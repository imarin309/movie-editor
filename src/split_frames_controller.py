# -*- coding: utf-8 -*-
import logging
from pathlib import Path

from src.service import FrameExtractService

logger = logging.getLogger(__name__)


def split_frames_controller(
    input_path: str,
    n_frames: int,
    center_crop_ratio: float = 1.0,
) -> None:
    video_file = Path(input_path)
    if not video_file.exists() or not video_file.is_file():
        raise FileNotFoundError(f"動画ファイルが見つかりません: {input_path}")

    output_dir = video_file.parent / "output" / "image" / video_file.stem
    logger.info(f"{video_file.name} を {n_frames} 枚に分割 -> {output_dir}")

    saved = FrameExtractService.extract_n_frames(
        input_movie_path=str(video_file),
        output_dir=output_dir,
        n_frames=n_frames,
        center_crop_ratio=center_crop_ratio,
    )
    logger.info(f"{len(saved)} フレーム保存完了")
