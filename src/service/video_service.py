from typing import Tuple

import cv2

from src.model import VideoMetaData


class VideoService:
    """動画の初期化とメタデータ取得を行うサービスクラス"""

    @classmethod
    def get_video_meta(cls, input_movie_path: str, sampling_fps: int) -> VideoMetaData:

        video_capture = cv2.VideoCapture(input_movie_path)
        if not video_capture.isOpened():
            raise RuntimeError(f"Could not open video: {input_movie_path}")
        original_fps = video_capture.get(cv2.CAP_PROP_FPS)
        sampling_step, effective_fps = cls._get_effective_fps(
            original_fps, sampling_fps
        )

        metadata = VideoMetaData(
            video_capture=video_capture,
            width=int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            orig_fps=original_fps,
            total_frames=int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            sampling_step=sampling_step,
            effective_fps=effective_fps,
        )

        return metadata

    @staticmethod
    def _get_effective_fps(original_fps: float, sampling_fps: int) -> Tuple[int, float]:
        """
        元動画のFPSとサンプリングFPSから、実効FPSとフレームステップを計算する。

        Args:
            original_fps: 元動画のFPS
            sampling_fps: サンプリングFPS

        Returns:
            (sampling_step, eff_fps) のタプル
            - sampling_step: 何フレームごとに処理するか
            - eff_fps: 実効FPS
        """
        if original_fps > 0:
            sampling_step = max(1, int(round(original_fps / sampling_fps)))
            effective_fps = original_fps / sampling_step
        else:
            sampling_step = 1
            effective_fps = sampling_fps
        return sampling_step, effective_fps
