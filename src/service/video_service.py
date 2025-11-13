from typing import Tuple

import cv2

from src.model import VideoMetaData


class VideoService:
    """動画の初期化とメタデータ取得を行うサービスクラス"""

    @classmethod
    def setup_video_capture(
        cls, video_path: str
    ) -> Tuple[cv2.VideoCapture, VideoMetaData]:
        """
        VideoCaptureを初期化し、動画のメタデータを取得する。

        Args:
            video_path: 動画ファイルのパス

        Returns:
            初期化されたVideoCaptureオブジェクトとメタデータ

        Raises:
            RuntimeError: 動画を開けない場合
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        metadata = VideoMetaData(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            orig_fps=cap.get(cv2.CAP_PROP_FPS),
            total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

        return cap, metadata

    @staticmethod
    def get_effective_fps(original_fps: float, sampling_fps: int) -> Tuple[int, float]:
        """
        元動画のFPSとサンプリングFPSから、実効FPSとフレームステップを計算する。

        Args:
            original_fps: 元動画のFPS
            sampling_fps: サンプリングFPS

        Returns:
            (step, eff_fps) のタプル
            - step: 何フレームごとに処理するか
            - eff_fps: 実効FPS
        """
        if original_fps > 0:
            step = max(1, int(round(original_fps / sampling_fps)))
            effective_fps = original_fps / step
        else:
            step = 1
            effective_fps = sampling_fps
        return step, effective_fps
