from dataclasses import dataclass
import cv2
from typing import Tuple

@dataclass
class VideoMetaData:
    """動画のメタデータを保持するデータクラス"""

    width: int
    height: int
    orig_fps: float
    total_frames: int

    @classmethod
    def setup_video_capture(cls, video_path: str) -> Tuple[cv2.VideoCapture, "VideoMetaData"]:
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

