from dataclasses import dataclass

@dataclass
class VideoMetaData:
    """動画のメタデータを保持するデータクラス"""

    width: int
    height: int
    orig_fps: float
    total_frames: int