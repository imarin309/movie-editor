from dataclasses import dataclass


@dataclass
class VideoMetaData:
    
    width: int
    height: int
    orig_fps: float
    total_frames: int
