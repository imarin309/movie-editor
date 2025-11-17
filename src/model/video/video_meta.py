from dataclasses import dataclass

import cv2


@dataclass
class VideoMetaData:

    video_capture: cv2.VideoCapture
    width: int
    height: int
    orig_fps: float
    total_frames: int
    sampling_step: int
    effective_fps: float
