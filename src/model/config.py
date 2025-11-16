from dataclasses import dataclass


@dataclass
class Config:

    fps_sample: int
    center_postion_x: float
    center_detection_ratio: float
    movie_speed: int
