from dataclasses import dataclass

@dataclass
class CroppingConfig:
    
    fps_sample: int
    min_conf: float
    min_area_ratio: float
    landmark_horizontal_ratio: float
    landmark_vertical_ratio: float
    smooth_window_size: int
    crop_zoom_ratio: float
    auto_zoom: bool
    target_landmark_ratio: float