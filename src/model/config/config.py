from dataclasses import dataclass


@dataclass
class Config:

    fps_sample :int
    min_conf :float
    min_area_ratio :float
    min_keep_sec :float
    pad_sec :float
    merge_gap_sec :float
    landmark_horizontal_ratio :float
    landmark_vertical_ratio :float
    smooth_window_size :int
    auto_zoom :float
    target_landmark_ratio :float
    crop_zoom_ratio :float