from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class LandmarkInfo:

    has_landmark_frame: List[bool]  # 対象が存在するフレーム
    landmark_size: List[Tuple[float, float] | None]
    landmark_position: List[Tuple[float, float] | None]
