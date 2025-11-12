from dataclasses import dataclass
from typing import Any

@dataclass
class BoundingBox:
    
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    width: float
    height: float
    area: float
    center_x: float
    center_y: float

    @classmethod
    def _calculate_bounding_box_in_target_randmark(cls, landmarks: Any) -> "BoundingBox":
        """
        対象のランドマークからバウンディングボックスを計算する。

        Args:
            landmarks: MediaPipeのランドマーク

        Returns:
            対象のバウンディングボックス情報
        """
        x_coords = [lm.x for lm in landmarks.landmark]
        y_coords = [lm.y for lm in landmarks.landmark]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        width = x_max - x_min
        height = y_max - y_min
        area = width * height

        return cls(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
            width=width,
            height=height,
            area=area,
            center_x=(x_min + x_max) / 2,
            center_y=(y_min + y_max) / 2,
        )