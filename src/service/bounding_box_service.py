from typing import Any

from src.model import BoundingBox


class BoundingBoxService:

    @staticmethod
    def calculate_from_landmarks(landmarks: Any) -> BoundingBox:
        """
        対象のランドマークからバウンディングボックスを計算する。

        Args:
            landmarks: MediaPipeのランドマーク

        Returns:
            計算されたバウンディングボックス
        """
        x_coords = [lm.x for lm in landmarks.landmark]
        y_coords = [lm.y for lm in landmarks.landmark]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        width = x_max - x_min
        height = y_max - y_min
        area = width * height

        return BoundingBox(
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
