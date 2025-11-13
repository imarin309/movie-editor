from dataclasses import dataclass


@dataclass
class BoundingBox:
    """バウンディングボックスを表すデータクラス"""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    width: float
    height: float
    area: float
    center_x: float
    center_y: float
