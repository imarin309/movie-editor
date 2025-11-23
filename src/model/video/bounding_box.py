from dataclasses import dataclass
from typing import List, Optional, Union


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


BoundingBoxes = List[Optional[Union[BoundingBox, List[BoundingBox]]]]
