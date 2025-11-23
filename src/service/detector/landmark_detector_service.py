import abc
from typing import Any, List, Optional

from src.model import BoundingBox, Config, VideoMetaData
from src.model.service_abstract.landmark_detector_abstract import (
    LandmarkDetectorAbstract,
)
from src.service.bounding_boxes_service import BoundingBoxesService
from src.service.detector.const import MIN_TARGET_AREA


class LandmarkDetectorService(LandmarkDetectorAbstract):

    video_path: str
    fps_sample: int
    center_position_x: float
    center_detection_ratio: float

    def __init__(self, config: Config, video_meta: VideoMetaData) -> None:
        self.center_position_x = config.center_postion_x
        self.center_detection_ratio = config.center_detection_ratio
        self.video_meta = video_meta

    def _make_bounding_boxes(self) -> None:
        self.detector = self._create_detector()
        with self.detector:
            self.bounding_boxes = BoundingBoxesService.make_bounding_boxes(
                self.video_meta,
                self._make_bounding_box,
            )

    @abc.abstractmethod
    def _create_detector(self) -> Any:
        """対象物に依存するのでサブクラスで定義する"""
        pass

    @abc.abstractmethod
    def _make_bounding_box(self, frame: Any) -> Optional[List[BoundingBox]]:
        """対象物に依存するのでサブクラスで定義する"""
        pass

    def _is_valid_detection(self, bounding_box: BoundingBox) -> bool:
        """
        検出されたバウンディングボックスが有効かどうかを判定する

        面積チェックと中央位置チェックの両方を行う
        """
        # 面積チェック
        if bounding_box.area < MIN_TARGET_AREA:
            return False

        # X座標の中央位置チェック
        min_pos = self.center_position_x - self.center_detection_ratio
        max_pos = self.center_position_x + self.center_detection_ratio
        is_in_horizontal_range = min_pos <= bounding_box.center_x <= max_pos

        return is_in_horizontal_range

    def _select_best_detection(
        self, bounding_boxes: List[BoundingBox]
    ) -> Optional[BoundingBox]:
        """
        複数の検出から最適なものを選択する。

        有効な検出の中から、_get_selection_key()の値が最大のものを選択する。

        Args:
            bounding_boxes: 検出されたバウンディングボックスのリスト

        Returns:
            選択されたバウンディングボックス、または有効な検出がない場合はNone
        """
        best_detection = None
        best_key = -float("inf")

        for bounding_box in bounding_boxes:
            # 有効な検出のみを候補とする
            if self._is_valid_detection(bounding_box):
                key = self._get_selection_key(bounding_box)
                if key > best_key:
                    best_key = key
                    best_detection = bounding_box

        return best_detection

    def extract_mask(self) -> List[bool]:

        self._make_bounding_boxes()
        return [bool(bounding_box) for bounding_box in self.bounding_boxes]

    @abc.abstractmethod
    def _get_selection_key(self, bounding_box: BoundingBox) -> float:
        """対象物に依存するのでサブクラスで定義"""
        pass
