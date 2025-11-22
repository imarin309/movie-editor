import abc
import math
from typing import Any, List, Optional, Tuple

import cv2
from tqdm import tqdm

from src.model import BoundingBox, Config, LandmarkInfo, VideoMetaData
from src.model.service_abstract.landmark_detector_abstract import (
    LandmarkDetectorAbstract,
)
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

    def _make_video_editor(self) -> None:

        self.detector = self._create_detector()
        self.bounding_boxes = self._calculate_process_frame()

    def _calculate_process_frame(self) -> List[Optional[List[BoundingBox]]]:
        """
        動画フレームをサンプリングして処理し、各フレームの検出結果をリストで返す。

        元動画の全フレームではなく、fps_sampleに基づいて計算されたステップ間隔で
        フレームをサンプリングして処理する。
        検出対象は _create_detector() を実装するサブクラスによって決定される。

        Returns:
            各フレームのバウンディングボックスのリスト、または検出されなかった場合はNone
        """
        bounding_boxes = []

        try:
            idx = 0
            pbar = tqdm(
                total=math.ceil(
                    self.video_meta.total_frames / self.video_meta.sampling_step
                ),
                desc="detecting target...",
                unit="f",
            )

            with self.detector:
                while True:
                    ret, frame = self.video_meta.video_capture.read()
                    if not ret:
                        break

                    if idx % self.video_meta.sampling_step != 0:
                        idx += 1
                        continue

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = self.detector.process(rgb)

                    bounding_box = self._make_bounding_box(result)

                    bounding_boxes.append(bounding_box)

                    pbar.update(1)
                    idx += 1

            pbar.close()
        finally:
            self.video_meta.video_capture.release()

        return bounding_boxes

    @abc.abstractmethod
    def _create_detector(self) -> Any:
        """対象物に依存するのでサブクラスで定義する"""
        pass

    @abc.abstractmethod
    def _make_bounding_box(self, result: Any) -> Optional[List[BoundingBox]]:
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

    def extract_landmark_info(self) -> None:
        """
        動画を読み、フレームごとにランドマークの中心座標とサイズを返す。

        """
        self._make_video_editor()

        positions: List[Optional[Tuple[float, float]]] = []
        sizes: List[Optional[Tuple[float, float]]] = []
        has_detections: List[bool] = []

        for bounding_box in self.bounding_boxes:
            position = None
            size = None
            has_detection = False

            if bounding_box is not None:
                best_detection = self._select_best_detection(bounding_box)

                if best_detection is not None:
                    position = (best_detection.center_x, best_detection.center_y)
                    size = (best_detection.width, best_detection.height)
                    has_detection = True

            positions.append(position)
            sizes.append(size)
            has_detections.append(has_detection)

        self.landmark_info = LandmarkInfo(
            has_landmark_frame=has_detections,
            landmark_size=sizes,
            landmark_position=positions,
        )

    @abc.abstractmethod
    def _get_selection_key(self, bounding_box: BoundingBox) -> float:
        """対象物に依存するのでサブクラスで定義"""
        pass
