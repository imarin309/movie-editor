import math
from typing import Any, List, Optional, Tuple

import cv2
from tqdm import tqdm

from src.model import BoundingBox, Config, LandmarkInfo
from src.model.service_abstract.landmark_detector_abstract import (
    LandmarkDetectorAbstract,
)
from src.service.video_service import VideoService


class LandmarkDetectorService(LandmarkDetectorAbstract):

    video_path: str
    fps_sample: int
    min_conf: float
    min_area_ratio: float

    def __init__(self, video_path: str, config: Config) -> None:
        self.video_path = video_path
        self.fps_sample = config.fps_sample
        self.min_conf = config.min_conf
        self.min_area_ratio = config.min_area_ratio

    def _make_video_editor(self):

        self.video_caputre = cv2.VideoCapture(self.video_path)
        if not self.video_caputre.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")
        self.video_meta = VideoService.get_video_meta(self.video_caputre)
        self.sampling_step, self.effective_fps = VideoService.get_effective_fps(
            self.video_meta.orig_fps, self.fps_sample
        )

        self.detector = self._create_detector(self.min_conf)
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
                total=math.ceil(self.video_meta.total_frames / self.sampling_step),
                desc="detecting target...",
                unit="f",
            )

            with self.detector:
                while True:
                    ret, frame = self.video_caputre.read()
                    if not ret:
                        break

                    if idx % self.sampling_step != 0:
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
            self.video_caputre.release()

        return bounding_boxes

    def _create_detector(self, min_conf: float) -> Any:
        """対象物に依存するのでサブクラスで定義する"""
        pass

    def _make_bounding_box(self, result: Any) -> Optional[List[BoundingBox]]:
        """対象物に依存するのでサブクラスで定義する"""
        pass

    def _is_valid_detection(
        self, bounding_box: BoundingBox, min_area_ratio: float
    ) -> bool:
        """検出されたバウンディングボックスが有効かどうかを判定する"""
        return bounding_box.area >= min_area_ratio

    def _select_best_detection(
        self, bounding_boxes: List[BoundingBox], min_area_ratio: float
    ) -> Optional[BoundingBox]:
        """
        複数の検出から最適なものを選択する。

        有効な検出の中から、_get_selection_key()の値が最大のものを選択する。

        Args:
            bounding_boxes: 検出されたバウンディングボックスのリスト
            min_area_ratio: 最小面積比

        Returns:
            選択されたバウンディングボックス、または有効な検出がない場合はNone
        """
        best_detection = None
        best_key = -float("inf")

        for bounding_box in bounding_boxes:
            # 有効な検出のみを候補とする
            if self._is_valid_detection(bounding_box, min_area_ratio):
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
                best_detection = self._select_best_detection(
                    bounding_box, self.min_area_ratio
                )

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

    def _get_selection_key(self, bounding_box: BoundingBox) -> float:
        """対象物に依存するのでサブクラスで定義"""
        pass
