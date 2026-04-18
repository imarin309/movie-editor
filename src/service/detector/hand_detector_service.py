import os
from typing import Any, List, Optional

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from src.model import BoundingBox, Config, VideoMetaData
from src.service.bounding_box_service import BoundingBoxService
from src.service.detector.const import MAX_DETECT_WIDTH, MIN_CONFIDENCE
from src.service.detector.landmark_detector_service import LandmarkDetectorService

_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../models/hand_landmarker.task")
)


class HandDetectorService(LandmarkDetectorService):

    def __init__(self, config: Config, video_meta: VideoMetaData):
        super().__init__(config, video_meta)

    def _create_detector(self) -> Any:
        base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=MIN_CONFIDENCE,
            min_hand_presence_confidence=MIN_CONFIDENCE,
            min_tracking_confidence=MIN_CONFIDENCE,
        )
        return mp_vision.HandLandmarker.create_from_options(options)

    def _make_bounding_box(self, frame: Any) -> Optional[List[BoundingBox]]:
        h, w = frame.shape[:2]
        if w > MAX_DETECT_WIDTH:
            frame = cv2.resize(frame, (MAX_DETECT_WIDTH, int(h * MAX_DETECT_WIDTH / w)))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)

        if result.hand_landmarks:
            return [
                BoundingBoxService.calculate_from_landmarks(hand_landmarks)
                for hand_landmarks in result.hand_landmarks
            ]
        return None

    def _get_selection_key(self, bounding_box: BoundingBox) -> float:
        """右側の手を優先するため、x座標の中心を返す。"""
        return bounding_box.center_x
