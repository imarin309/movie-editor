from typing import Any, List, Optional

import mediapipe as mp

from src.model import BoundingBox
from src.service.bounding_box_service import BoundingBoxService
from src.service.detector.landmark_detector_service import LandmarkDetectorService


class HandDetectorService(LandmarkDetectorService):
    """
    手の検出に特化したサービスクラス。

    MediaPipe Handsを使用して手を検出し、
    右側の手を優先的に選択する。
    """

    def _create_detector(self, min_conf: float) -> Any:
        """
        MediaPipe Hands検出器を作成する。

        Args:
            min_conf: 最小信頼度

        Returns:
            MediaPipe Hands検出器インスタンス
        """
        mp_hands = mp.solutions.hands
        return mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )

    def _make_bounding_box(self, result: Any) -> Optional[List[BoundingBox]]:
        """
        MediaPipe Handsの検出結果からバウンディングボックスのリストを生成する。

        Args:
            result: MediaPipe Handsの検出結果

        Returns:
            手のバウンディングボックスのリスト、または検出されなかった場合はNone
        """
        if result.multi_hand_landmarks and result.multi_handedness:
            return [
                BoundingBoxService.calculate_from_landmarks(hand_landmarks)
                for hand_landmarks in result.multi_hand_landmarks
            ]
        return None

    def _get_selection_key(self, bounding_box: BoundingBox) -> float:
        """
        手の選択基準値を返す。

        右側の手を優先するため、x座標の中心を返す。

        Args:
            bounding_box: 検出された手のバウンディングボックス
        Returns:
            選択基準値（x座標の中心、大きいほど右側）
        """
        return bounding_box.center_x

    def _get_progress_desc_for_mask(self) -> str:
        """
        マスク検出時のプログレスバー説明文を返す。

        Returns:
            "Detecting hands"
        """
        return "Detecting hands"

    def _get_progress_desc_for_positions(self) -> str:
        """
        位置検出時のプログレスバー説明文を返す。

        Returns:
            "Detecting hand positions"
        """
        return "Detecting hand positions"
