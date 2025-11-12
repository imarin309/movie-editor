import math
from typing import Any, List, Optional, Tuple

import mediapipe as mp

from src.model import BoundingBox
from src.model.landmark_detector.base import LandmarkDetector


class HandDetector(LandmarkDetector):
    """手のランドマーク検出器。"""

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
                BoundingBox._calculate_bounding_box_in_target_randmark(hand_landmarks)
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
        return "Detecting hands"

    def _get_progress_desc_for_positions(self) -> str:
        return "Detecting hand positions"

# TODO: 移動させる
def calculate_auto_zoom_ratio(
    sizes: List[Optional[Tuple[float, float]]],
    target_hand_ratio: float = 0.25,
) -> float:
    """
    手のサイズから自動的にズーム率を計算する。
    手が画面の一定割合（デフォルト1/4）を占めるようにズーム率を決定する。

    Args:
        sizes: 各フレームの手のサイズ (width, height) のリスト
        target_hand_ratio: 手が占める目標の画面比率（0.25 = 1/4）

    Returns:
        計算されたズーム率（crop_zoom_ratio）
    """
    # 有効な手のサイズを抽出（面積で評価）
    valid_areas = []
    for size in sizes:
        if size is not None:
            width, height = size
            area = width * height
            valid_areas.append(area)

    if not valid_areas:
        # 手が検出されなかった場合はデフォルト値を返す
        return 0.5

    # 中央値を使用（外れ値の影響を軽減）
    valid_areas.sort()
    median_area = valid_areas[len(valid_areas) // 2]

    # 手が target_hand_ratio を占めるようにズーム率を計算
    # median_area / crop_ratio^2 = target_hand_ratio
    # crop_ratio^2 = median_area / target_hand_ratio
    # crop_ratio = sqrt(median_area / target_hand_ratio)
    crop_ratio = math.sqrt(median_area / target_hand_ratio)

    # ズーム率を合理的な範囲に制限（0.2 ~ 0.9）
    # 0.2 = 5倍ズーム、0.9 = 約1.1倍ズーム
    crop_ratio = max(0.2, min(0.9, crop_ratio))

    return crop_ratio
