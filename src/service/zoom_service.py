import logging
import math
from typing import List, Optional, Tuple

from src.model import CroppingConfig

logger = logging.getLogger(__name__)


def _calculate_auto_zoom_ratio(
    sizes: List[Optional[Tuple[float, float]]],
    target_landmark_ratio: float = 0.25,
) -> float:
    """
    ランドマークのサイズから自動的にズーム率を計算する。
    ランドマークが画面の一定割合（デフォルト1/4）を占めるようにズーム率を決定する。

    Args:
        sizes: 各フレームのランドマークサイズ (width, height) のリスト
        target_landmark_ratio: ランドマークが占める目標の画面比率（0.25 = 1/4）

    Returns:
        計算されたズーム率（crop_zoom_ratio）
    """
    # 有効なランドマークのサイズを抽出（面積で評価）
    valid_areas = []
    for size in sizes:
        if size is not None:
            width, height = size
            area = width * height
            valid_areas.append(area)

    if not valid_areas:
        # ランドマークが検出されなかった場合はデフォルト値を返す
        logger.warning("No valid landmark sizes found, using default zoom ratio")
        return 0.5

    # 中央値を使用（外れ値の影響を軽減）
    valid_areas.sort()
    median_area = valid_areas[len(valid_areas) // 2]

    # ランドマークが target_landmark_ratio を占めるようにズーム率を計算
    # median_area / crop_ratio^2 = target_landmark_ratio
    # crop_ratio^2 = median_area / target_landmark_ratio
    # crop_ratio = sqrt(median_area / target_landmark_ratio)
    crop_ratio = math.sqrt(median_area / target_landmark_ratio)

    # ズーム率を合理的な範囲に制限（0.2 ~ 0.9）
    # 0.2 = 5倍ズーム、0.9 = 約1.1倍ズーム
    crop_ratio = max(0.2, min(0.9, crop_ratio))

    logger.info(
        f"Calculated zoom ratio: {crop_ratio:.3f} from {len(valid_areas)} frames (median area: {median_area:.4f})"
    )

    return crop_ratio


def calculate_zoom_ratio(
    config: CroppingConfig,
    sizes: List[Optional[Tuple[float, float]]],
) -> float:
    """
    ズーム率を計算する。

    Args:
        config: クロップ設定
        sizes: ランドマークのサイズリスト
        calculate_auto_zoom_fn: 自動ズーム率計算関数

    Returns:
        計算されたズーム率

    Raises:
        ValueError: auto_zoomがTrueでcalculate_auto_zoom_fnがNoneの場合
    """
    if config.auto_zoom:
        logger.info("Calculating auto zoom ratio based on landmark size")
        zoom_ratio = _calculate_auto_zoom_ratio(sizes, config.target_landmark_ratio)
        logger.info(f"Auto zoom ratio: {zoom_ratio:.3f} (zoom: {1 / zoom_ratio:.1f}x)")
        return zoom_ratio
    else:
        logger.info(
            f"Using manual zoom ratio: {config.crop_zoom_ratio:.3f} (zoom: {1 / config.crop_zoom_ratio:.1f}x)"
        )
        return config.crop_zoom_ratio
