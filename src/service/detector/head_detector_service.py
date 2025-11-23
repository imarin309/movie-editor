import logging
import math
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from src.model import BoundingBox, Config, LandmarkInfo, VideoMetaData
from src.service.detector.const import MIN_TARGET_AREA

logger = logging.getLogger(__name__)

# area
BOTTOM_REGION_RATIO = 0.1  # 下部領域の割合（画面下部から何%を対象とするか）

# color
DARK_THRESHOLD = 60  # 暗い色の閾値（HSVのV値） V < 50:暗い
DARK_RATIO_THRESHOLD = (
    0.4  # 暗いピクセルの割合閾値（この値以上なら頭が映っていると判定）
)

# shape
MIN_CIRCULARITY = 0.1  # 最小円形度
MAX_CIRCULARITY = 1.0  # 最大円形度
MIN_ASPECT_RATIO = 1.0  # 最小アスペクト比（幅/高さ）
MAX_ASPECT_RATIO = 15.0  # 最大アスペクト比（横長の形状も許容）


class HeadDetectorService:
    """
    TODO: 全体的にリファクタリング
    色+形状ベースの頭部検出

    画面下部の特定領域に暗い色が一定面積以上占めており、
    かつその輪郭が半円形状である場合、頭が映っていると判定する。
    """

    def __init__(self, config: Config, video_meta: VideoMetaData):
        self.center_position_x = config.center_postion_x
        self.center_detection_ratio = config.center_detection_ratio
        self.video_meta = video_meta
        self.frame_count = 0

    def _is_semicircle_shape(self, contour: np.ndarray) -> bool:
        """
        輪郭が「上下平ら、左右丸い」形状かどうかを判定する。

        頭部が画面下部に映る場合、上下は切れて平らになり、
        左右（耳のあたり）が丸く膨らんでいる形状を検出する。

        Args:
            contour: OpenCVの輪郭データ

        Returns:
            該当する形状ならTrue
        """
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return False

        # 円形度を計算: 4π × area / perimeter²
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # バウンディングボックスを取得してアスペクト比を計算
        x, _, w, h = cv2.boundingRect(contour)
        if h == 0:
            return False
        aspect_ratio = w / h

        # 輪郭の点を取得して形状を詳細に解析
        points = contour.reshape(-1, 2)

        # 上下の平坦性を確認
        # 輪郭の上端10%と下端10%のY座標のばらつきを確認
        y_coords = points[:, 1]
        y_min, y_max = y_coords.min(), y_coords.max()
        y_range = y_max - y_min

        if y_range == 0:
            return False

        # 上端10%の領域
        top_threshold = y_min + y_range * 0.1
        top_points = points[y_coords <= top_threshold]
        top_flatness = 0.0
        if len(top_points) > 1:
            top_y_std = np.std(top_points[:, 1])
            top_flatness = top_y_std / y_range  # 正規化

        # 下端10%の領域
        bottom_threshold = y_max - y_range * 0.1
        bottom_points = points[y_coords >= bottom_threshold]
        bottom_flatness = 0.0
        if len(bottom_points) > 1:
            bottom_y_std = np.std(bottom_points[:, 1])
            bottom_flatness = bottom_y_std / y_range  # 正規化

        # 左右の丸みを確認（片方だけでもOK）
        # 中央部（上下40%-60%）での左または右への膨らみを確認
        mid_y_min = y_min + y_range * 0.4
        mid_y_max = y_min + y_range * 0.6
        mid_points = points[(y_coords >= mid_y_min) & (y_coords <= mid_y_max)]

        has_side_bulge = False
        if len(mid_points) > 0:
            x_coords_mid = mid_points[:, 0]
            x_min_mid, x_max_mid = x_coords_mid.min(), x_coords_mid.max()

            # 左側の膨らみ：左端から中央部までの広がり
            left_bulge = x_min_mid - x
            # 右側の膨らみ：中央部から右端までの広がり
            right_bulge = (x + w) - x_max_mid

            # バウンディングボックスの幅に対する割合
            left_bulge_ratio = left_bulge / w if w > 0 else 0
            right_bulge_ratio = right_bulge / w if w > 0 else 0

            # 片方でも膨らみがあればOK（幅の15%以内であれば丸みがあると判定）
            if left_bulge_ratio < 0.15 or right_bulge_ratio < 0.15:
                has_side_bulge = True

        # 判定基準
        is_horizontal = aspect_ratio >= MIN_ASPECT_RATIO  # 横長
        is_top_flat = top_flatness < 0.15  # 上部が平ら（ばらつきが小さい）
        is_bottom_flat = bottom_flatness < 0.15  # 下部が平ら
        is_valid_circularity = MIN_CIRCULARITY <= circularity <= MAX_CIRCULARITY
        is_valid_aspect = aspect_ratio <= MAX_ASPECT_RATIO

        return (
            is_horizontal
            and is_top_flat
            and is_bottom_flat
            and has_side_bulge
            and is_valid_circularity
            and is_valid_aspect
        )

    def _make_bouding_boxes(self) -> List[Optional[BoundingBox]]:
        bounding_boxes = []

        try:
            idx = 0
            pbar = tqdm(
                total=math.ceil(
                    self.video_meta.total_frames / self.video_meta.sampling_step
                ),
                desc="detecting head...",
                unit="f",
            )

            while True:
                ret, frame = self.video_meta.video_capture.read()
                if not ret:
                    break

                if idx % self.video_meta.sampling_step != 0:
                    idx += 1
                    continue

                bounding_box = self._make_bounding_box(frame)
                bounding_boxes.append(bounding_box)

                pbar.update(1)
                idx += 1

            pbar.close()
        finally:
            self.video_meta.video_capture.release()

        return bounding_boxes

    def _make_bounding_box(self, result: Any) -> Optional[BoundingBox]:
        """
        画面下部の暗い領域から頭部のバウンディングボックスを生成する。

        画面下部の指定領域で暗い色（黒）のピクセル割合を計算し、
        閾値以上なら頭が映っていると判定してバウンディングボックスを返す。

        Args:
            result: フレーム（BGR画像）

        Returns:
            頭部のバウンディングボックスのリスト、または検出されなかった場合はNone
        """
        frame = result
        height, width = frame.shape[:2]

        # 画面下部の領域を取得
        bottom_region_start = int(height * (1.0 - BOTTOM_REGION_RATIO))
        bottom_region = frame[bottom_region_start:, :]

        # HSV色空間に変換
        hsv = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2HSV)

        # 暗い色（黒）を検出
        # H: 0-180（色相）, S: 0-255（彩度）, V: 0-255（明度）
        # 暗い色の定義: V（明度）が低い
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, DARK_THRESHOLD])

        mask = cv2.inRange(hsv, lower_dark, upper_dark)

        # 暗いピクセルの割合を計算
        total_pixels = bottom_region.shape[0] * bottom_region.shape[1]
        dark_pixels = np.sum(mask > 0)
        dark_ratio = dark_pixels / total_pixels

        # デバッグログ（最初の10フレームのみ）
        if self.frame_count < 10:
            logger.info(
                f"Frame {self.frame_count}: dark_ratio={dark_ratio:.3f}, "
                f"dark_pixels={dark_pixels}, total_pixels={total_pixels}"
            )

        # 閾値以上なら形状チェックを実行
        if dark_ratio >= DARK_RATIO_THRESHOLD:
            # 輪郭を検出
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if self.frame_count < 10:
                logger.info(f"  Found {len(contours)} contours")

            # 各輪郭の形状をチェック
            valid_contours = []
            for contour in contours:
                # 小さすぎる輪郭は除外
                area = cv2.contourArea(contour)
                if area < total_pixels * 0.05:  # 領域の5%未満は除外
                    continue

                # 半円形状かチェック
                if self._is_semicircle_shape(contour):
                    valid_contours.append(contour)

            if self.frame_count < 10:
                logger.info(f"  Valid semicircle contours: {len(valid_contours)}")

            self.frame_count += 1

            # 半円形状の輪郭が見つかった場合のみ頭と判定
            if valid_contours:
                # 頭が映っている場合、下部領域全体をバウンディングボックスとして返す
                x_min = 0.0
                y_min = 1.0 - BOTTOM_REGION_RATIO
                x_max = 1.0
                y_max = 1.0
                width = 1.0
                height = BOTTOM_REGION_RATIO
                area = BOTTOM_REGION_RATIO

                bounding_box = BoundingBox(
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                    width=width,
                    height=height,
                    area=area,
                    center_x=0.5,
                    center_y=1.0 - BOTTOM_REGION_RATIO / 2,
                )

                if self._is_valid_detection(bounding_box):
                    return bounding_box
        else:
            self.frame_count += 1

        return None

    def _is_valid_detection(self, bounding_box: BoundingBox) -> bool:
        if bounding_box.area < MIN_TARGET_AREA:
            return False

        min_pos = self.center_position_x - self.center_detection_ratio
        max_pos = self.center_position_x + self.center_detection_ratio
        is_in_horizontal_range = min_pos <= bounding_box.center_x <= max_pos

        return is_in_horizontal_range

    def extract_landmark_info(self) -> None:
        self.bounding_boxes = self._make_bouding_boxes()

        positions: List[Optional[Tuple[float, float]]] = []
        sizes: List[Optional[Tuple[float, float]]] = []
        has_detections: List[bool] = []

        for bounding_box in self.bounding_boxes:
            position = None
            size = None
            has_detection = False

            if bounding_box is not None:
                position = (bounding_box.center_x, bounding_box.center_y)
                size = (bounding_box.width, bounding_box.height)
                has_detection = True

            positions.append(position)
            sizes.append(size)
            has_detections.append(has_detection)

        self.landmark_info = LandmarkInfo(
            has_landmark_frame=has_detections,
            landmark_size=sizes,
            landmark_position=positions,
        )
