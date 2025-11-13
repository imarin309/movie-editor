import logging
from typing import Callable, List, Optional, Tuple, Union

from moviepy import VideoFileClip, concatenate_videoclips
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.VideoClip import VideoClip
from tqdm import tqdm

from src.model import CroppingConfig
from src.service.detector.landmark_detector_service import LandmarkDetectorService
from src.service.zoom_service import calculate_zoom_ratio

logger = logging.getLogger(__name__)


def _prepare_positions(
    positions: List[Optional[Tuple[float, float]]],
    config: CroppingConfig,
    smooth_positions_fn: Optional[
        Callable[
            [List[Optional[Tuple[float, float]]], int],
            List[Optional[Tuple[float, float]]],
        ]
    ],
) -> Tuple[List[Optional[Tuple[float, float]]], List[int]]:
    """
    位置データをスムージングし、有効フレームを抽出する。

    Args:
        positions: ランドマーク位置のリスト
        config: クロップ設定
        smooth_positions_fn: 位置スムージング関数

    Returns:
        (スムージング済み位置リスト, 有効フレームインデックスリスト)
    """
    # スムージング
    if smooth_positions_fn is not None:
        logger.info("Smoothing landmark positions")
        positions = smooth_positions_fn(positions, config.smooth_window_size)

    # 有効フレームの抽出
    valid_frames = [i for i, pos in enumerate(positions) if pos is not None]

    if not valid_frames:
        logger.warning("No landmarks detected for cropping")

    return positions, valid_frames


def _calculate_crop_position(
    cx: float,
    cy: float,
    crop_w: int,
    crop_h: int,
    orig_w: int,
    orig_h: int,
    config: CroppingConfig,
) -> Tuple[int, int, bool]:
    """
    クロップ位置を計算し、境界チェックを行う。

    Args:
        cx: ランドマークのx座標（ピクセル）
        cy: ランドマークのy座標（ピクセル）
        crop_w: クロップ幅
        crop_h: クロップ高さ
        orig_w: 元動画の幅
        orig_h: 元動画の高さ
        config: クロップ設定

    Returns:
        (調整後x1, 調整後y1, 調整されたか)
    """
    # クロップ範囲の左上座標を計算
    x1 = cx - crop_w * config.landmark_horizontal_ratio
    y1 = cy - crop_h * config.landmark_vertical_ratio

    adjusted = False

    # 水平方向の境界チェック
    if x1 < 0:
        x1 = 0
        adjusted = True
    elif x1 + crop_w > orig_w:
        x1 = orig_w - crop_w
        adjusted = True

    # 垂直方向の境界チェック
    if y1 < 0:
        y1 = 0
        adjusted = True
    elif y1 + crop_h > orig_h:
        y1 = orig_h - crop_h
        adjusted = True

    return int(x1), int(y1), adjusted


def _log_first_frame_info(
    cx: float,
    cy: float,
    x1: int,
    y1: int,
    x1_original: float,
    crop_w: int,
    crop_h: int,
    orig_w: int,
    orig_h: int,
    config: CroppingConfig,
    adjusted: bool,
) -> None:
    """
    最初のフレームのクロップ情報をログに出力する。

    Args:
        cx: ランドマークのx座標
        cy: ランドマークのy座標
        x1: 調整後のx1座標
        y1: 調整後のy1座標
        x1_original: 調整前のx1座標
        crop_w: クロップ幅
        crop_h: クロップ高さ
        orig_w: 元動画の幅
        orig_h: 元動画の高さ
        config: クロップ設定
        adjusted: 境界調整が行われたか
    """
    landmark_x_ratio = (cx - x1) / crop_w if crop_w > 0 else 0
    landmark_y_ratio = (cy - y1) / crop_h if crop_h > 0 else 0

    logger.info("First frame crop info:")
    logger.info(f"  Landmark position: ({cx:.1f}, {cy:.1f}) in {orig_w}x{orig_h}")
    logger.info(f"  Crop size: {crop_w}x{crop_h}")
    logger.info(
        f"  Target landmark ratio: H={config.landmark_horizontal_ratio:.2f}, V={config.landmark_vertical_ratio:.2f}"
    )
    logger.info(
        f"  Calculated x1: {x1_original:.1f} -> {x1:.1f} (adjusted: {adjusted})"
    )
    logger.info(
        f"  Actual landmark position in crop: H={landmark_x_ratio:.2f}, V={landmark_y_ratio:.2f}"
    )


def _crop_frames(
    source_clip: VideoFileClip,
    positions: List[Optional[Tuple[float, float]]],
    valid_frames: List[int],
    landmark_fps: float,
    zoom_ratio: float,
    config: CroppingConfig,
) -> List[VideoClip]:
    """
    フレームごとにクロップ処理を行う。

    Args:
        source_clip: 元の動画クリップ
        positions: ランドマーク位置のリスト
        valid_frames: 有効フレームのインデックスリスト
        landmark_fps: ランドマーク検出時のFPS
        zoom_ratio: ズーム率
        config: クロップ設定

    Returns:
        クロップされたクリップのリスト
    """
    logger.info(f"Processing {len(valid_frames)} frames with detected landmarks")

    orig_w, orig_h = source_clip.size
    crop_w = int(orig_w * zoom_ratio)
    crop_h = int(orig_h * zoom_ratio)

    logger.info(
        f"Original size: {orig_w}x{orig_h}, Crop size: {crop_w}x{crop_h} (zoom: {1 / zoom_ratio:.1f}x)"
    )

    clips = []

    for frame_idx in tqdm(valid_frames, desc="Cropping to landmark center"):
        t_start = frame_idx / landmark_fps
        t_end = min((frame_idx + 1) / landmark_fps, source_clip.duration)

        if t_end <= t_start:
            continue

        pos = positions[frame_idx]
        if pos is None:
            continue

        # ピクセル座標に変換
        cx, cy = pos[0] * orig_w, pos[1] * orig_h

        # クロップ位置を計算
        x1_original = cx - crop_w * config.landmark_horizontal_ratio
        x1, y1, adjusted = _calculate_crop_position(
            cx, cy, crop_w, crop_h, orig_w, orig_h, config
        )

        # 最初のフレームの情報をログ出力
        if frame_idx == valid_frames[0]:
            _log_first_frame_info(
                cx,
                cy,
                x1,
                y1,
                x1_original,
                crop_w,
                crop_h,
                orig_w,
                orig_h,
                config,
                adjusted,
            )

        # クロップを実行
        try:
            subclip = source_clip.subclipped(t_start, t_end)
            cropped = subclip.cropped(x1=x1, y1=y1, width=crop_w, height=crop_h)
            clips.append(cropped)
        except Exception as e:
            logger.warning(f"Failed to crop frame {frame_idx}: {e}")
            continue

    return clips


def crop_to_landmark_center(
    source_clip: VideoFileClip,
    input_movie_path: str,
    detector: LandmarkDetectorService,
    config: CroppingConfig,
    calculate_auto_zoom_fn: Optional[
        Callable[[List[Optional[Tuple[float, float]]], float], float]
    ] = None,
    smooth_positions_fn: Optional[
        Callable[
            [List[Optional[Tuple[float, float]]], int],
            List[Optional[Tuple[float, float]]],
        ]
    ] = None,
) -> Optional[Union[VideoClip, CompositeVideoClip]]:
    """
    ランドマークの位置を中心にクロップする。

    Args:
        source_clip: 元の動画クリップ
        input_movie_path: 入力動画のパス
        detector: ランドマーク検出器
        config: クロップ設定
        calculate_auto_zoom_fn: 自動ズーム率計算関数（auto_zoom=Trueの場合に必須）
        smooth_positions_fn: 位置スムージング関数（Noneの場合はスムージングなし）

    Returns:
        クロップされた動画クリップ、またはランドマークが検出されなかった場合はNone
    """
    # ステップ1: ランドマーク検出
    logger.info("Detecting landmark positions and sizes for cropping")
    positions, sizes, landmark_fps = detector.detect_positions_and_sizes(
        video_path=input_movie_path,
        fps_sample=config.fps_sample,
        min_conf=config.min_conf,
        min_area_ratio=config.min_area_ratio,
    )

    # ステップ2: ズーム率の決定
    zoom_ratio = calculate_zoom_ratio(config, sizes, calculate_auto_zoom_fn)

    # ステップ3: 位置のスムージングと有効フレーム抽出
    positions, valid_frames = _prepare_positions(positions, config, smooth_positions_fn)

    if not valid_frames:
        return None

    # ステップ4: フレームごとのクロップ処理
    clips = _crop_frames(
        source_clip, positions, valid_frames, landmark_fps, zoom_ratio, config
    )

    # ステップ5: クリップの連結
    if clips:
        logger.info(f"Concatenating {len(clips)} cropped clips")
        return concatenate_videoclips(clips, method="compose")
    else:
        logger.error("No valid clips to concatenate")
        return None
