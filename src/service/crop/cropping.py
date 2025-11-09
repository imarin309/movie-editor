import logging
from typing import Optional, Union

from moviepy import VideoFileClip, concatenate_videoclips
from moviepy.video.VideoClip import VideoClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from tqdm import tqdm

from src.service.masking.hand import (
    calculate_auto_zoom_ratio,
    detect_hands_position_and_size,
    smooth_positions,
)

logger = logging.getLogger(__name__)


def crop_to_hand_center(
    source_clip: VideoFileClip,
    input_movie_path: str,
    fps_sample: int,
    min_conf: float,
    min_area_ratio: float,
    hand_horizontal_ratio: float,
    hand_vertical_ratio: float,
    smooth_window_size: int,
    crop_zoom_ratio: float,
    auto_zoom: bool,
    target_hand_ratio: float,
) -> Optional[Union[VideoClip, CompositeVideoClip]]:
    """
    手の位置を中心にクロップする。

    Args:
        source_clip: 元の動画クリップ
        input_movie_path: 入力動画のパス
        fps_sample: サンプリングFPS
        min_conf: 最小信頼度
        min_area_ratio: 最小面積比
        hand_horizontal_ratio: 手を画面の左から何%の位置に配置するか (0.0-1.0)
        hand_vertical_ratio: 手を画面の上から何%の位置に配置するか (0.0-1.0)
        smooth_window_size: 手の位置スムージングの移動平均ウィンドウサイズ
        crop_zoom_ratio: 手動ズーム時のズーム率 (auto_zoom=Falseの場合のみ使用)
        auto_zoom: 手のサイズに基づいて自動的にズーム率を調整するかどうか
        target_hand_ratio: 自動ズーム時に手が占める目標の画面比率 (0.25 = 1/4)

    Returns:
        クロップされた動画クリップ、または手が検出されなかった場合はNone
    """
    logger.info("Detecting hand positions and sizes for cropping")

    positions, sizes, hand_fps = detect_hands_position_and_size(
        video_path=input_movie_path,
        fps_sample=fps_sample,
        min_conf=min_conf,
        min_area_ratio=min_area_ratio,
    )

    if auto_zoom:
        logger.info("Calculating auto zoom ratio based on hand size")
        auto_zoom_ratio = calculate_auto_zoom_ratio(sizes, target_hand_ratio=target_hand_ratio)
        logger.info(f"Auto zoom ratio: {auto_zoom_ratio:.3f} (zoom: {1/auto_zoom_ratio:.1f}x)")
        crop_zoom_ratio = auto_zoom_ratio
    else:
        logger.info(f"Using manual zoom ratio: {crop_zoom_ratio:.3f} (zoom: {1/crop_zoom_ratio:.1f}x)")

    logger.info("Smoothing hand positions")
    positions = smooth_positions(positions, window_size=smooth_window_size)

    # 手が検出されたフレームのインデックスを取得
    valid_frames = [i for i, pos in enumerate(positions) if pos is not None]

    if not valid_frames:
        logger.warning("No hands detected for cropping")
        return None

    logger.info(f"Processing {len(valid_frames)} frames with detected hands")

    orig_w, orig_h = source_clip.size

    # クロップサイズを計算（ズーム率に基づく）
    crop_w = int(orig_w * crop_zoom_ratio)
    crop_h = int(orig_h * crop_zoom_ratio)

    logger.info(f"Original size: {orig_w}x{orig_h}, Crop size: {crop_w}x{crop_h} (zoom: {1/crop_zoom_ratio:.1f}x)")

    clips = []

    for frame_idx in tqdm(valid_frames, desc="Cropping to hand center"):
        t_start = frame_idx / hand_fps
        t_end = min((frame_idx + 1) / hand_fps, source_clip.duration)

        if t_end <= t_start:
            continue

        pos = positions[frame_idx]
        if pos is None:  # 型チェック用（実際には None にはならない）
            continue
        cx, cy = pos[0] * orig_w, pos[1] * orig_h

        # クロップパラメータを計算
        # 手の位置からクロップ範囲の左上座標を計算
        x1_original = cx - crop_w * hand_horizontal_ratio
        y1_original = cy - crop_h * hand_vertical_ratio

        x1 = x1_original
        y1 = y1_original

        # 境界チェックと調整（可能な限り手の位置比率を維持）
        adjusted = False

        # 水平方向の調整
        if x1 < 0:
            # 左端を超える場合: x1=0に設定し、手の位置比率は変わる
            x1 = 0
            adjusted = True
        elif x1 + crop_w > orig_w:
            # 右端を超える場合: 右端に合わせる
            x1 = orig_w - crop_w
            adjusted = True

        # 垂直方向の調整
        if y1 < 0:
            y1 = 0
            adjusted = True
        elif y1 + crop_h > orig_h:
            y1 = orig_h - crop_h
            adjusted = True

        # デバッグログ（最初のフレームのみ）
        if frame_idx == valid_frames[0]:
            hand_x_ratio = (cx - x1) / crop_w if crop_w > 0 else 0
            hand_y_ratio = (cy - y1) / crop_h if crop_h > 0 else 0
            logger.info("First frame crop info:")
            logger.info(f"  Hand position: ({cx:.1f}, {cy:.1f}) in {orig_w}x{orig_h}")
            logger.info(f"  Crop size: {crop_w}x{crop_h}")
            logger.info(f"  Target hand ratio: H={hand_horizontal_ratio:.2f}, V={hand_vertical_ratio:.2f}")
            logger.info(f"  Calculated x1: {x1_original:.1f} -> {x1:.1f} (adjusted: {adjusted})")
            logger.info(f"  Actual hand position in crop: H={hand_x_ratio:.2f}, V={hand_y_ratio:.2f}")

        # サブクリップを作成してクロップ
        try:
            subclip = source_clip.subclipped(t_start, t_end)
            cropped = subclip.cropped(
                x1=int(x1), y1=int(y1), width=crop_w, height=crop_h
            )
            clips.append(cropped)
        except Exception as e:
            logger.warning(f"Failed to crop frame {frame_idx}: {e}")
            continue

    if clips:
        logger.info(f"Concatenating {len(clips)} cropped clips")
        return concatenate_videoclips(clips, method="compose")
    else:
        logger.error("No valid clips to concatenate")
        return None
