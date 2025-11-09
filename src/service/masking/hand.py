import math
from dataclasses import dataclass
from typing import Any, Generator, List, Optional, Tuple

import cv2
import mediapipe as mp
from tqdm import tqdm


@dataclass
class _VideoMetadata:
    """動画のメタデータを保持するデータクラス"""

    width: int
    height: int
    orig_fps: float
    total_frames: int


@dataclass
class _HandBoundingBox:
    """手のバウンディングボックス情報"""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    width: float
    height: float
    area: float
    center_x: float
    center_y: float


def _setup_video_capture(video_path: str) -> Tuple[cv2.VideoCapture, _VideoMetadata]:
    """
    VideoCaptureを初期化し、動画のメタデータを取得する。

    Args:
        video_path: 動画ファイルのパス

    Returns:
        初期化されたVideoCaptureオブジェクトとメタデータ

    Raises:
        RuntimeError: 動画を開けない場合
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    metadata = _VideoMetadata(
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        orig_fps=cap.get(cv2.CAP_PROP_FPS),
        total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )

    return cap, metadata


def _calculate_sampling(orig_fps: float, fps_sample: int) -> Tuple[int, float]:
    """
    フレームサンプリングのステップと実効FPSを計算する。

    Args:
        orig_fps: 元動画のFPS
        fps_sample: サンプリングFPS

    Returns:
        (step, eff_fps) のタプル
    """
    step = max(1, int(round(orig_fps / fps_sample))) if orig_fps and orig_fps > 0 else 1
    eff_fps = (orig_fps / step) if orig_fps and orig_fps > 0 else fps_sample
    return step, eff_fps


def _calculate_hand_bbox(hand_landmarks: Any) -> _HandBoundingBox:
    """
    手のランドマークからバウンディングボックスを計算する。

    Args:
        hand_landmarks: MediaPipeの手のランドマーク

    Returns:
        手のバウンディングボックス情報
    """
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    width = x_max - x_min
    height = y_max - y_min
    area = width * height

    return _HandBoundingBox(
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        width=width,
        height=height,
        area=area,
        center_x=(x_min + x_max) / 2,
        center_y=(y_min + y_max) / 2,
    )


def _process_video_frames(
    video_path: str,
    fps_sample: int,
    min_conf: float,
    progress_desc: str,
) -> Generator[Tuple[int, Optional[List[_HandBoundingBox]]], None, Tuple[float, int]]:
    """
    動画フレームを順次処理し、各フレームの手検出結果を yield する。

    Args:
        video_path: 動画ファイルのパス
        fps_sample: サンプリングFPS
        min_conf: MediaPipeの最小信頼度
        progress_desc: プログレスバーの説明文

    Yields:
        (frame_index, hand_bounding_boxes) のタプル
        - frame_index: フレームのインデックス
        - hand_bounding_boxes: 検出された手のバウンディングボックスのリスト（検出されない場合はNone）

    Returns:
        (eff_fps, processed_frames) のタプル
    """
    cap, metadata = _setup_video_capture(video_path)
    step, eff_fps = _calculate_sampling(metadata.orig_fps, fps_sample)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=min_conf,
        min_tracking_confidence=min_conf,
    )

    try:
        idx = 0
        processed_frames = 0
        pbar = tqdm(
            total=math.ceil(metadata.total_frames / step), desc=progress_desc, unit="f"
        )

        with hands:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if idx % step != 0:
                    idx += 1
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                hand_bounding_boxes = None
                if result.multi_hand_landmarks and result.multi_handedness:
                    hand_bounding_boxes = [
                        _calculate_hand_bbox(hand_landmarks)
                        for hand_landmarks in result.multi_hand_landmarks
                    ]

                yield processed_frames, hand_bounding_boxes

                pbar.update(1)
                processed_frames += 1
                idx += 1

        pbar.close()
    finally:
        cap.release()

    return eff_fps, processed_frames


def detect_hands_mask(
    video_path: str,
    fps_sample: int,
    min_conf: float,
    min_area_ratio: float,
) -> Tuple[List[bool], float]:
    """
    動画を読み、フレームごとに「手が検出されたか」を True/False で返す。
    - fps_sample で解析（元動画 FPS と異なっても OK、マスクは fps_sample 基準で返す）
    - min_conf を下回る検出は無視
    - 手全体のバウンディングボックスを計算して判定
    - ROI は使わず、画面全体で判定
    - min_area_ratio で小さすぎる検出を除外

    Returns:
        mask: 各フレームの手の検出結果（True/False）のリスト
        eff_fps: 実効FPS
    """
    mask: List[bool] = []

    generator = _process_video_frames(
        video_path, fps_sample, min_conf, "Detecting hands"
    )

    for _, hand_bounding_boxes in generator:
        has_hand = False

        if hand_bounding_boxes is not None:
            # 面積比が閾値以上の手があるかチェック
            for bounding_box in hand_bounding_boxes:
                if bounding_box.area >= min_area_ratio:
                    has_hand = True
                    break  # 1つでも十分

        mask.append(has_hand)

    # ジェネレーターの戻り値を取得
    try:
        eff_fps, _ = generator.send(None)
    except StopIteration as e:
        eff_fps, _ = e.value

    return mask, eff_fps


def detect_hands_position_and_size(
    video_path: str,
    fps_sample: int,
    min_conf: float,
    min_area_ratio: float,
) -> Tuple[
    List[Optional[Tuple[float, float]]], List[Optional[Tuple[float, float]]], float
]:
    """
    動画を読み、フレームごとに手の中心座標とサイズを返す。
    - fps_sample で解析
    - 手が検出されない場合は None を返す
    - 複数の手が検出された場合は最初の手の座標を使用
    - 座標とサイズは正規化された値 (0.0-1.0)
    - 手全体のバウンディングボックスから中心とサイズを計算
    - min_area_ratio で小さすぎる検出を除外

    Returns:
        positions: 各フレームの手の中心座標 (x, y) のリスト。検出されない場合は None
        sizes: 各フレームの手のサイズ (width, height) のリスト。検出されない場合は None
        eff_fps: 実効FPS
    """
    positions: List[Optional[Tuple[float, float]]] = []
    sizes: List[Optional[Tuple[float, float]]] = []

    generator = _process_video_frames(
        video_path, fps_sample, min_conf, "Detecting hand positions"
    )

    for _, hand_bounding_boxes in generator:
        hand_pos = None
        hand_size = None

        if hand_bounding_boxes is not None:
            # 複数の手が検出された場合は、最も右側（x座標が大きい）の手を使用
            best_hand = None
            best_center_x = -1.0

            for bounding_box in hand_bounding_boxes:
                # 面積比が閾値以上の手のみを候補とする
                if bounding_box.area >= min_area_ratio:
                    # より右側の手を選択
                    if bounding_box.center_x > best_center_x:
                        best_center_x = bounding_box.center_x
                        best_hand = bounding_box

            # 有効な手が見つかった場合のみ設定
            if best_hand is not None:
                hand_pos = (best_hand.center_x, best_hand.center_y)
                hand_size = (best_hand.width, best_hand.height)

        positions.append(hand_pos)
        sizes.append(hand_size)

    # ジェネレーターの戻り値を取得
    try:
        eff_fps, _ = generator.send(None)
    except StopIteration as e:
        eff_fps, _ = e.value

    return positions, sizes, eff_fps


def detect_hands_position(
    video_path: str,
    fps_sample: int,
    min_conf: float,
    min_area_ratio: float,
) -> Tuple[List[Optional[Tuple[float, float]]], float]:
    """
    動画を読み、フレームごとに手の中心座標を返す。
    - fps_sample で解析
    - 手が検出されない場合は None を返す
    - 複数の手が検出された場合は最初の手の座標を使用
    - 座標は正規化された値 (0.0-1.0)

    Returns:
        positions: 各フレームの手の中心座標 (x, y) のリスト。検出されない場合は None
        eff_fps: 実効FPS
    """
    positions, _, eff_fps = detect_hands_position_and_size(
        video_path, fps_sample, min_conf, min_area_ratio
    )
    return positions, eff_fps


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


def smooth_positions(
    positions: List[Optional[Tuple[float, float]]],
    window_size: int = 5,
) -> List[Optional[Tuple[float, float]]]:
    """
    手の位置をスムージングする（移動平均）。

    Args:
        positions: 手の位置のリスト（None は手が検出されなかったフレーム）
        window_size: 移動平均のウィンドウサイズ

    Returns:
        スムージングされた位置のリスト
    """
    smoothed: List[Optional[Tuple[float, float]]] = []

    for i, pos in enumerate(positions):
        if pos is None:
            smoothed.append(None)
            continue

        # ウィンドウ内の有効な位置を収集
        valid_positions: List[Tuple[float, float]] = []
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(positions), i + window_size // 2 + 1)

        for j in range(start_idx, end_idx):
            p = positions[j]
            if p is not None:
                valid_positions.append(p)

        if valid_positions:
            # 平均を計算
            avg_x = sum(p[0] for p in valid_positions) / len(valid_positions)
            avg_y = sum(p[1] for p in valid_positions) / len(valid_positions)
            smoothed.append((avg_x, avg_y))
        else:
            smoothed.append(pos)

    return smoothed
