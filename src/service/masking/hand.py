import math
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
from tqdm import tqdm


def detect_hands_mask(
    video_path: str,
    fps_sample: int,
    min_conf: float,
    min_area_ratio: float,
    debug_draw: bool = False,
    debug_out: Optional[str] = None,
) -> Tuple[List[bool], float]:
    """
    動画を読み、フレームごとに「手が検出されたか」を True/False で返す。
    - fps_sample で解析（元動画 FPS と異なっても OK、マスクは fps_sample 基準で返す）
    - min_conf を下回る検出は無視
    - 手のバウンディングボックス面積が画面に対し min_area_ratio 未満ならノイズ扱い
    - ROI は使わず、画面全体で判定
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 解析間引き
    step = max(1, int(round(orig_fps / fps_sample))) if orig_fps and orig_fps > 0 else 1
    eff_fps = (orig_fps / step) if orig_fps and orig_fps > 0 else fps_sample

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=min_conf,
        min_tracking_confidence=min_conf,
    )

    mask: List[bool] = []

    # デバッグ動画出力
    dbg_writer = None
    if debug_draw and debug_out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        dbg_writer = cv2.VideoWriter(debug_out, fourcc, eff_fps, (width, height))

    with hands:
        idx = 0
        pbar = tqdm(
            total=math.ceil(total_frames / step), desc="Detecting hands", unit="f"
        )
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if idx % step != 0:
                idx += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            has_hand = False
            if result.multi_hand_landmarks and result.multi_handedness:
                for hand_landmarks in result.multi_hand_landmarks:
                    xs = [lm.x for lm in hand_landmarks.landmark]
                    ys = [lm.y for lm in hand_landmarks.landmark]
                    minx, maxx = min(xs), max(xs)
                    miny, maxy = min(ys), max(ys)

                    # 画像座標系へ
                    bx1 = int(minx * width)
                    by1 = int(miny * height)
                    bx2 = int(maxx * width)
                    by2 = int(maxy * height)
                    bw = max(1, bx2 - bx1)
                    bh = max(1, by2 - by1)
                    area_ratio = (bw * bh) / float(width * height)

                    if area_ratio >= min_area_ratio:
                        has_hand = True
                        break  # 1つでも十分

            mask.append(has_hand)

            if dbg_writer is not None:
                # 可視化（単純にKEEP/CUTをオーバーレイ）
                cv2.putText(
                    frame,
                    f"{'KEEP' if has_hand else 'CUT'}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0) if has_hand else (0, 0, 255),
                    2,
                )
                dbg_writer.write(frame)

            pbar.update(1)
            idx += 1

        pbar.close()
    cap.release()
    if dbg_writer is not None:
        dbg_writer.release()

    return mask, eff_fps
