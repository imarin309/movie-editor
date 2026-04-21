# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FrameExtractService:

    @staticmethod
    def extract_frames(
        input_movie_path: str,
        output_dir: Path,
        interval_sec: int,
        search_window_sec: float,
    ) -> List[Path]:
        """動画から一定間隔で手ブレが少ないフレームを抽出して画像として保存する。

        各インターバル時点の前後 search_window_sec 秒の範囲でフレームを探索し、
        ラプラシアン分散（鮮明度スコア）が最も高いフレームを選択することで
        手ブレ・ピンボケの影響を回避する。
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(input_movie_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {input_movie_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps

        timestamps = list(range(0, int(duration_sec), interval_sec))
        search_radius = max(1, int(search_window_sec * fps))

        saved_paths: List[Path] = []

        try:
            for ts in tqdm(timestamps, desc="extracting frames..."):
                target_frame = int(ts * fps)
                start_frame = max(0, target_frame - search_radius)
                end_frame = min(total_frames - 1, target_frame + search_radius)

                best_frame: np.ndarray | None = None
                best_score = -1.0

                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                for _ in range(end_frame - start_frame + 1):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    score = FrameExtractService._sharpness_score(frame)
                    if score > best_score:
                        best_score = score
                        best_frame = frame.copy()

                if best_frame is not None:
                    output_path = output_dir / f"frame_{ts:06d}.jpg"
                    # cv2.imwrite はWindowsで非ASCII文字を含むパスで無音失敗するため imencode+write_bytes を使う
                    ok, buf = cv2.imencode(".jpg", best_frame)
                    if not ok:
                        logger.warning(f"フレームのエンコードに失敗しました: {output_path}")
                        continue
                    output_path.write_bytes(buf.tobytes())
                    saved_paths.append(output_path)
                    logger.info(f"saved: {output_path} (sharpness={best_score:.2f})")
        finally:
            cap.release()

        return saved_paths

    @staticmethod
    def _sharpness_score(frame: np.ndarray) -> float:
        """ラプラシアン分散で鮮明度スコアを計算する（値が大きいほど鮮明）"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
