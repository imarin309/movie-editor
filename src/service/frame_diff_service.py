import math
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

from src.model import VideoMetaData


class FrameDiffService:

    @staticmethod
    def extract_mask(video_meta: VideoMetaData, threshold: float) -> List[bool]:
        """フレーム間差分で動きがあるフレームを検出する。

        隣接サンプルフレーム間のグレースケール差分の平均がthreshold以上のフレームをTrueとする。
        先頭フレームは比較対象がないためFalseとする。
        """
        mask: List[bool] = []
        prev_gray = None

        pbar = tqdm(
            total=math.ceil(video_meta.total_frames / video_meta.sampling_step),
            desc="detecting motion...",
            unit="f",
        )

        try:
            idx = 0
            while True:
                if not video_meta.video_capture.grab():
                    break

                if idx % video_meta.sampling_step != 0:
                    idx += 1
                    continue

                ret, frame = video_meta.video_capture.retrieve()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if prev_gray is None:
                    mask.append(False)
                else:
                    diff = cv2.absdiff(prev_gray, gray)
                    mean_diff = float(np.mean(diff)) / 255.0
                    mask.append(mean_diff >= threshold)

                prev_gray = gray
                pbar.update(1)
                idx += 1

            pbar.close()
        finally:
            video_meta.video_capture.release()

        return mask
