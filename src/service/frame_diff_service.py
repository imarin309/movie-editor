import math
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

import config
from src.model import VideoMetaData


class FrameDiffService:

    @staticmethod
    def extract_mask(video_meta: VideoMetaData) -> List[bool]:
        """フレーム間差分で動きがあるフレームを検出する。

        隣接サンプルフレーム間で輝度差が PIXEL_DIFF_THRESHOLD を超えた画素を「変化画素」とし、
        変化画素の割合が CHANGED_RATIO_THRESHOLD 以上のフレームを True とする。
        全体に薄く乗る圧縮ノイズ・照明チラつきを無視し、局所的な動きだけを拾う。
        先頭フレームは比較対象がないため False とする。
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
                    changed_pixels = int(
                        np.count_nonzero(diff > config.PIXEL_DIFF_THRESHOLD)
                    )
                    changed_ratio = changed_pixels / diff.size
                    mask.append(changed_ratio >= config.CHANGED_RATIO_THRESHOLD)

                prev_gray = gray
                pbar.update(1)
                idx += 1

            pbar.close()
        finally:
            video_meta.video_capture.release()

        return mask
