import math
from typing import Any, Callable, List, Union

from tqdm import tqdm

from src.model import BoundingBox, BoundingBoxes, VideoMetaData


class BoundingBoxesService:

    @staticmethod
    def make_bounding_boxes(
        video_meta: VideoMetaData,
        frame_processor: Callable[[Any], Union[BoundingBox, List[BoundingBox], None]],
    ) -> BoundingBoxes:
        """
        動画から各フレームのバウンディングボックスを作成する。

        Args:
            video_meta: 動画のメタデータ
            frame_processor: フレームからBoundingBoxを作成する関数

        Returns:
            バウンディングボックスのリスト
        """
        bounding_boxes: BoundingBoxes = []

        try:
            idx = 0
            pbar = tqdm(
                total=math.ceil(video_meta.total_frames / video_meta.sampling_step),
                desc="detecting...",
                unit="f",
            )

            while True:
                ret, frame = video_meta.video_capture.read()
                if not ret:
                    break

                if idx % video_meta.sampling_step != 0:
                    idx += 1
                    continue

                bounding_box = frame_processor(frame)
                bounding_boxes.append(bounding_box)

                pbar.update(1)
                idx += 1

            pbar.close()
        finally:
            video_meta.video_capture.release()

        return bounding_boxes
