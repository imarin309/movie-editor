import math
from typing import List, Optional

from src.model import BoundingBox, Segment


class MotionScoreService:

    @staticmethod
    def apply_motion_filter(
        mask: List[bool],
        bounding_boxes: List[Optional[BoundingBox]],
        threshold: float,
    ) -> List[bool]:
        """静止フレームを除外する（基準2）

        隣接フレーム間のBoundingBox中心座標の移動量がthreshold未満のフレームをFalseにする。
        """
        frame_motions = MotionScoreService._compute_frame_motions(bounding_boxes)
        return [
            mask[i] and bounding_boxes[i] is not None and frame_motions[i] >= threshold
            for i in range(len(mask))
        ]

    @staticmethod
    def assign_motion_scores_to_segments(
        segments: List[Segment],
        bounding_boxes: List[Optional[BoundingBox]],
        effective_fps: float,
    ) -> List[Segment]:
        """各セグメントに手の動き量スコアを付与する（基準4の前処理）"""
        n = len(bounding_boxes)
        frame_motions = MotionScoreService._compute_frame_motions(bounding_boxes)

        result = []
        for segment in segments:
            start_idx = max(0, min(int(segment.start * effective_fps), n))
            end_idx = max(0, min(int(segment.end * effective_fps), n))
            segment_motions = [frame_motions[i] for i in range(start_idx, end_idx)]
            score = (
                sum(segment_motions) / len(segment_motions) if segment_motions else 0.0
            )
            result.append(
                Segment(start=segment.start, end=segment.end, motion_score=score)
            )

        return result

    @staticmethod
    def select_segments_by_target_duration(
        segments: List[Segment],
        target_duration: int,
    ) -> List[Segment]:
        """スコア上位のセグメントを目標時間に達するまで選択する（基準4）"""
        sorted_by_score = sorted(
            segments,
            key=lambda s: s.motion_score if s.motion_score is not None else 0.0,
            reverse=True,
        )

        selected = []
        total = 0.0
        for segment in sorted_by_score:
            if total >= target_duration:
                break
            selected.append(segment)
            total += segment.end - segment.start

        return sorted(selected, key=lambda s: s.start)

    @staticmethod
    def _compute_frame_motions(
        bounding_boxes: List[Optional[BoundingBox]],
    ) -> List[float]:
        """各フレームの移動量（前フレームとの中心座標の距離）を計算"""
        motions = []
        for i, bb in enumerate(bounding_boxes):
            prev = bounding_boxes[i - 1] if i > 0 else None
            if bb is None or prev is None:
                motions.append(0.0)
                continue
            dx = bb.center_x - prev.center_x
            dy = bb.center_y - prev.center_y
            motions.append(math.sqrt(dx * dx + dy * dy))
        return motions
