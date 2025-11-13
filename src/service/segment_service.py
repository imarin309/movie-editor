from typing import List

from model.video.segment import Segment


class SegmentService:

    @classmethod
    def create_segments_from_mask(
        cls,
        mask: List[bool],
        fps: float,
        min_keep_sec: float,
        pad_sec: float,
        merge_gap_sec: float,
    ) -> List[Segment]:
        """True/False マスクから保持セグメント(秒)へ変換し、フィルタリング・結合・パディングを行う

        Args:
            mask: フレームごとのTrue/Falseマスク
            fps: マスクのFPS
            min_keep_sec: この秒数より短いセグメントを除外
            pad_sec: 各セグメントの前後に追加するパディング秒数
            merge_gap_sec: 小さな隙間で区切られたセグメントを結合する秒数

        Returns:
            処理されたセグメントのリスト
        """
        raw_segments = cls._convert_mask_to_raw_segments(mask, fps)
        filtered_segments = cls._filter_short_segments(raw_segments, min_keep_sec)
        merged_segments = cls._merge_close_segments(filtered_segments, merge_gap_sec)
        return cls._add_padding(merged_segments, pad_sec)

    @classmethod
    def clamp_segments_to_duration(
        cls, segments: List[Segment], duration: float
    ) -> List[Segment]:
        """セグメントの時間範囲を動画の長さに収める

        Args:
            segments: クランプ対象のセグメントリスト
            duration: 動画の全体の長さ（秒）

        Returns:
            開始/終了時刻が 0.0 ~ duration の範囲内に制限されたセグメントリスト
            終了時刻が開始時刻以下になったセグメントは除外される
        """
        clamped = []
        for s in segments:
            start = max(0.0, min(s.start, duration))
            end = max(0.0, min(s.end, duration))
            if end > start:
                clamped.append(Segment(start, end))
        return clamped

    @staticmethod
    def _convert_mask_to_raw_segments(mask: List[bool], fps: float) -> List[Segment]:
        """マスクから生のセグメントリストを抽出（フィルタリングなし）

        Args:
            mask: フレームごとのTrue/Falseマスク
            fps: マスクのFPS

        Returns:
            Trueが連続する区間をセグメントに変換したリスト
        """
        segments: List[Segment] = []
        n = len(mask)
        i = 0
        while i < n:
            if not mask[i]:
                i += 1
                continue
            j = i + 1
            while j < n and mask[j]:
                j += 1
            start = i / fps
            end = j / fps
            segments.append(Segment(start, end))
            i = j
        return segments

    @staticmethod
    def _filter_short_segments(
        segments: List[Segment], min_keep_sec: float
    ) -> List[Segment]:
        """短すぎるセグメントを除外

        Args:
            segments: フィルタリング対象のセグメント
            min_keep_sec: 最小保持秒数

        Returns:
            min_keep_sec以上の長さを持つセグメントのみ
        """
        return [s for s in segments if (s.end - s.start) >= min_keep_sec]

    @staticmethod
    def _merge_close_segments(
        segments: List[Segment], merge_gap_sec: float
    ) -> List[Segment]:
        """近接するセグメントを結合

        Args:
            segments: 結合対象のセグメント
            merge_gap_sec: この秒数以下の隙間があるセグメントを結合

        Returns:
            結合されたセグメントのリスト
        """
        if not segments:
            return []

        merged: List[Segment] = []
        current = segments[0]

        for segment in segments[1:]:
            if (segment.start - current.end) <= merge_gap_sec:
                # 隙間が小さいので結合
                current = Segment(current.start, segment.end)
            else:
                # 隙間が大きいので現在のセグメントを保存
                merged.append(current)
                current = segment

        merged.append(current)
        return merged

    @staticmethod
    def _add_padding(segments: List[Segment], pad_sec: float) -> List[Segment]:
        return [Segment(max(0.0, s.start - pad_sec), s.end + pad_sec) for s in segments]
