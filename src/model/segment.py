from dataclasses import dataclass
from typing import List

@dataclass
class Segment:
    """動画の中で保持したい時間区間を表すクラス

    手が検出された区間など、動画の特定の時間範囲を表現するために使用される。
    複数のセグメントを組み合わせることで、動画の必要な部分だけを切り出して連結できる。

    Attributes:
        start: 区間の開始時刻（秒）
        end: 区間の終了時刻（秒）

    Example:
        >>> segment = Segment(start=10.5, end=25.3)
        >>> # 10.5秒から25.3秒までの区間を表す
    """

    start: float
    end: float

    @classmethod
    def bools_to_segments(
        cls,
        mask: List[bool],
        fps: float,
        min_keep_sec: float,
        pad_sec: float,
        merge_gap_sec: float,
    ) -> List["Segment"]:
        """True/False マスクから保持セグメント（秒）へ変換し、短すぎる区間除去・パディング・小さい隙間の結合を行う"""
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

        # 短すぎる区間を除外
        segments = [s for s in segments if (s.end - s.start) >= min_keep_sec]

        # 近接区間の結合
        merged: List[Segment] = []
        if segments:
            cur = segments[0]
            for s in segments[1:]:
                if (s.start - cur.end) <= merge_gap_sec:
                    cur = Segment(cur.start, s.end)
                else:
                    merged.append(cur)
                    cur = s
            merged.append(cur)

        # パディング付与
        padded = [Segment(max(0.0, s.start - pad_sec), s.end + pad_sec) for s in merged]
        return padded

    @classmethod
    def clamp_segments(cls, segments: List["Segment"], duration: float) -> List["Segment"]:
        """セグメントの時間範囲を動画の長さに収める

        Args:
            segments: クランプ対象のセグメントリスト
            duration: 動画の全体の長さ（秒）

        Returns:
            開始/終了時刻が 0.0 ~ duration の範囲内に制限されたセグメントリスト
            終了時刻が開始時刻以下になったセグメントは除外される
        """
        out = []
        for s in segments:
            start = max(0.0, min(s.start, duration))
            end = max(0.0, min(s.end, duration))
            if end > start:
                out.append(Segment(start, end))
        return out