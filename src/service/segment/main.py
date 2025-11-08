from dataclasses import dataclass
from typing import List


@dataclass
class Segment:
    start: float  # 秒
    end: float  # 秒


def bools_to_segments(
    mask: List[bool],
    fps: float,
    min_keep_sec: float,
    pad_sec: float,
    merge_gap_sec: float,
) -> List[Segment]:
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


def clamp_segments(segments: List[Segment], duration: float) -> List[Segment]:
    out = []
    for s in segments:
        start = max(0.0, min(s.start, duration))
        end = max(0.0, min(s.end, duration))
        if end > start:
            out.append(Segment(start, end))
    return out
