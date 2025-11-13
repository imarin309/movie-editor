from dataclasses import dataclass


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
