from abc import ABC, abstractmethod
from typing import Any, List, Optional

from src.model import BoundingBox


class LandmarkDetectorAbstract(ABC):
    """
    ランドマーク検出器の抽象インターフェース。

    このクラスはインターフェースのみを定義し、実装を持たない。
    実装は service 層の LandmarkDetectorService と各具体的な検出器で行う。
    """

    @abstractmethod
    def _create_detector(self, min_conf: float) -> Any:
        """
        MediaPipe検出器を作成する。

        Args:
            min_conf: 最小信頼度

        Returns:
            MediaPipe検出器インスタンス
        """
        pass

    @abstractmethod
    def _make_bounding_box(self, result: Any) -> Optional[List[BoundingBox]]:
        """
        MediaPipeの検出結果からバウンディングボックスのリストを生成する。

        Args:
            result: MediaPipeの検出結果

        Returns:
            バウンディングボックスのリスト、または検出されなかった場合はNone
        """
        pass

    @abstractmethod
    def _get_selection_key(self, bounding_box: BoundingBox) -> float:
        """
        検出から最適なものを選択するための基準値を返す。

        Args:
            bounding_box: 検出されたバウンディングボックス

        Returns:
            選択基準値（大きいほど優先）
        """
        pass

    @abstractmethod
    def _get_progress_desc_for_mask(self) -> str:
        """
        マスク検出時のプログレスバー説明文を返す。

        Returns:
            プログレスバー説明文
        """
        pass

    @abstractmethod
    def _get_progress_desc_for_positions(self) -> str:
        """
        位置検出時のプログレスバー説明文を返す。

        Returns:
            プログレスバー説明文
        """
        pass
