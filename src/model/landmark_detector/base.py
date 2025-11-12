import math
from abc import ABC, abstractmethod
from typing import Any, Generator, List, Optional, Tuple

import cv2
from tqdm import tqdm

from src.model import BoundingBox
from src.service.editing_movie.carculate_sampling import get_effective_fps
from src.service.setup_video import setup_video_capture


class LandmarkDetector(ABC):
    
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

    def _is_valid_detection(
        self, bounding_box: BoundingBox, min_area_ratio: float
    ) -> bool:
        """
        検出されたバウンディングボックスが有効かどうかを判定する。

        デフォルトでは面積比が閾値以上の場合に有効とみなす。
        必要に応じてサブクラスでオーバーライド可能。

        Args:
            bounding_box: 検出されたバウンディングボックス
            min_area_ratio: 最小面積比

        Returns:
            有効な検出の場合True、そうでない場合False
        """
        return bounding_box.area >= min_area_ratio

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

    def _select_best_detection(
        self, bounding_boxes: List[BoundingBox], min_area_ratio: float
    ) -> Optional[BoundingBox]:
        """
        複数の検出から最適なものを選択する。

        有効な検出の中から、_get_selection_key()の値が最大のものを選択する。

        Args:
            bounding_boxes: 検出されたバウンディングボックスのリスト
            min_area_ratio: 最小面積比

        Returns:
            選択されたバウンディングボックス、または有効な検出がない場合はNone
        """
        best_detection = None
        best_key = -float("inf")

        for bounding_box in bounding_boxes:
            # 有効な検出のみを候補とする
            if self._is_valid_detection(bounding_box, min_area_ratio):
                key = self._get_selection_key(bounding_box)
                if key > best_key:
                    best_key = key
                    best_detection = bounding_box

        return best_detection

    def _process_video_frames(
        self,
        video_path: str,
        fps_sample: int,
        min_conf: float,
        progress_desc: str,
    ) -> Generator[Tuple[int, Optional[List[BoundingBox]]], None, Tuple[float, int]]:
        """
        動画フレームを順次処理し、各フレームのランドマーク検出結果を yield する。

        Args:
            video_path: 動画ファイルのパス
            fps_sample: サンプリングFPS
            min_conf: MediaPipeの最小信頼度
            progress_desc: プログレスバーの説明文

        Yields:
            (frame_index, bounding_boxes) のタプル
            - frame_index: フレームのインデックス
            - bounding_boxes: 検出されたバウンディングボックスのリスト（検出されない場合はNone）

        Returns:
            (eff_fps, processed_frames) のタプル
        """
        cap, metadata = setup_video_capture(video_path)
        step, eff_fps = get_effective_fps(metadata.orig_fps, fps_sample)

        detector = self._create_detector(min_conf)

        try:
            idx = 0
            processed_frames = 0
            pbar = tqdm(
                total=math.ceil(metadata.total_frames / step),
                desc=progress_desc,
                unit="f",
            )

            with detector:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if idx % step != 0:
                        idx += 1
                        continue

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = detector.process(rgb)

                    bounding_boxes = self._make_bounding_box(result)

                    yield processed_frames, bounding_boxes

                    pbar.update(1)
                    processed_frames += 1
                    idx += 1

            pbar.close()
        finally:
            cap.release()

        return eff_fps, processed_frames

    def detect_mask(
        self,
        video_path: str,
        fps_sample: int,
        min_conf: float,
        min_area_ratio: float,
    ) -> Tuple[List[bool], float]:
        """
        動画を読み、フレームごとに「ランドマークが検出されたか」をTrue/Falseで返す。

        Args:
            video_path: 動画ファイルのパス
            fps_sample: サンプリングFPS
            min_conf: MediaPipeの最小信頼度
            min_area_ratio: 最小バウンディングボックス面積比

        Returns:
            mask: 各フレームの検出結果（True/False）のリスト
            eff_fps: 実効FPS
        """
        mask: List[bool] = []

        generator = self._process_video_frames(
            video_path, fps_sample, min_conf, self._get_progress_desc_for_mask()
        )

        for _, bounding_boxes in generator:
            has_detection = False

            if bounding_boxes is not None:
                # 有効な検出があるかチェック
                for bounding_box in bounding_boxes:
                    if self._is_valid_detection(bounding_box, min_area_ratio):
                        has_detection = True
                        break  # 1つでも十分

            mask.append(has_detection)

        # ジェネレーターの戻り値を取得
        try:
            eff_fps, _ = generator.send(None)
        except StopIteration as e:
            eff_fps, _ = e.value

        return mask, eff_fps

    def detect_positions_and_sizes(
        self,
        video_path: str,
        fps_sample: int,
        min_conf: float,
        min_area_ratio: float,
    ) -> Tuple[
        List[Optional[Tuple[float, float]]],
        List[Optional[Tuple[float, float]]],
        float,
    ]:
        """
        動画を読み、フレームごとにランドマークの中心座標とサイズを返す。

        Args:
            video_path: 動画ファイルのパス
            fps_sample: サンプリングFPS
            min_conf: MediaPipeの最小信頼度
            min_area_ratio: 最小バウンディングボックス面積比

        Returns:
            positions: 各フレームの中心座標 (x, y) のリスト。検出されない場合は None
            sizes: 各フレームのサイズ (width, height) のリスト。検出されない場合は None
            eff_fps: 実効FPS
        """
        positions: List[Optional[Tuple[float, float]]] = []
        sizes: List[Optional[Tuple[float, float]]] = []

        generator = self._process_video_frames(
            video_path, fps_sample, min_conf, self._get_progress_desc_for_positions()
        )

        for _, bounding_boxes in generator:
            position = None
            size = None

            if bounding_boxes is not None:
                # 最適な検出を選択
                best_detection = self._select_best_detection(
                    bounding_boxes, min_area_ratio
                )

                if best_detection is not None:
                    position = (best_detection.center_x, best_detection.center_y)
                    size = (best_detection.width, best_detection.height)

            positions.append(position)
            sizes.append(size)

        # ジェネレーターの戻り値を取得
        try:
            eff_fps, _ = generator.send(None)
        except StopIteration as e:
            eff_fps, _ = e.value

        return positions, sizes, eff_fps

    def detect_positions(
        self,
        video_path: str,
        fps_sample: int,
        min_conf: float,
        min_area_ratio: float,
    ) -> Tuple[List[Optional[Tuple[float, float]]], float]:
        """
        動画を読み、フレームごとにランドマークの中心座標を返す。

        Args:
            video_path: 動画ファイルのパス
            fps_sample: サンプリングFPS
            min_conf: MediaPipeの最小信頼度
            min_area_ratio: 最小バウンディングボックス面積比

        Returns:
            positions: 各フレームの中心座標 (x, y) のリスト。検出されない場合は None
            eff_fps: 実効FPS
        """
        positions, _, eff_fps = self.detect_positions_and_sizes(
            video_path, fps_sample, min_conf, min_area_ratio
        )
        return positions, eff_fps

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


def smooth_positions(
    positions: List[Optional[Tuple[float, float]]],
    window_size: int = 5,
) -> List[Optional[Tuple[float, float]]]:
    """
    指定された位置をスムージングする（移動平均）。

    Args:
        positions: 位置のリスト（None は検出されなかったフレーム）
        window_size: 移動平均のウィンドウサイズ

    Returns:
        スムージングされた位置のリスト
    """
    smoothed: List[Optional[Tuple[float, float]]] = []

    for i, pos in enumerate(positions):
        if pos is None:
            smoothed.append(None)
            continue

        # ウィンドウ内の有効な位置を収集
        valid_positions: List[Tuple[float, float]] = []
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(positions), i + window_size // 2 + 1)

        for j in range(start_idx, end_idx):
            p = positions[j]
            if p is not None:
                valid_positions.append(p)

        if valid_positions:
            # 平均を計算
            avg_x = sum(p[0] for p in valid_positions) / len(valid_positions)
            avg_y = sum(p[1] for p in valid_positions) / len(valid_positions)
            smoothed.append((avg_x, avg_y))
        else:
            smoothed.append(pos)

    return smoothed
