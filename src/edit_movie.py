import logging
from pathlib import Path

from moviepy import VideoFileClip, concatenate_videoclips, vfx

import config
from src.service.crop import crop_to_hand_center
from src.service.masking.hand import detect_hands_mask
from src.service.segment.main import bools_to_segments, clamp_segments

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EditMovie:

    def __init__(self, input_movie_path: str) -> None:
        self.input_movie_path = input_movie_path
        input_path = Path(input_movie_path)
        output_filename = f"{input_path.stem}_edited{input_path.suffix}"
        self.output_movie_path = str(input_path.parent / output_filename)

        logger.info(f"Input: {self.input_movie_path}")
        logger.info(f"Output: {self.output_movie_path}")

        self.fps_sample = config.DEFAULT_FPS_SAMPLE
        self.min_conf = config.DEFAULT_MIN_CONFIDENCE
        self.min_area_ratio = config.DEFAULT_MIN_AREA_RATIO
        self.min_keep_sec = config.DEFAULT_MIN_KEEP_SEC
        self.pad_sec = config.DEFAULT_PAD_SEC
        self.merge_gap_sec = config.DEFAULT_MERGE_GAP_SEC
        self.hand_horizontal_ratio = config.DEFAULT_CROP_HAND_HORIZONTAL_RATIO
        self.hand_vertical_ratio = config.DEFAULT_CROP_HAND_VERTICAL_RATIO
        self.smooth_window_size = config.DEFAULT_SMOOTH_WINDOW_SIZE
        self.auto_zoom = config.DEFAULT_AUTO_ZOOM
        self.target_hand_ratio = config.DEFAULT_TARGET_HAND_RATIO
        self.crop_zoom_ratio = config.DEFAULT_CROP_ZOOM_RATIO
        self.source_clip = VideoFileClip(self.input_movie_path)
        with VideoFileClip(input_movie_path) as probe:
            self.duration = probe.duration

    def _make_mask(self) -> None:
        self.mask, self.eff_fps = detect_hands_mask(
            video_path=self.input_movie_path,
            fps_sample=self.fps_sample,
            min_conf=self.min_conf,
            min_area_ratio=self.min_area_ratio,
        )

    def _make_segment(self) -> None:
        segments = bools_to_segments(
            mask=self.mask,
            fps=self.eff_fps,
            min_keep_sec=self.min_keep_sec,
            pad_sec=self.pad_sec,
            merge_gap_sec=self.merge_gap_sec,
        )
        self.segments = clamp_segments(segments, self.duration)

    def _clip_movie(self) -> None:
        if not self.segments:
            logger.warning(
                "No hand segments found. Writing tiny clip to avoid empty output."
            )
            with VideoFileClip(self.input_movie_path) as clip:
                sub = clip.subclipped(1, clip.duration)
                sub.write_videofile(
                    self.output_movie_path,
                    codec="libx264",
                    audio=False,
                )
            return

    def _concat_movie(self) -> None:
        clips = []
        for s in self.segments:
            start = max(0.0, min(s.start, self.source_clip.duration))
            end = max(0.0, min(s.end, self.source_clip.duration))
            if end > start:
                clips.append(self.source_clip.subclipped(start, end))

        self.output_movie = concatenate_videoclips(clips, method="compose")

    def _output(self) -> None:
        self.output_movie.write_videofile(
            self.output_movie_path,
            codec="libx264",
            audio=False,
        )

    def _clean(self) -> None:
        self.output_movie.close()
        if hasattr(self, "source_clip"):
            self.source_clip.close()

    def _crop_to_hand_center(self) -> None:
        """
        手の位置を中心にクロップする。
        - 手が検出されたフレームのみを保持
        - スムージングを適用してカメラの動きを滑らかに
        - 手のサイズに基づいて自動的にズーム率を調整（デフォルト：手が画面の1/4を占めるように）
        """
        self.output_movie = crop_to_hand_center(
            source_clip=self.source_clip,
            input_movie_path=self.input_movie_path,
            fps_sample=self.fps_sample,
            min_conf=self.min_conf,
            min_area_ratio=self.min_area_ratio,
            hand_horizontal_ratio=self.hand_horizontal_ratio,
            hand_vertical_ratio=self.hand_vertical_ratio,
            smooth_window_size=self.smooth_window_size,
            crop_zoom_ratio=self.crop_zoom_ratio,
            auto_zoom=self.auto_zoom,
            target_hand_ratio=self.target_hand_ratio,
        )

    def _change_speed(self) -> None:
        self.output_movie = self.output_movie.with_effects([vfx.MultiplySpeed(3.0)])

    def run(self) -> None:

        self._make_mask()
        self._make_segment()
        self._clip_movie()
        self._concat_movie()
        self._crop_to_hand_center()
        self._change_speed()
        self._output()
        self._clean()
