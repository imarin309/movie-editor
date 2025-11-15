import logging
from pathlib import Path

from moviepy import VideoFileClip, concatenate_videoclips, vfx

import config
from src.model import Config
from src.service.detector import HandDetectorService
from src.service.segment_service import SegmentService

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

        self.config = Config(
            fps_sample=config.DEFAULT_FPS_SAMPLE,
            min_conf=config.DEFAULT_MIN_CONFIDENCE,
            min_area_ratio=config.DEFAULT_MIN_AREA_RATIO,
            min_keep_sec=config.DEFAULT_MIN_KEEP_SEC,
            pad_sec=config.DEFAULT_PAD_SEC,
            merge_gap_sec=config.DEFAULT_MERGE_GAP_SEC,
            landmark_horizontal_ratio=config.DEFAULT_CROP_HAND_HORIZONTAL_RATIO,
            landmark_vertical_ratio=config.DEFAULT_CROP_HAND_VERTICAL_RATIO,
            smooth_window_size=config.DEFAULT_SMOOTH_WINDOW_SIZE,
            auto_zoom=config.DEFAULT_AUTO_ZOOM,
            target_landmark_ratio=config.DEFAULT_TARGET_HAND_RATIO,
            crop_zoom_ratio=config.DEFAULT_CROP_ZOOM_RATIO,
        )

    def _setup(self) -> None:
        self.source_clip = VideoFileClip(self.input_movie_path)
        with VideoFileClip(self.input_movie_path) as probe:
            self.duration = probe.duration
        self.detector = HandDetectorService(
            video_path=self.input_movie_path, config=self.config
        )

    def _detect_hand(self) -> None:
        self.detector.extract_landmark_info()

    def _make_segment(self) -> None:
        segments = SegmentService.create_segments_from_mask(
            mask=self.detector.landmark_info.has_landmark_frame,
            fps=self.detector.effective_fps,
            min_keep_sec=self.config.min_keep_sec,
            pad_sec=self.config.pad_sec,
            merge_gap_sec=self.config.merge_gap_sec,
        )
        self.segments = SegmentService.clamp_segments_to_duration(
            segments, self.duration
        )

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

    def _change_speed(self) -> None:
        self.output_movie = self.output_movie.with_effects([vfx.MultiplySpeed(3.0)])

    def run(self) -> None:

        self._setup()
        self._detect_hand()
        self._make_segment()
        if not self.segments:
            logger.info("対象物が検出されませんでした。終了します。")
            return
        self._concat_movie()
        self._change_speed()
        self._output()
        self._clean()
