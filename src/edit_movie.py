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
            fps_sample=config.SAMPLING_FPS,
            center_detection_ratio=config.CENTER_DETECTION_RATIO,
            center_postion_x=config.CENTER_POSTION_X,
            movie_speed=config.MOVIE_SPEED,
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
        )
        self.segments = SegmentService.clamp_segments_to_duration(
            segments, self.duration
        )

    def _concat_movie(self) -> None:
        clips = []
        for segment in self.segments:
            start = max(0.0, min(segment.start, self.source_clip.duration))
            end = max(0.0, min(segment.end, self.source_clip.duration))
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
        self.output_movie = self.output_movie.with_effects(
            [vfx.MultiplySpeed(self.config.movie_speed)]
        )

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
