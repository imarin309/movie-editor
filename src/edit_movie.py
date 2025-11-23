import logging
from pathlib import Path
from typing import List

from moviepy import VideoClip, VideoFileClip, concatenate_videoclips, vfx

import config
from src.model import Config, Segment, VideoMetaData
from src.service.detector import HandDetectorService
from src.service.detector.head_detector_service import HeadDetectorService
from src.service.segment_service import SegmentService
from src.service.video_service import VideoService

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EditMovie:

    input_movie_path: str
    output_movie_path: str
    is_ignore_head_detect: bool
    config: Config

    source_clip: VideoFileClip
    duration: float
    video_meta: VideoMetaData

    hand_detector: HandDetectorService
    head_detector: HeadDetectorService

    hand_mask: List[bool]
    head_mask: List[bool]
    combined_mask: List[bool]

    segments: List[Segment]

    output_movie: VideoClip

    def __init__(self, input_movie_path: str, is_ignore_head_detect: bool) -> None:
        self.input_movie_path = input_movie_path
        input_path = Path(input_movie_path)
        output_filename = f"{input_path.stem}_edited{input_path.suffix}"
        self.output_movie_path = str(input_path.parent / output_filename)
        self.is_ignore_head_detect = is_ignore_head_detect

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

    def _detect_hand(self) -> None:
        """手と顔を検出し、手があり顔がないフレームのマスクを生成"""
        logger.info("手のフレームを検出")
        hand_video_meta = VideoService.get_video_meta(
            self.input_movie_path, self.config.fps_sample
        )
        # TODO: video_metaの変数を共通化する
        self.video_meta = hand_video_meta
        self.hand_detector = HandDetectorService(
            config=self.config,
            video_meta=hand_video_meta,
        )
        self.hand_detector.extract_landmark_info()
        self.hand_mask = self.hand_detector.landmark_info.has_landmark_frame

    def _detect_head(self) -> None:
        head_video_meta = VideoService.get_video_meta(
            self.input_movie_path, self.config.fps_sample
        )
        self.head_detector = HeadDetectorService(
            config=self.config,
            video_meta=head_video_meta,
        )
        self.head_detector.extract_landmark_info()
        self.head_mask = self.head_detector.landmark_info.has_landmark_frame

    def _combined_mask(self) -> None:
        self.combined_mask = [
            has_hand and not has_head
            for has_hand, has_head in zip(self.hand_mask, self.head_mask)
        ]

    def _make_segment(self, mask: List[bool]) -> None:
        segments = SegmentService.create_segments_from_mask(
            mask=mask,
            fps=self.video_meta.effective_fps,
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
        if not self.is_ignore_head_detect:
            self._detect_head()
            self._combined_mask()
            mask = self.combined_mask
        else:
            mask = self.hand_mask
        self._make_segment(mask)
        if not self.segments:
            logger.info("対象物が検出されませんでした。終了します。")
            return
        self._concat_movie()
        self._change_speed()
        self._output()
        self._clean()
