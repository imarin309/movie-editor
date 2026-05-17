import logging
from pathlib import Path
from typing import List

from moviepy import VideoClip, VideoFileClip, concatenate_videoclips, vfx

import config
from src.model import Config, Segment, VideoMetaData
from src.service import FrameDiffService, SegmentService, VideoService

# from src.service import HandDetectorService

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EditMovie:

    input_movie_path: str
    output_movie_path: str
    config: Config

    source_clip: VideoFileClip
    duration: float
    video_meta: VideoMetaData

    # hand_mask: List[bool]  # Step2 ポーズ検出で再利用予定
    active_mask: List[bool]
    segments: List[Segment]

    output_movie: VideoClip

    def __init__(self, input_movie_path: str) -> None:
        self.input_movie_path = input_movie_path
        input_path = Path(input_movie_path)
        output_filename = f"{input_path.stem}{input_path.suffix}"
        output_dir = input_path.parent / "output" / "movie"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_movie_path = str(output_dir / output_filename)

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
        self.duration = self.source_clip.duration

    def _detect_active(self) -> None:

        # TODO: 安定したら手のフレーム検出処理を削除
        # logger.info("手のフレームを検出")
        # hand_video_meta = VideoService.get_video_meta(
        #     self.input_movie_path, self.config.fps_sample
        # )
        # self.video_meta = hand_video_meta
        # hand_detector = HandDetectorService(config=self.config, video_meta=hand_video_meta)
        # self.hand_mask = hand_detector.extract_mask()

        logger.info("動きフレームを検出")
        motion_video_meta = VideoService.get_video_meta(
            self.input_movie_path, self.config.fps_sample
        )
        self.video_meta = motion_video_meta
        self.active_mask = FrameDiffService.extract_mask(motion_video_meta)

        # self.active_mask = [h or m for h, m in zip(self.hand_mask, self.active_mask)]

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

        self.output_movie = concatenate_videoclips(clips, method="chain")

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
        self._detect_active()
        self._make_segment(self.active_mask)
        if not self.segments:
            logger.info("対象物が検出されませんでした。終了します。")
            return
        self._concat_movie()
        if self.config.movie_speed != 1:
            self._change_speed()
        self._output()
        self._clean()
