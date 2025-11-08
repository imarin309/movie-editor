import argparse
import logging

from moviepy import VideoFileClip, concatenate_videoclips, vfx

from src.service.masking.hand import detect_hands_mask
from src.service.segment.main import bools_to_segments, clamp_segments

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EditMovie:

    def __init__(self, args: argparse.Namespace) -> None:
        self.input_movie_path = args.input
        self.output_movie_path = args.output
        self.fps_sample = args.fps_sample
        self.min_conf = args.min_conf
        self.min_area_ratio = args.min_area_ratio
        self.debug_draw = args.debug_draw
        self.debug_out = args.debug_out
        self.min_keep_sec = args.min_keep_sec
        self.pad_sec = args.pad_sec
        self.merge_gap_sec = args.merge_gap_sec
        with VideoFileClip(args.input) as probe:
            self.duration = probe.duration

    def _make_mask(self) -> None:
        self.mask, self.eff_fps = detect_hands_mask(
            video_path=self.input_movie_path,
            fps_sample=self.fps_sample,
            min_conf=self.min_conf,
            min_area_ratio=self.min_area_ratio,
            debug_draw=self.debug_draw,
            debug_out=self.debug_out,
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
                    audio_codec="aac",
                    temp_audiofile="__temp_aac.m4a",
                    remove_temp=True,
                )
            return

    def _concat_movie(self) -> None:
        # withを使わずにクリップを保持（後で明示的にcloseする）
        self.source_clip = VideoFileClip(self.input_movie_path)
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
            audio_codec="aac",
            temp_audiofile="__temp_aac.m4a",
            remove_temp=True,
        )
        # リソースのクリーンアップ
        self.output_movie.close()
        if hasattr(self, 'source_clip'):
            self.source_clip.close()

    def _change_speed(self) -> None:
        self.output_movie = self.output_movie.with_effects([vfx.MultiplySpeed(3.0)])

    def run(self) -> None:

        self._make_mask()
        self._make_segment()
        self._clip_movie()
        self._concat_movie()
        self._change_speed()
        self._output()
