import argparse

from src.edit_movie import EditMovie


def main():
    ap = argparse.ArgumentParser(
        description="Detect hands (proxy for brush work) and cut video accordingly. (No ROI)"
    )
    ap.add_argument("input", help="Input video path (mp4 etc.)")
    ap.add_argument("output", help="Output video path (mp4)")
    ap.add_argument(
        "--fps-sample",
        type=int,
        default=15,
        help="FPS to sample for analysis (default: 15)",
    )
    ap.add_argument(
        "--min-conf",
        type=float,
        default=0.5,
        help="MediaPipe min confidence (0-1, default: 0.5)",
    )
    ap.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.003,
        help="Min hand bbox area / frame area (default: 0.003 = 0.3%%)",
    )
    ap.add_argument(
        "--min-keep-sec",
        type=float,
        default=1.0,
        help="Drop segments shorter than this (seconds)",
    )
    ap.add_argument(
        "--merge-gap-sec",
        type=float,
        default=0.25,
        help="Merge segments separated by small gaps (seconds)",
    )
    ap.add_argument(
        "--pad-sec",
        type=float,
        default=0.3,
        help="Padding seconds added before/after each segment",
    )
    ap.add_argument(
        "--debug-draw",
        action="store_true",
        help="Write debug overlay video for inspection",
    )
    ap.add_argument(
        "--debug-out",
        default="debug_detection.mp4",
        help="Path for debug video (when --debug-draw)",
    )
    args = ap.parse_args()

    edit_movie = EditMovie(args)
    edit_movie.run()


if __name__ == "__main__":
    main()
