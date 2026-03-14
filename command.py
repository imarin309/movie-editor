import argparse
import logging

from src.edit_movie_controller import edit_movie_controller
from src.extract_frames_controller import extract_frames_controller

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main() -> None:
    ap = argparse.ArgumentParser(description="動画編集ツール")
    subparsers = ap.add_subparsers(dest="command", required=True)

    edit_parser = subparsers.add_parser("edit", help="動画を編集する")
    edit_parser.add_argument("input_path", help="Input video path (mp4 etc.)")

    extract_parser = subparsers.add_parser(
        "extract-frames",
        help="動画から一定間隔でフレームを画像として保存する（手ブレ回避あり）",
    )
    extract_parser.add_argument("input_path", help="Input directory path")
    extract_parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="フレーム抽出間隔（秒）(デフォルト: 5)",
    )
    extract_parser.add_argument(
        "--window",
        type=float,
        default=1.0,
        help="手ブレ回避のための探索ウィンドウ幅（秒）(デフォルト: 1.0)",
    )

    args = ap.parse_args()

    if args.command == "edit":
        edit_movie_controller(args.input_path)
    elif args.command == "extract-frames":
        extract_frames_controller(
            input_dir=args.input_path,
            interval_sec=args.interval,
            search_window_sec=args.window,
        )


if __name__ == "__main__":
    main()
