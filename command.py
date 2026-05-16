import argparse
import logging
from pathlib import Path

from src.edit_movie_controller import edit_movie_controller
from src.extract_frames_controller import extract_frames_controller
from src.split_frames_controller import split_frames_controller

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

    split_parser = subparsers.add_parser(
        "split-frames",
        help="1本の動画をN等分して各区間から1枚ずつ画像を保存する（デフォルト: 36枚）",
    )
    split_parser.add_argument("input_path", help="Input video file path (mp4 etc.)")
    split_parser.add_argument(
        "--count",
        type=int,
        default=36,
        help="分割枚数 (デフォルト: 36)",
    )
    split_parser.add_argument(
        "--crop",
        type=float,
        default=1.0,
        help="横幅中央からクロップする比率 0.0〜1.0 (デフォルト: 1.0 = クロップなし)",
    )

    edit_extract_parser = subparsers.add_parser(
        "edit-and-extract",
        help="動画を編集してからフレームを抽出する",
    )
    edit_extract_parser.add_argument("input_path", help="Input video/directory path")
    edit_extract_parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="フレーム抽出間隔（秒）(デフォルト: 5)",
    )
    edit_extract_parser.add_argument(
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
    elif args.command == "split-frames":
        split_frames_controller(
            input_path=args.input_path,
            n_frames=args.count,
            center_crop_ratio=args.crop,
        )
    elif args.command == "edit-and-extract":
        edit_movie_controller(args.input_path)
        input_path = Path(args.input_path)
        input_dir = input_path.parent if input_path.is_file() else input_path
        extract_frames_controller(
            input_dir=str(input_dir / "output" / "movie"),
            interval_sec=args.interval,
            search_window_sec=args.window,
        )


if __name__ == "__main__":
    main()
