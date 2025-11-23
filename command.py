import argparse

from src.edit_movie_controller import edit_movie_controller


def main() -> None:
    ap = argparse.ArgumentParser(description="")
    ap.add_argument("input_path", help="Input video path (mp4 etc.)")
    ap.add_argument(
        "-i", "--is_ignore_head_detect", action="store_true", help="頭の検出を行わない"
    )
    args = ap.parse_args()

    edit_movie_controller(args.input_path, args.is_ignore_head_detect)


if __name__ == "__main__":
    main()
