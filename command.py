import argparse

from src.edit_movie import EditMovie


def main() -> None:
    ap = argparse.ArgumentParser(description="")
    ap.add_argument("input", help="Input video path (mp4 etc.)")
    ap.add_argument(
        "-i", "--ignore_head_detect", action="store_true", help="頭の検出を行わない"
    )
    args = ap.parse_args()

    edit_movie = EditMovie(args.input, args.ignore_head_detect)
    edit_movie.run()


if __name__ == "__main__":
    main()
