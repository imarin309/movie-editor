import argparse

from src.edit_movie import EditMovie


def main():
    ap = argparse.ArgumentParser(
        description="Detect hands (proxy for brush work) and cut video accordingly. (No ROI)"
    )
    ap.add_argument("input", help="Input video path (mp4 etc.)")
    args = ap.parse_args()

    edit_movie = EditMovie(args.input)
    edit_movie.run()


if __name__ == "__main__":
    main()
