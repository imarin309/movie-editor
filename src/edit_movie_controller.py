import logging
from pathlib import Path

from src.edit_movie import EditMovie

logger = logging.getLogger(__name__)



def _run_edit_movie(input_movie_path: str, is_ignore_head_detect: bool) -> None:
    edit_movie = EditMovie(input_movie_path, is_ignore_head_detect)
    edit_movie.run()


def edit_movie_controller(target_path: str, is_ignore_head_detect: bool = False) -> None:
    path = Path(target_path)

    if not path.exists():
        raise FileNotFoundError(f"target path is invalid: {target_path}")

    if path.is_file():
        logger.info(f"[1/1]exe:  {target_path}")
        _run_edit_movie(str(path), is_ignore_head_detect)

    elif path.is_dir():
        video_files = [f for f in path.iterdir()]

        if not video_files:
            logger.warning(f"target path is invalid: {target_path}")
            return

        for idx, video_file in enumerate(video_files, start=1):
            logger.info(f"[{idx}/{len(video_files)}]exe:  {video_file}")
            _run_edit_movie(str(video_file), is_ignore_head_detect)
        logger.info("complete")
    else:
        raise ValueError(f"target_path is invalid: {target_path}")