# -*- coding: utf-8 -*-
import logging
import queue
import threading
from pathlib import Path
from typing import Callable

import streamlit as st

import config
from src.edit_movie_controller import edit_movie_controller
from src.extract_frames_controller import extract_frames_controller


class _QueueHandler(logging.Handler):
    def __init__(self, q: "queue.Queue[str | None]") -> None:
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord) -> None:
        self.q.put(self.format(record))


def _run_with_log_stream(
    fn: Callable[[], None],
    log_area: st.delta_generator.DeltaGenerator,
) -> None:
    q: queue.Queue[str | None] = queue.Queue()
    handler = _QueueHandler(q)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)

    logs: list[str] = []
    error_holder: list[BaseException | None] = [None]

    def _target() -> None:
        try:
            fn()
        except Exception as e:
            error_holder[0] = e
        finally:
            q.put(None)

    t = threading.Thread(target=_target, daemon=True)
    t.start()

    with st.spinner("処理中..."):
        while True:
            try:
                msg = q.get(timeout=0.05)
                if msg is None:
                    break
                logs.append(msg)
                log_area.text("\n".join(logs))
            except queue.Empty:
                pass

    t.join()
    root.removeHandler(handler)

    if error_holder[0] is not None:
        st.error(f"エラー: {error_holder[0]}")
    else:
        st.success("完了しました")


def _edit_sliders(key_prefix: str) -> dict[str, float | int]:
    col1, col2 = st.columns(2)
    with col1:
        sampling_fps = st.slider(
            "SAMPLING_FPS", 1, 30, config.SAMPLING_FPS, key=f"{key_prefix}_fps"
        )
        center_x = st.slider(
            "CENTER_POSTION_X",
            0.0,
            1.0,
            float(config.CENTER_POSTION_X),
            step=0.05,
            key=f"{key_prefix}_cx",
        )
        center_ratio = st.slider(
            "CENTER_DETECTION_RATIO",
            0.0,
            1.0,
            float(config.CENTER_DETECTION_RATIO),
            step=0.05,
            key=f"{key_prefix}_cr",
        )
        movie_speed = st.slider(
            "MOVIE_SPEED", 1, 10, int(config.MOVIE_SPEED), key=f"{key_prefix}_speed"
        )
    with col2:
        min_keep = st.slider(
            "MIN_KEEP_SEC",
            0.1,
            5.0,
            float(config.MIN_KEEP_SEC),
            step=0.1,
            key=f"{key_prefix}_min_keep",
        )
        merge_gap = st.slider(
            "MERGE_GAP_SEC",
            0.0,
            2.0,
            float(config.MERGE_GAP_SEC),
            step=0.05,
            key=f"{key_prefix}_merge_gap",
        )
        pad_sec = st.slider(
            "PAD_SEC",
            0.0,
            3.0,
            float(config.PAD_SEC),
            step=0.1,
            key=f"{key_prefix}_pad",
        )
    return {
        "sampling_fps": sampling_fps,
        "center_x": center_x,
        "center_ratio": center_ratio,
        "movie_speed": movie_speed,
        "min_keep": min_keep,
        "merge_gap": merge_gap,
        "pad_sec": pad_sec,
    }


def _apply_config(params: dict[str, float | int]) -> None:
    config.SAMPLING_FPS = params["sampling_fps"]
    config.CENTER_POSTION_X = params["center_x"]
    config.CENTER_DETECTION_RATIO = params["center_ratio"]
    config.MOVIE_SPEED = params["movie_speed"]
    config.MIN_KEEP_SEC = params["min_keep"]
    config.MERGE_GAP_SEC = params["merge_gap"]
    config.PAD_SEC = params["pad_sec"]


def _tab_edit() -> None:
    input_path = st.text_input("ファイル/フォルダパス", key="edit_input")
    params = _edit_sliders("edit")
    log_area = st.empty()

    if st.button("実行", key="run_edit"):
        if not input_path:
            st.warning("パスを入力してください")
            return
        _apply_config(params)

        def _run() -> None:
            edit_movie_controller(input_path)

        _run_with_log_stream(_run, log_area)


def _tab_extract() -> None:
    input_path = st.text_input("ディレクトリパス", key="extract_input")
    interval = st.slider("interval（秒）", 1, 60, 5, key="extract_interval")
    window = st.slider("window（秒）", 0.1, 5.0, 1.0, step=0.1, key="extract_window")
    log_area = st.empty()

    if st.button("実行", key="run_extract"):
        if not input_path:
            st.warning("パスを入力してください")
            return

        def _run() -> None:
            extract_frames_controller(
                input_dir=input_path,
                interval_sec=interval,
                search_window_sec=window,
            )

        _run_with_log_stream(_run, log_area)


def _tab_edit_and_extract() -> None:
    input_path = st.text_input("ファイル/フォルダパス", key="eae_input")
    params = _edit_sliders("eae")

    st.subheader("フレーム抽出設定")
    interval = st.slider("interval（秒）", 1, 60, 5, key="eae_interval")
    window = st.slider("window（秒）", 0.1, 5.0, 1.0, step=0.1, key="eae_window")
    log_area = st.empty()

    if st.button("実行", key="run_eae"):
        if not input_path:
            st.warning("パスを入力してください")
            return
        _apply_config(params)

        def _run() -> None:
            edit_movie_controller(input_path)
            path = Path(input_path)
            extract_dir = str(path.parent if path.is_file() else path)
            extract_frames_controller(
                input_dir=extract_dir,
                interval_sec=interval,
                search_window_sec=window,
            )

        _run_with_log_stream(_run, log_area)


def main() -> None:
    st.set_page_config(page_title="動画編集ツール", layout="wide")
    st.title("動画編集ツール")

    tab1, tab2, tab3 = st.tabs(["動画編集", "フレーム抽出", "編集 + フレーム抽出"])

    with tab1:
        _tab_edit()
    with tab2:
        _tab_extract()
    with tab3:
        _tab_edit_and_extract()


main()
