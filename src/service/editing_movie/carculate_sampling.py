from typing import Tuple

def calculate_sampling(orig_fps: float, fps_sample: int) -> Tuple[int, float]:
    """
    フレームサンプリングのステップと実効FPSを計算する。

    Args:
        orig_fps: 元動画のFPS
        fps_sample: サンプリングFPS

    Returns:
        (step, eff_fps) のタプル
    """
    step = max(1, int(round(orig_fps / fps_sample))) if orig_fps and orig_fps > 0 else 1
    eff_fps = (orig_fps / step) if orig_fps and orig_fps > 0 else fps_sample
    return step, eff_fps