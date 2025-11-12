from typing import Tuple

def get_effective_fps(original_fps: float, sampling_fps: int) -> Tuple[int, float]:
    """
    実行するfgsを抽出する

    Args:
        original_fps: 元動画のFPS
        fps_sample: サンプリングFPS

    Returns:
        (step, eff_fps) のタプル
    """
    if original_fps > 0:
        step = max(1, int(round(original_fps / sampling_fps)))
        effective_fps = original_fps / step
    else:
        step = 1 
        effective_fps = sampling_fps
    return step, effective_fps