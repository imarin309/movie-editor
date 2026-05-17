"""
処理が重い時: SAMPLING_FPSを下げる
対象物の検出区間を設定する：CENTER_DETECTION_RATIOをいじる
"""

SAMPLING_FPS: int = 1
CENTER_POSTION_X: float = 0.7
CENTER_DETECTION_RATIO: float = 0.3
MOVIE_SPEED = 1

# セグメント処理の設定
MIN_KEEP_SEC: float = 5.0
MERGE_GAP_SEC: float = 0.25
PAD_SEC: float = 1.2

# フレーム間差分の動き判定
# PIXEL_DIFF_THRESHOLD: 1画素あたりの輝度差。これ以上動いた画素を「変化画素」と数える
# CHANGED_RATIO_THRESHOLD: 「変化画素 / 全画素」がこの割合を超えたフレームを動きありと判定
PIXEL_DIFF_THRESHOLD: int = 25
CHANGED_RATIO_THRESHOLD: float = 0.005
