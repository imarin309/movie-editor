"""
動画編集ツールの設定定数
"""

# MediaPipe手検出のデフォルト設定
DEFAULT_FPS_SAMPLE: int = 15  # 解析時のサンプリングFPS
DEFAULT_MIN_CONFIDENCE: float = 0.5  # MediaPipeの最小信頼度 (0-1)
DEFAULT_MIN_AREA_RATIO: float = 0.003  # 最小の手のバウンディングボックス面積比 (0.3%)

# セグメント処理のデフォルト設定
DEFAULT_MIN_KEEP_SEC: float = 1.0  # この秒数より短いセグメントを除外
DEFAULT_MERGE_GAP_SEC: float = 0.25  # 小さな隙間で区切られたセグメントを結合する秒数
# DEFAULT_PAD_SEC: float = 0.3  # 各セグメントの前後に追加するパディング秒数
DEFAULT_PAD_SEC: float = 0.8  # 各セグメントの前後に追加するパディング秒数

# クロップ位置のデフォルト設定
DEFAULT_CROP_HAND_HORIZONTAL_RATIO: float = (
    0.8  # 手を画面の左から何%の位置に配置するか (0.5=中央, 0.40=やや左寄り)
)
DEFAULT_CROP_HAND_VERTICAL_RATIO: float = (
    0.5  # 手を画面の上から何%の位置に配置するか (0.5=中央)
)
DEFAULT_SMOOTH_WINDOW_SIZE: int = 1  # 手の位置スムージングの移動平均ウィンドウサイズ

# クロップズームのデフォルト設定
DEFAULT_AUTO_ZOOM: bool = True  # 手のサイズに基づいて自動的にズーム率を調整するかどうか
DEFAULT_TARGET_HAND_RATIO: float = (
    0.15  # 自動ズーム時に手が占める目標の画面比率 (0.15 = 15%、手全体のバウンディングボックス)
)
DEFAULT_CROP_ZOOM_RATIO: float = (
    0.3  # 手動ズーム時のズーム率 (0.3=約3.3倍ズーム、0.5=2倍ズーム、小さいほど大きくズーム)
)
