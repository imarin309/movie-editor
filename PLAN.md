# 複数長さの動画圧縮 実装計画

目標: 30分動画 → 10分・5分・1分の3バージョンを自動生成

## 方針

- 既存コードをなるべく変えない
- 1ステップずつレビューしながら進める
- 音声なし・タイムラプス形式

---

## 速度倍率について

- **現状**: `MOVIE_SPEED` const で固定倍率（現在3倍）を定義している
- **方針**: 速度変更機能はいったんなしにする
  - 対応は `MOVIE_SPEED = 1` に変えるだけなので、**最後のステップで行う**
- **将来**: 動画をインプットにして速度だけを変える機能を別途追加する予定

---

## カット基準

| 基準 | 内容 | 対象バージョン |
|------|------|-------------|
| 基準1 | 手の存在（既存実装） | 全バージョン |
| 基準2 | 手の動き量（静止フレームをカット） | 5分版 |
| 基準4 | セグメントスコア選別（アクティブ区間を優先） | 1分版 |

---

## 既存コードの問題点（コードレビュー結果）

### 🔴 デッドコード（バグ）: 手検出フィルタが機能していない

`extract_mask()` が単純な `bool(bounding_box)` チェックのみで、
`_is_valid_detection` / `_select_best_detection` / `_get_selection_key` が呼ばれていない。

→ `CENTER_POSTION_X` / `CENTER_DETECTION_RATIO` が手検出に**効いていない**。

### 🟡 動画ファイルの重複オープン（パフォーマンス）

`_setup()` で `VideoFileClip` を2回オープン（duration取得のためだけに2回目）。
`_detect_hand()` / `_detect_head()` でさらに `cv2.VideoCapture` を各1回 → 合計4回。

→ `self.duration = self.source_clip.duration` で1回に減らせる。

### 🟡 セグメント定数が config.py に存在しない

`segment_service.py` 内の `MIN_KEEP_SEC`, `MERGE_GAP_SEC`, `PAD_SEC` が
CLAUDE.md の記載（「すべてのパラメータは config.py で定義」）と矛盾している。

### 🟡 ディレクトリ処理でファイル種別フィルタなし

`edit_movie_controller.py` の `path.iterdir()` が動画以外も対象にしてしまう。

---

## TODO

### ステップ0: 既存負債の修正 ✅ 完了

- [x] `_setup()` の重複オープンを1回に削減
- [x] `extract_mask()` に `_is_valid_detection` / `_select_best_detection` を組み込む（デッドコード解消）
- [x] `segment_service.py` の定数を `config.py` に移動
- [x] `edit_movie_controller.py` に動画ファイルフィルタを追加

### ステップ1: モデル変更

- [ ] `src/model/video/segment.py` に `motion_score: Optional[float] = None` を追加

### ステップ2: config 追加

- [ ] `config.py` に以下を追加
  - `TARGET_DURATIONS: list[int] = [600, 300, 60]`
  - `MOTION_THRESHOLD: float = 0.02`

### ステップ3: 手のBoundingBox取得メソッド追加

- [ ] `LandmarkDetectorService` に `get_selected_bounding_boxes() -> List[Optional[BoundingBox]]` を追加
  - `extract_mask()` 呼び出し後に各フレームの最適な BB を返すだけ

### ステップ4: MotionScoreService 新規作成

- [ ] `src/service/motion_score_service.py` を作成
  - `apply_motion_filter()`: 静止フレームを除外（基準2）
  - `assign_motion_scores_to_segments()`: セグメントにスコアを付与（基準4の前処理）
  - `select_segments_by_target_duration()`: スコア上位セグメントを選択（基準4）
- [ ] `src/service/__init__.py` に追加

### ステップ5: EditMovie を複数ターゲット対応に変更

- [ ] `run()` に `target_durations: List[int]` 引数を追加
- [ ] 出力ファイル名を `{stem}_{label}{suffix}` 形式に変更（例: `input_10min.mp4`）
- [ ] 各ターゲットに応じてフィルタ戦略を切り替え
  - 最長ターゲット → 基準1のみ
  - 中間ターゲット → 基準1 + 基準2
  - 最短ターゲット → 基準1 + 基準4（スコア選別）

### ステップ6: command.py / controller 更新

- [ ] `command.py` に `--targets` オプション追加（秒数のリスト）
- [ ] `edit_movie_controller.py` に `target_durations` 引数を追加

### ステップ7: 速度倍率を無効化（最終）

- [ ] `config.py` の `MOVIE_SPEED = 3` を `MOVIE_SPEED = 1` に変更


---

## 出力例

```
input.mp4
  → input_10min.mp4  （基準1）
  → input_5min.mp4   （基準1 + 基準2）
  → input_1min.mp4   （基準1 + 基準4）
```


1. スキップフレームを read() でデコードしている（最大の損失）

  src/service/bounding_boxes_service.py:41-43

  if idx % video_meta.sampling_step != 0:
      idx += 1
      continue  # ← デコード済みのフレームを捨てている

  cv2.VideoCapture.read() は毎回フルデコードを行う。スキップするフレームには
  grab()（デコードしないシーク）を使えば大幅に速くなる。

  # 改善案
  for _ in range(sampling_step - 1):
      cap.grab()  # デコードせずに進む
  ret, frame = cap.read()

  2. 手・頭の検出で動画を2回フルスキャン

  src/edit_movie.py:65-87

  _detect_hand() と _detect_head() が独立して VideoService.get_video_meta() を呼び、それぞれ別の cv2.VideoCapture
  を開いて全フレームをスキャンしている。1回のループで両方を同時に処理すれば半分の時間になる。

  3. フレームをリサイズせず MediaPipe に渡している

  src/service/detector/hand_detector_service.py:45-46

  rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  result = self.detector.process(rgb)  # フルサイズのまま

  4K動画なら検出前に 640x360 程度にリサイズするだけで数倍速くなる。

  ---
  🔴 バグ（デッドコード）

  4. HeadDetectorService._is_valid_detection が LandmarkDetectorService と重複定義

  src/service/detector/head_detector_service.py:238-246

  HeadDetectorService は LandmarkDetectorService を継承せず、_is_valid_detection
  を独自に再定義している。継承階層から外れているため、親クラスの改修が反映されない。

  5. HeadDetectorService._make_bounding_box の引数名が result

  src/service/detector/head_detector_service.py:138

  引数名は result: Any だが実体はフレーム（BGR画像）。frame にすべき。

  ---
  🟡 設計上の問題

  6. VideoMetaData に cv2.VideoCapture（状態を持つオブジェクト）を入れている

  src/model/video/video_meta.py:9

  データクラスがリソース（VideoCapture）を保持しており、release() を呼ぶ責務が BoundingBoxesService
  に漏れ出している（finally: video_meta.video_capture.release()）。モデル層がリソース管理をするのは設計の混濁。

  7. segment_service.py が config を直接 import

  src/service/segment_service.py:4

  import config  # ← service 層が command 層相当の config に依存

  CLAUDE.md の依存方向ルール（command.py → edit_movie.py → service/ → model/）に違反。定数は引数として渡すべき。

  8. EditMovie クラス変数宣言が実質インスタンス変数

  src/edit_movie.py:24-42

  class EditMovie:
      hand_mask: List[bool]   # ← クラス変数として宣言
      head_mask: List[bool]

  型アノテーションのみで __init__ で初期化されていない。_detect_hand() が呼ばれる前にアクセスするとエラー。

  ---
  🟤 コード品質の問題

  9. デバッグログが本番コードに残っている

  src/service/detector/head_detector_service.py:175-204

  if self.frame_count < 10:
      logger.info(f"Frame {self.frame_count}: dark_ratio=...")

  最初の10フレームのみ詳細ログを出す処理が本番コードに残っており、logger.debug にすべき。

  10. edit_movie_controller.py のディレクトリ処理が再帰しない

  src/edit_movie_controller.py:28-29

  path.iterdir() はトップレベルのみ。サブディレクトリ内の動画は処理されない（意図的かは不明）。