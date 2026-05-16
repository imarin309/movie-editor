# packy

## setup

```
poetry install
```

手の検出に使用する MediaPipe のモデルファイルをダウンロードします。

**macOS / Linux**
```bash
curl -L "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" \
  -o models/hand_landmarker.task --create-dirs
```

**Windows (PowerShell)**
```powershell
New-Item -ItemType Directory -Force -Path models | Out-Null
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" -OutFile "models\hand_landmarker.task"
```

## コマンド一覧

### edit - 動画編集

手が映っている箇所のみを抽出・カットして動画を書き出します。

```
poetry run python command.py edit {input_path}

# 例
poetry run python command.py edit data
```

- `input_path`: 動画ファイルまたはディレクトリを指定。ディレクトリの場合は配下の動画ファイルをすべて処理します。
- 出力ファイルは入力ファイルと同じディレクトリの `output/movie/` 配下に生成されます。

  ```
  /path/to/video.mp4  →  /path/to/output/movie/video.mp4
  ```

### extract-frames - フレーム抽出

動画から一定間隔でフレームを画像として保存します。手ブレが少ないフレームを自動選択します。

```
poetry run python command.py extract-frames {input_path} [--interval N] [--window N]

# 例
poetry run python command.py extract-frames data
```

- `input_path`: 動画ファイルが含まれるディレクトリを指定。サブディレクトリも再帰的に処理します。
- 抽出した画像は入力ディレクトリの `output/image/` 配下に保存されます。

  ```
  /path/to/dir/  →  /path/to/dir/output/image/frame_000000.jpg
  ```

| オプション | デフォルト | 説明 |
|---|---|---|
| `--interval` | 5 | フレーム抽出間隔（秒） |
| `--window` | 1.0 | 手ブレ回避のための探索ウィンドウ幅（秒） |

### split-frames - 均等分割フレーム抽出

1本の動画をN等分し、各区間から手ブレが少ないフレームを1枚ずつ画像として保存します。
36枚フィルムのように動画全体を均等にカバーした画像セットを作りたい場合に使います。

```
poetry run python command.py split-frames {input_path} [--count N]

# 例（デフォルト36枚）
poetry run python command.py split-frames data/video.mp4

# 枚数を指定する場合
poetry run python command.py split-frames data/video.mp4 --count 24
```

- `input_path`: 動画ファイルを直接指定します。
- 抽出した画像は動画ファイルと同じディレクトリの `output/image/{動画名}/` 配下に保存されます。

  ```
  /path/to/video.mp4  →  /path/to/output/image/video/frame_001.jpg
                          /path/to/output/image/video/frame_002.jpg
                          ...
                          /path/to/output/image/video/frame_036.jpg
  ```

| オプション | デフォルト | 説明 |
|---|---|---|
| `--count` | 36 | 分割枚数 |
| `--crop` | 1.0 | 横幅中央からクロップする比率（例: `0.6` で中央60%） |

### edit-and-extract - 動画編集 + フレーム抽出

動画を編集してからフレームを抽出します。`edit` と `extract-frames` を連続して実行します。

```
poetry run python command.py edit-and-extract {input_path} [--interval N] [--window N]

# example
poetry run python command.py edit-and-extract data
```

- `input_path`: 動画ファイルまたはディレクトリを指定。
- 編集済み動画・抽出フレームともに、入力ファイルの親ディレクトリ配下に出力されます。

  ```
  /path/to/video.mp4  →  /path/to/output/movie/video.mp4
                          /path/to/output/image/frame_000000.jpg
  ```

| オプション | デフォルト | 説明 |
|---|---|---|
| `--interval` | 5 | フレーム抽出間隔（秒） |
| `--window` | 1.0 | 手ブレ回避のための探索ウィンドウ幅（秒） |

## ユースケース

プラモ制作中の動画編集で無駄なカットを切るのがめんどい、、、そんなときに使おう(^^)
