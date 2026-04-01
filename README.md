# movie_editor

## setup

```
poetry install
```

## Streamlit UI

GUIで操作したい場合は以下のコマンドで起動します。ブラウザが自動で開きます。

```
poetry run streamlit run app.py
```

## コマンド一覧

### edit - 動画編集

手が映っている箇所のみを抽出・カットして動画を書き出します。

```
poetry run python command.py edit {input_path}
```

- `input_path`: 動画ファイルまたはディレクトリを指定。ディレクトリの場合は配下の動画ファイルをすべて処理します。
- 出力ファイルは `output/movie/{ファイル名}_edited.{拡張子}` として生成されます（例: `video.mp4` → `output/movie/video_edited.mp4`）。

### extract-frames - フレーム抽出

動画から一定間隔でフレームを画像として保存します。手ブレが少ないフレームを自動選択します。

```
poetry run python command.py extract-frames {input_path} [--interval N] [--window N]
```

- `input_path`: 動画ファイルが含まれるディレクトリを指定。サブディレクトリも再帰的に処理します。
- 出力先: `output/image/`

| オプション | デフォルト | 説明 |
|---|---|---|
| `--interval` | 5 | フレーム抽出間隔（秒） |
| `--window` | 1.0 | 手ブレ回避のための探索ウィンドウ幅（秒） |

### edit-and-extract - 動画編集 + フレーム抽出

動画を編集してからフレームを抽出します。`edit` と `extract-frames` を連続して実行します。

```
poetry run python command.py edit-and-extract {input_path} [--interval N] [--window N]
```

- `input_path`: 動画ファイルまたはディレクトリを指定。
- 編集済み動画は `output/movie/`、抽出フレームは `output/image/` に出力されます。

| オプション | デフォルト | 説明 |
|---|---|---|
| `--interval` | 5 | フレーム抽出間隔（秒） |
| `--window` | 1.0 | 手ブレ回避のための探索ウィンドウ幅（秒） |

## ユースケース

プラモ制作中の動画編集で無駄なカットを切るのがめんどい、、、そんなときに使おう(^^)
