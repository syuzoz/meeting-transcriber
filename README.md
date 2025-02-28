# meeting-transcriber

## 概要

動画・音声ファイルから文字起こしを行うツールです。OpenAIの[Whisper](https://github.com/openai/whisper)を使用して音声認識を実現します。

主な特徴：
- 複数の音声認識モデルに対応（tiny, base, small, medium, large）
- 長時間の音声ファイルを自動的にチャンク分割して処理
- 自動言語検出機能
- Docker対応で環境構築が容易

## ディレクトリ構成

```
meeting-transcriber/
├── README.md           # このファイル
├── Dockerfile          # Dockerイメージ定義
├── requirements.txt    # Pythonパッケージ依存関係
└── transcribe.py       # メインスクリプト
└── 入力ファイル          # 文字起こし対象のファイル
```

## 使用方法

### 1. ビルド

```bash
docker build -t meeting-transcriber .
```

### 2. 実行

```bash
docker run --rm -v "$(pwd)":/data meeting-transcriber /data/入力ファイル \
    --output /data/出力ファイル.txt \
    --model モデルサイズ \
    --chunk_sec チャンク秒数 \
    --device デバイス
```

### パラメータ説明

- `入力ファイル`: 文字起こしする動画・音声ファイル（.mp4, .wav, .mp3など）
- `--output`: 文字起こし結果の出力先（指定しない場合は標準出力）
- `--model`: Whisperモデルのサイズ（デフォルト: base）
  - 選択肢: tiny, base, small, medium, large-v2
- `--chunk_sec`: 分割するチャンクの長さ（秒）（デフォルト: 30）
- `--device`: 推論に使用するデバイス（デフォルト: cpu）
  - 選択肢: cpu, cuda（NVIDIAのGPUがある場合）

### 使用例

```bash
# mediumモデルを使用して文字起こし
docker run --rm -v "$(pwd)":/data meeting-transcriber /data/meeting.mp4 \
    --output /data/meeting.txt \
    --model medium

# チャンクサイズを60秒に設定
docker run --rm -v "$(pwd)":/data meeting-transcriber /data/meeting.mp4 \
    --output /data/meeting.txt \
    --chunk_sec 60
```