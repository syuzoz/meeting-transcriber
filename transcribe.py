#!/usr/bin/env python3
"""
OpenAI Whisperを使用して音声/動画ファイルを文字起こしするCLIツール
"""
import argparse
import os

import whisper


def transcribe_file(input_file: str, model_size: str = "small") -> str:
    """
    OpenAI Whisperを使い、ファイルを文字起こしして文字列として返す関数。

    Args:
        input_file: 入力ファイルのパス（音声/動画ファイル）
        model_size: Whisperモデルのサイズ（tiny/base/small/medium/large）

    Returns:
        str: 文字起こし結果のテキスト

    Raises:
        FileNotFoundError: 入力ファイルが存在しない場合
        Exception: 文字起こし処理中にエラーが発生した場合
    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"File not found: {input_file}")

    model = whisper.load_model(model_size)
    result = model.transcribe(input_file)
    return result["text"]


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパースする"""
    parser = argparse.ArgumentParser(
        description="A CLI tool to transcribe audio/video files using OpenAI Whisper."
    )
    parser.add_argument(
        "input",
        help="Path to the input file (video file such as mp4 or audio file "
             "like mp3/wav)."
    )
    parser.add_argument(
        "-m", "--model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size to use. Default=small"
    )
    return parser.parse_args()


def main() -> None:
    """メイン処理"""
    args = parse_args()

    try:
        transcription = transcribe_file(args.input, args.model)
        print("=== Transcription Result ===")
        print(transcription)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error during transcription: {e}")


if __name__ == "__main__":
    main()
