#!/usr/bin/env python3
import argparse
import os
import math
import whisper
import numpy as np
from tqdm import tqdm

class AudioChunkLoader:
    def __init__(self, audio_path, chunk_sec=30, sample_rate=16000):
        self.audio_path = audio_path
        self.chunk_sec = chunk_sec
        self.sample_rate = sample_rate
        self.full_audio = whisper.load_audio(audio_path)
        self.total_samples = len(self.full_audio)
        self.chunk_samples = chunk_sec * sample_rate
        self.num_chunks = math.ceil(self.total_samples / self.chunk_samples)

    def __iter__(self):
        for i in range(self.num_chunks):
            start = i * self.chunk_samples
            end = (i + 1) * self.chunk_samples
            chunk_audio = self.full_audio[start:end]
            chunk_audio = whisper.pad_or_trim(chunk_audio, self.chunk_samples)
            yield i, self.num_chunks, chunk_audio

class WhisperTranscriber:
    def __init__(self, model_size="base", device="cpu"):
        self.model = whisper.load_model(model_size, device=device)
        self.device = device

    def detect_language(self, audio):
        mel = whisper.log_mel_spectrogram(audio).unsqueeze(0).to(self.device)
        _, probs = self.model.detect_language(mel)
        if isinstance(probs, dict):
            return max(probs, key=probs.get)
        return probs[0]

    def transcribe_chunk(self, chunk_audio):
        mel = whisper.log_mel_spectrogram(chunk_audio).unsqueeze(0).to(self.device)
        options = whisper.DecodingOptions(fp16=False)
        result = self.model.decode(mel, options)
        if isinstance(result, list):
            return result[0].text.strip()
        return result.text.strip()

    def transcribe_file(self, audio_path, chunk_sec=30):
        chunks = AudioChunkLoader(audio_path, chunk_sec)
        all_texts = []

        # 言語検出
        _, _, first_chunk = next(iter(chunks))
        detected_lang = self.detect_language(first_chunk)
        print(f"Detected language: {detected_lang}")

        # 文字起こし（最初のチャンクも処理するために、chunksを再初期化）
        chunks = AudioChunkLoader(audio_path, chunk_sec)
        for i, num_chunks, chunk_audio in tqdm(chunks, total=chunks.num_chunks, desc="Transcribing", unit="chunk"):
            text = self.transcribe_chunk(chunk_audio)
            all_texts.append(text)

        return "\n".join(all_texts)

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio/video using Whisper on Apple Silicon (MPS).")
    parser.add_argument("input", help="Input file path (e.g., .mp4, .wav, .mp3)")
    parser.add_argument("-o", "--output", default=None, help="Path to output text file")
    parser.add_argument("-m", "--model", default="base", help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--chunk_sec", type=int, default=30, help="Chunk length in seconds (default: 30)")
    parser.add_argument("--device", default="cpu", help="Device to use for inference (default: cpu)")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: File not found -> {args.input}")
        return

    transcriber = WhisperTranscriber(model_size=args.model, device=args.device)
    text_result = transcriber.transcribe_file(args.input, chunk_sec=args.chunk_sec)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text_result)
        print(f"\nTranscription saved to: {args.output}")
    else:
        print("\n=== Transcription Result ===")
        print(text_result)

if __name__ == "__main__":
    main()
