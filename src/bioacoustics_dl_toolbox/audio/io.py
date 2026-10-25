"""
Audio file loading utilities.

Ported from ANIMAL-SPOT ``data/transforms.py`` (Bergler & Schroeter, GPL-3.0).
"""

from __future__ import annotations

import os
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import resampy
import soundfile as sf
import torch

_PCM_COPY_CHUNK_BYTES = 64 * 1024


class PcmExtractionCancelled(InterruptedError):
    """Raised when a caller-provided PCM extraction cancellation check fires."""


@dataclass(frozen=True)
class PcmSegmentFacts:
    """Source-frame and format facts for one extracted PCM WAV segment."""

    start_frame: int
    end_frame: int
    sample_rate_hz: int
    channels: int
    sample_width_bytes: int
    frame_count: int


def _nearest_frame(milliseconds: int, sample_rate_hz: int) -> int:
    return (milliseconds * sample_rate_hz + 500) // 1_000


def extract_pcm_wav_segment(
    source: str | Path,
    destination: str | Path,
    *,
    start_ms: int,
    end_ms: int,
    cancelled: Callable[[], bool] | None = None,
) -> PcmSegmentFacts:
    """Copy a half-open source-time interval without decoding or changing PCM.

    Millisecond endpoints are mapped to the nearest source frame, with half
    frames rounded upward. The destination is created exclusively, copied in
    bounded chunks, finalized and flushed before this function returns. Any
    cancellation, malformed/truncated input, or I/O failure removes the newly
    created destination; an existing destination is never overwritten.
    """
    source_path = Path(source)
    destination_path = Path(destination)
    if start_ms < 0 or end_ms <= start_ms:
        raise ValueError("PCM WAV segment source-time interval is invalid.")
    if destination_path.exists():
        raise FileExistsError(f"Refusing to overwrite PCM WAV destination: {destination_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    created_destination = False
    try:
        try:
            input_context = wave.open(str(source_path), "rb")
        except wave.Error as exc:
            raise ValueError("Input is not a supported uncompressed PCM WAV.") from exc
        with input_context as input_wav:
            if input_wav.getcomptype() != "NONE":
                raise ValueError("Input is not a supported uncompressed PCM WAV.")
            channels = input_wav.getnchannels()
            sample_width = input_wav.getsampwidth()
            sample_rate = input_wav.getframerate()
            source_frames = input_wav.getnframes()
            start_frame = _nearest_frame(start_ms, sample_rate)
            end_frame = _nearest_frame(end_ms, sample_rate)
            if start_frame < 0 or end_frame <= start_frame or end_frame > source_frames:
                raise ValueError("PCM WAV segment frame interval is outside the source.")

            input_wav.setpos(start_frame)
            frames_remaining = end_frame - start_frame
            bytes_per_frame = channels * sample_width
            chunk_frames = max(1, _PCM_COPY_CHUNK_BYTES // bytes_per_frame)
            with destination_path.open("xb") as output_stream:
                created_destination = True
                with wave.open(output_stream, "wb") as output_wav:
                    output_wav.setnchannels(channels)
                    output_wav.setsampwidth(sample_width)
                    output_wav.setframerate(sample_rate)
                    while frames_remaining:
                        if cancelled is not None and cancelled():
                            raise PcmExtractionCancelled("PCM WAV extraction cancelled.")
                        count = min(chunk_frames, frames_remaining)
                        payload = input_wav.readframes(count)
                        actual_frames = len(payload) // bytes_per_frame
                        if actual_frames != count:
                            raise ValueError("PCM WAV source ended before the requested interval.")
                        output_wav.writeframesraw(payload)
                        frames_remaining -= actual_frames
                output_stream.flush()
                os.fsync(output_stream.fileno())
    except Exception:
        if created_destination:
            destination_path.unlink(missing_ok=True)
        raise
    return PcmSegmentFacts(
        start_frame=start_frame,
        end_frame=end_frame,
        sample_rate_hz=sample_rate,
        channels=channels,
        sample_width_bytes=sample_width,
        frame_count=end_frame - start_frame,
    )


def load_audio_file(
    file_name: str,
    sample_rate: int | None = None,
    mono: bool = True,
) -> torch.Tensor:
    """Load an audio file and return a float tensor of shape ``(channels, samples)``.

    Parameters
    ----------
    file_name:
        Path to the audio file (any format supported by ``soundfile``).
    sample_rate:
        Target sample rate. If provided and different from the file's native
        rate, the audio is resampled using a Kaiser-best filter.
    mono:
        If ``True``, multi-channel audio is averaged to mono.
    """
    audio_data, native_sample_rate = sf.read(file_name, always_2d=True, dtype="float32")
    if mono and audio_data.ndim == 2 and audio_data.shape[1] > 1:
        audio_data = np.mean(audio_data, axis=1, keepdims=True)
    if sample_rate is not None and sample_rate != native_sample_rate:
        audio_data = resampy.resample(
            audio_data, native_sample_rate, sample_rate, axis=0, filter="kaiser_best"
        )
    return torch.from_numpy(audio_data).float().t()
