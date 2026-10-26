"""Focused unit tests for the public, raw-PCM-preserving WAV extraction API."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from bioacoustics_dl_toolbox.audio import (
    PcmExtractionCancelled,
    extract_pcm_wav_segment,
)


def _pcm_payload(frame_count: int, channels: int, sample_width: int) -> bytes:
    byte_count = frame_count * channels * sample_width
    return bytes((index * 37 + 11) % 256 for index in range(byte_count))


def _write_wav(
    path: Path,
    *,
    frame_count: int,
    sample_rate: int,
    channels: int,
    sample_width: int,
    payload: bytes | None = None,
) -> bytes:
    payload = payload or _pcm_payload(frame_count, channels, sample_width)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(sample_width)
        stream.setframerate(sample_rate)
        stream.writeframes(payload)
    return payload


def _read_wav(path: Path) -> tuple[tuple[int, int, int, int], bytes]:
    with wave.open(str(path), "rb") as stream:
        facts = (
            stream.getframerate(),
            stream.getnchannels(),
            stream.getsampwidth(),
            stream.getnframes(),
        )
        return facts, stream.readframes(stream.getnframes())


@pytest.mark.parametrize(
    ("sample_rate", "channels", "sample_width", "start_ms", "end_ms"),
    [
        (44_100, 1, 1, 5, 6),
        (44_100, 2, 2, 1, 3),
        (48_000, 1, 3, 9, 10),
        (48_000, 2, 4, 0, 2),
    ],
)
def test_extract_pcm_wav_segment_preserves_exact_frames_and_wav_facts(
    tmp_path, sample_rate, channels, sample_width, start_ms, end_ms
):
    source = tmp_path / "source.wav"
    destination = tmp_path / "nested" / "segment.wav"
    source_frames = sample_rate // 100
    source_payload = _write_wav(
        source,
        frame_count=source_frames,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
    )
    source_bytes = source.read_bytes()

    facts = extract_pcm_wav_segment(source, destination, start_ms=start_ms, end_ms=end_ms)

    start_frame = (start_ms * sample_rate + 500) // 1_000
    end_frame = (end_ms * sample_rate + 500) // 1_000
    bytes_per_frame = channels * sample_width
    expected_payload = source_payload[start_frame * bytes_per_frame : end_frame * bytes_per_frame]
    expected = tmp_path / "expected.wav"
    _write_wav(
        expected,
        frame_count=end_frame - start_frame,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        payload=expected_payload,
    )

    assert facts.start_frame == start_frame
    assert facts.end_frame == end_frame
    assert facts.sample_rate_hz == sample_rate
    assert facts.channels == channels
    assert facts.sample_width_bytes == sample_width
    assert facts.frame_count == end_frame - start_frame
    assert _read_wav(destination) == (
        (sample_rate, channels, sample_width, end_frame - start_frame),
        expected_payload,
    )
    assert destination.read_bytes() == expected.read_bytes()
    assert source.read_bytes() == source_bytes


@pytest.mark.parametrize(
    ("start_ms", "end_ms", "message"),
    [
        (-1, 1, "interval"),
        (1, 1, "interval"),
        (2, 1, "interval"),
        (0, 11, "outside"),
    ],
)
def test_extract_pcm_wav_segment_rejects_invalid_bounds_without_output(
    tmp_path, start_ms, end_ms, message
):
    source = tmp_path / "source.wav"
    destination = tmp_path / "segment.wav"
    _write_wav(
        source,
        frame_count=480,
        sample_rate=48_000,
        channels=1,
        sample_width=2,
    )

    with pytest.raises(ValueError, match=message):
        extract_pcm_wav_segment(source, destination, start_ms=start_ms, end_ms=end_ms)

    assert not destination.exists()


@pytest.mark.parametrize("kind", ["corrupt", "non-pcm", "truncated"])
def test_extract_pcm_wav_segment_rejects_bad_input_and_removes_output(tmp_path, kind):
    source = tmp_path / "source.wav"
    destination = tmp_path / "segment.wav"
    if kind == "corrupt":
        source.write_bytes(b"not a RIFF WAVE")
    else:
        _write_wav(
            source,
            frame_count=100,
            sample_rate=48_000,
            channels=1,
            sample_width=1,
        )
        content = bytearray(source.read_bytes())
        if kind == "non-pcm":
            content[20:22] = (3).to_bytes(2, "little")
        else:
            del content[-20:]
        source.write_bytes(content)

    with pytest.raises(ValueError):
        extract_pcm_wav_segment(source, destination, start_ms=0, end_ms=2)

    assert not destination.exists()


def test_extract_pcm_wav_segment_never_clobbers_existing_destination(tmp_path):
    source = tmp_path / "source.wav"
    destination = tmp_path / "segment.wav"
    _write_wav(
        source,
        frame_count=480,
        sample_rate=48_000,
        channels=1,
        sample_width=2,
    )
    destination.write_bytes(b"keep me")

    with pytest.raises(FileExistsError):
        extract_pcm_wav_segment(source, destination, start_ms=0, end_ms=5)

    assert destination.read_bytes() == b"keep me"


def test_extract_pcm_wav_segment_cancellation_removes_partial_output(tmp_path):
    source = tmp_path / "source.wav"
    destination = tmp_path / "segment.wav"
    _write_wav(
        source,
        frame_count=96_000,
        sample_rate=48_000,
        channels=2,
        sample_width=2,
    )
    source_bytes = source.read_bytes()
    checks = 0

    def cancelled() -> bool:
        nonlocal checks
        checks += 1
        return checks > 1

    with pytest.raises(PcmExtractionCancelled, match="cancelled"):
        extract_pcm_wav_segment(source, destination, start_ms=0, end_ms=2_000, cancelled=cancelled)

    assert checks == 2
    assert not destination.exists()
    assert source.read_bytes() == source_bytes


def test_extract_pcm_wav_segment_write_error_removes_partial_output(tmp_path, monkeypatch):
    source = tmp_path / "source.wav"
    destination = tmp_path / "segment.wav"
    _write_wav(
        source,
        frame_count=48_000,
        sample_rate=48_000,
        channels=1,
        sample_width=2,
    )
    original = wave.Wave_write.writeframesraw
    raised = False

    def fail_after_write(stream, payload):
        nonlocal raised
        result = original(stream, payload)
        if not raised and payload:
            raised = True
            raise OSError("injected write failure")
        return result

    monkeypatch.setattr(wave.Wave_write, "writeframesraw", fail_after_write)
    with pytest.raises(OSError, match="injected write failure"):
        extract_pcm_wav_segment(source, destination, start_ms=0, end_ms=1_000)

    assert not destination.exists()


def test_extract_pcm_wav_segment_uses_bounded_frame_reads(tmp_path, monkeypatch):
    source = tmp_path / "source.wav"
    destination = tmp_path / "segment.wav"
    _write_wav(
        source,
        frame_count=96_000,
        sample_rate=48_000,
        channels=2,
        sample_width=4,
    )
    original = wave.Wave_read.readframes
    requested: list[int] = []

    def traced_read(stream, count):
        requested.append(count)
        return original(stream, count)

    monkeypatch.setattr(wave.Wave_read, "readframes", traced_read)
    extract_pcm_wav_segment(source, destination, start_ms=0, end_ms=2_000)

    assert len(requested) > 1
    assert max(requested) * 2 * 4 <= 64 * 1_024
