"""Audio I/O, spectral transforms, and dataset classes."""

from bioacoustics_dl_toolbox.audio.datasets import (
    AudioDataset,
    CsvSplit,
    DatabaseCsvSplit,
    SpectrogramDataset,
    StridedAudioDataset,
    get_audio_files_from_dir,
    get_broken_audio_files,
)
from bioacoustics_dl_toolbox.audio.io import (
    PcmExtractionCancelled,
    PcmSegmentFacts,
    extract_pcm_wav_segment,
    load_audio_file,
)
from bioacoustics_dl_toolbox.audio.preprocessing import (
    FeaturePreprocessor,
    build_feature_preprocessor,
    spectrogram_frames_for_samples,
)

__all__ = [
    "load_audio_file",
    "PcmExtractionCancelled",
    "PcmSegmentFacts",
    "extract_pcm_wav_segment",
    "FeaturePreprocessor",
    "build_feature_preprocessor",
    "spectrogram_frames_for_samples",
    "AudioDataset",
    "SpectrogramDataset",
    "StridedAudioDataset",
    "DatabaseCsvSplit",
    "CsvSplit",
    "get_audio_files_from_dir",
    "get_broken_audio_files",
]
