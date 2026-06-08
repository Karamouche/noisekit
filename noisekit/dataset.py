from __future__ import annotations

import io
import itertools

import numpy as np
import soundfile as sf


def load_samples(
    dataset_name: str,
    n: int,
    seed: int,
    split: str = "train",
    config: str | None = None,
) -> list[dict]:
    from datasets import Audio, load_dataset

    kwargs: dict = dict(split=split, streaming=True)
    ds = load_dataset(dataset_name, config, **kwargs) if config is not None else load_dataset(dataset_name, **kwargs)

    # Disable auto-decoding so we can decode with soundfile ourselves,
    # avoiding the torchcodec requirement introduced in datasets 4.x.
    ds = ds.cast_column("audio", Audio(decode=False))
    ds = ds.shuffle(seed=seed, buffer_size=min(1000, n * 10))
    return list(itertools.islice(ds, n))


def _config_to_bcp47(config: str) -> str:
    """Convert a HuggingFace config name to a BCP-47 tag.

    ``en_us`` → ``en-US``, ``fr_fr`` → ``fr-FR``, ``en`` → ``en``.
    """
    parts = config.split("_", 1)
    if len(parts) == 2:
        return f"{parts[0].lower()}-{parts[1].upper()}"
    return config.lower()


def extract_language(sample: dict, config: str | None = None) -> str | None:
    """Return a BCP-47 language tag for the sample.

    Priority:
    1. ``locale`` — already BCP-47 (Common Voice, Mozilla datasets).
    2. ``config`` — HuggingFace subset name normalized to BCP-47
       (e.g. ``en_us`` → ``en-US``). The per-sample ``language`` column is
       intentionally skipped because datasets like FLEURS store full names
       (``"English"``) which are not valid BCP-47 tags.
    """
    if locale := sample.get("locale"):
        return locale
    if config:
        return _config_to_bcp47(config)
    return None


_FALLBACK_TRANSCRIPT_COLS = ("text", "sentence", "transcription", "normalized_text")


def extract_audio_and_text(
    sample: dict,
    transcript_column: str | None = None,
) -> tuple[np.ndarray, int, str]:
    audio_field = sample["audio"]

    raw_bytes = audio_field.get("bytes")
    path = audio_field.get("path")

    if raw_bytes:
        with sf.SoundFile(io.BytesIO(raw_bytes)) as f:
            array = f.read(dtype="float32", always_2d=False)
            sr = f.samplerate
    elif path:
        array, sr = sf.read(path, dtype="float32", always_2d=False)
    else:
        raise ValueError("Audio sample has neither 'bytes' nor 'path'.")

    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        array = array.mean(axis=1)  # (samples, channels) → (samples,) for mono-only metrics

    if transcript_column is not None:
        if transcript_column not in sample:
            raise ValueError(
                f"Transcript column '{transcript_column}' not found in dataset. "
                f"Available columns: {list(sample.keys())}"
            )
        text = sample.get(transcript_column) or ""
    else:
        text = next((sample[c] for c in _FALLBACK_TRANSCRIPT_COLS if sample.get(c)), None)
        if text is None:
            raise ValueError(
                f"No transcript column found. Tried: {list(_FALLBACK_TRANSCRIPT_COLS)}. "
                f"Available columns: {list(sample.keys())}. "
                "Use --transcript-column to specify the correct column."
            )

    return array, int(sr), str(text).strip()
