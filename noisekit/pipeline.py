from __future__ import annotations

import json
import re
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from rich.console import Console
from rich.progress import track

from .dataset import extract_audio_and_text, extract_language, load_samples
from .noise_cache import ensure_default_noise_dir
from .scoring import _NISQA_KEYS, audio_stats, compute_nisqa, compute_pesq, compute_snr_db
from .transforms import list_builtin_presets, load_preset, preset_requires_noise_dir

console = Console()


def _resample_to_16k(array: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return array
    return librosa.resample(array, orig_sr=sr, target_sr=16000)


def run_generate(
    dataset: str,
    samples: int,
    presets: list[str],
    output: Path,
    seed: int,
    split: str,
    config: str | None,
    preset_file: Path | None,
    noise_dir: Path | None = None,
    nisqa: bool = True,
    transcript_column: str | None = None,
) -> None:
    output_dir = Path(output)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    if not presets:
        presets = [p["name"] for p in list_builtin_presets()]
        console.print(f"[dim]No presets specified — using all built-in: {', '.join(presets)}[/dim]")

    if noise_dir is None and any(preset_requires_noise_dir(p, preset_file) for p in presets):
        noise_dir = ensure_default_noise_dir()

    console.print(f"Loading [bold]{samples}[/bold] samples from [cyan]{dataset}[/cyan] (split={split}) …")
    raw_samples = load_samples(dataset, samples, seed, split, config)
    console.print(f"Loaded {len(raw_samples)} samples.")

    manifest: list[dict] = []
    _seen_names: set[str] = set()

    for i, sample in enumerate(track(raw_samples, description="Generating …")):
        ref_array, ref_sr, transcript = extract_audio_and_text(sample, transcript_column)
        language = extract_language(sample, config)
        ref_16k = _resample_to_16k(ref_array, ref_sr)
        peak = np.abs(ref_16k).max()
        if peak > 1e-9:
            ref_16k = ref_16k / peak

        raw_path = sample.get("audio", {}).get("path") or ""
        raw_stem = Path(raw_path).stem if raw_path else f"sample_{i:04d}"
        original_stem = re.sub(r"[^a-z0-9_]", "_", raw_stem.lower())
        original_stem = re.sub(r"_+", "_", original_stem).strip("_") or f"sample_{i:04d}"
        source_filename = Path(raw_path).name if raw_path else None

        for preset_name in presets:
            preset = load_preset(preset_name, preset_file, noise_dir=noise_dir)
            deg = preset.full(ref_16k.copy(), sample_rate=16000)
            deg = deg.astype(np.float32)

            min_len = min(len(ref_16k), len(deg))

            base = f"{original_stem}_{preset_name}"
            filename = f"{base}.wav"
            if filename in _seen_names:
                n = 1
                while f"{base}_{n}.wav" in _seen_names:
                    n += 1
                filename = f"{base}_{n}.wav"
            _seen_names.add(filename)

            sf.write(audio_dir / filename, deg, 16000, subtype="PCM_16")

            # For presets with a restoration Resample at the end, compute PESQ
            # at the intermediate sample rate (before the upsample). This avoids
            # the double-resampling collapse (8k→16k→8k gives PESQ ~1.1 regardless
            # of noise level). The scoring transform stops before the final upsample.
            if preset.scoring is not None and preset.scoring_sr is not None:
                deg_scoring = preset.scoring(ref_16k.copy(), sample_rate=16000).astype(np.float32)
                ref_scoring = librosa.resample(ref_16k, orig_sr=16000, target_sr=preset.scoring_sr)
                min_s = min(len(ref_scoring), len(deg_scoring))
                pesq_score = compute_pesq(ref_scoring[:min_s], deg_scoring[:min_s], preset.scoring_sr)
                snr = compute_snr_db(ref_scoring[:min_s], deg_scoring[:min_s])
            else:
                pesq_score = compute_pesq(ref_16k[:min_len], deg[:min_len], 16000)
                snr = compute_snr_db(ref_16k[:min_len], deg[:min_len])

            nisqa_scores = compute_nisqa(deg, 16000) if nisqa else dict.fromkeys(_NISQA_KEYS)
            entry: dict = {
                "file_name": f"audio/{filename}",
                "source": source_filename,
                "dataset": dataset,
                "language": language,
                "preset": preset_name,
                "transcript": transcript,
                "snr_db": round(snr, 3),
                "pesq_mos": pesq_score,
                **nisqa_scores,
            }
            manifest.append(entry)

    metadata_path = output_dir / "metadata.jsonl"
    with open(metadata_path, "w", encoding="utf-8") as f:
        for entry in manifest:
            f.write(json.dumps(entry) + "\n")

    console.print(f"\n[green]Done.[/green] {len(manifest)} files written to [bold]{output_dir}[/bold]")
    console.print(f"Metadata: [bold]{metadata_path}[/bold]")
    console.print('[dim]Load with: datasets.load_dataset("audiofolder", data_dir="{output_dir}")[/dim]')


def run_score(
    input_dir: Path,
    reference_dir: Path | None,
    output: Path,
    nisqa: bool = True,
) -> None:
    wav_files = sorted(input_dir.glob("*.wav"))
    if not wav_files:
        console.print(f"[red]No WAV files found in {input_dir}[/red]")
        return

    results = []
    for wav_path in track(wav_files, description="Scoring …"):
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        entry: dict = {"file": wav_path.name, **audio_stats(audio, sr)}

        if nisqa:
            entry.update(compute_nisqa(audio, sr))

        if reference_dir is not None:
            ref_path = reference_dir / wav_path.name
            if ref_path.exists():
                ref, ref_sr = sf.read(str(ref_path), dtype="float32")
                if ref.ndim > 1:
                    ref = ref.mean(axis=1)
                if ref_sr != sr:
                    ref = librosa.resample(ref, orig_sr=ref_sr, target_sr=sr)
                entry["snr_db"] = round(compute_snr_db(ref, audio), 3)
                entry["pesq_mos"] = compute_pesq(ref, audio, sr)
            else:
                entry["snr_db"] = None
                entry["pesq_mos"] = None

        results.append(entry)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)

    console.print(f"\n[green]Scored {len(results)} files.[/green] Results: [bold]{output}[/bold]")

    if reference_dir is None:
        console.print("[dim]Tip: pass --reference-dir to compute PESQ scores.[/dim]")
