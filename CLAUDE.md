# noisekit — Claude Instructions

**Always update this file when making notable changes** (new commands, new presets, architectural decisions, scoring changes, dependency additions).

**Always update README.md** when changing presets, CLI flags, output format, or any user-facing behavior.

## Project

`noisekit` is a `uvx`-compatible Python CLI that generates degraded speech datasets from clean HuggingFace corpora. It simulates six atomic audio degradation scenarios — telecom (G.711 calls), low-bitrate codec compression, noisy environments (real ambient noise), far-field reverb, and clipping distortion — plus compound multi-condition scenarios built by chaining atomic presets. Designed for ASR noise-robustness benchmarking. A `clean_reference` control completes the catalog.

## Package Management

Use **UV** for everything: `uv add`, `uv run`, `uv sync`. Never use pip directly.

Key runtime dependencies: `audiomentations>=0.38`, `lameenc>=1.4` (pure-Python MP3 encoder used by `Mp3Compression` in `telecom` and `low_bitrate`; no system ffmpeg needed), `torchmetrics>=1.7.0` (NISQA scoring — downloads ~50 MB model weights to `~/.torchmetrics/NISQA/` on first use), `pyroomacoustics` (room acoustics simulation for `reverb` — now a core dependency, no extra install needed).

## Architecture

```
noisekit/
├── cli.py          # Typer app — 3 commands: generate, score, list-presets
├── pipeline.py     # generate + score logic
├── dataset.py      # HuggingFace dataset loading (soundfile decoder, no torchcodec)
├── transforms.py   # Preset loading; returns PresetTransforms(full, scoring, scoring_sr)
├── scoring.py      # PESQ + SNR + NISQA; PESQ NB at 8 kHz for telephony presets
├── noise_cache.py  # Auto-downloads MUSAN music+noise for noise
└── presets/        # YAML preset files bundled with the package

```

## CLI

```bash
noisekit generate --dataset <hf-name> --samples N --presets P1 P2 --output ./out --seed 42
noisekit generate ... --presets noise --noise-dir /path/to/noise_wavs
noisekit generate ... --no-nisqa          # skip NISQA (no model download, faster)
noisekit score ./audio_dir [--reference-dir ./ref] [--output scores.json]
noisekit score ./audio_dir --no-nisqa     # skip NISQA for standalone scoring
noisekit list-presets [--verbose]
```

Custom presets: `--preset-file ./my_preset.yaml`

The `noise` preset uses a directory of background-noise WAVs. If `--noise-dir` is omitted, noisekit auto-downloads a small MUSAN **noise-only** subset (~20 files, ~120 MB) from `Aynursusuz/musan-audio-dataset` on HuggingFace to `~/.cache/noisekit/noise/musan_ambient/` on first use. Both `speech` and `music` classes are excluded: speech pollutes ASR/PESQ scoring; music sounds artificial as a background and is indistinguishable from white noise at low levels. Only label 2 (`noise` — wind, rain, traffic, machinery) is downloaded.

Pass `--noise-dir /path/to/wavs` to use your own corpus (e.g. MUSAN, DEMAND, FSD50K) instead. Inside a preset YAML, use the literal string `${NOISE_DIR}` as a parameter value and `transforms.load_preset` substitutes the resolved path at load time. Auto-download is wired in `pipeline.run_generate` via `noise_cache.ensure_default_noise_dir()`, gated by `transforms.preset_requires_noise_dir()`.

### MUSAN download — shard strategy

`Aynursusuz/musan-audio-dataset` is **sorted by label**: speech fills parquet shards 0–21, music+noise occupy shards 22–44 (music-first within that range, then noise). `noise_cache.py` bypasses speech entirely by loading only shards 22–44 via `hf://` URLs, then filters to `label == 2` (noise only). The shard list is shuffled before streaming so noise-heavy shards are hit early; a `buffer_size=200` shuffle adds within-shard diversity. Constants `_N_SHARDS = 45` and `_FIRST_AMBIENT_SHARD = 22` must be updated if the dataset is re-sharded. If the download yields zero noise samples, bisect by testing individual shards to find where the noise class begins.

## Preset YAML Format

```yaml
name: my_preset
description: "..."
transforms:
  - type: <audiomentations class>
    parameters:
      key: value
    p: 1.0
```

Built-in presets:

### Atomic Presets

| Preset            | Scenario                                                                      | Bandwidth           | PESQ mode | Target MOS |
| ----------------- | ----------------------------------------------------------------------------- | ------------------- | --------- | ---------- |
| `clean_reference` | Minimal gain normalization (PESQ ceiling)                                     | full                | WB 16 kHz | 4.0-4.5    |
| `telecom`         | G.711 call: mu-law companding (ITU-T G.711) + low-bitrate MP3 codec artifacts | 300-3400 Hz @ 8 kHz | NB 8 kHz  | 3.5-4.5    |
| `low_bitrate`     | Wideband low-bitrate MP3 compression (16-32 kbps)                             | 80-7500 Hz @ 16 kHz | WB 16 kHz | 1.5-2.5    |
| `noise`           | Real ambient noise via `AddBackgroundNoise`                                   | up to 8-12 kHz      | WB 16 kHz | 2.0-3.5    |
| `clipping`        | Microphone overload / ADC saturation (`ClippingDistortion` 10-25%)            | full                | WB 16 kHz | 2.0-3.5    |
| `reverb`          | Far-field reverberant room via `RoomSimulator`                                | full                | WB 16 kHz | 2.0-3.5    |

`telecom` and any compound preset ending with `telecom` use the 8 kHz PESQ NB scoring split (see below). All other presets score in PESQ WB at 16 kHz.

### Compound Presets

Compound presets chain two or more atomic presets together. Noise is added first (acoustic environment), then codec/dropout (digital processing of the already-degraded signal).

| Preset             | Chain                  | Requires      | PESQ mode | Target MOS |
| ------------------ | ---------------------- | ------------- | --------- | ---------- |
| `noise_telecom`    | `noise` → `telecom`    | `--noise-dir` | NB 8 kHz  | 1.5-2.5    |
| `noise_reverb`     | `noise` → `reverb`     | `--noise-dir` | WB 16 kHz | 1.0-2.5    |
| `clipping_telecom` | `clipping` → `telecom` | —             | NB 8 kHz  | 1.0-2.5    |

### Compound Preset YAML Format

A preset can use `chain:` instead of `transforms:` to apply multiple atomic presets sequentially:

```yaml
name: my_compound
description: "..."
chain:
  - atomic_preset_a
  - atomic_preset_b
```

Rules:

- `chain` and `transforms` are mutually exclusive.
- Chained entries must be names of built-in atomic presets (no nesting chains).
- `${NOISE_DIR}` resolution and the PESQ NB scoring split are detected automatically across the full concatenated chain.
- `reverb` uses `pyroomacoustics` (bundled as a core dependency — no extra install needed).

### Why no white noise

The catalog deliberately avoids `AddGaussianSNR` — white Gaussian noise sounds artificial and doesn't reflect real production audio. Instead:

- `telecom` and `low_bitrate` rely on `Mp3Compression` at 16-32 kbps for realistic codec smearing/pre-echo.
- `noise` uses `AddBackgroundNoise` over a user-supplied WAV corpus (MUSAN/DEMAND/FSD50K), so the noise floor matches the real environment you care about.

## PESQ Scoring — Important Design Decision

For `telecom`, PESQ is computed at **8 kHz narrowband** on the audio **before** the final `Resample(16000)` restoration step. Output WAV files are still saved at 16 kHz.

**Why:** Computing PESQ NB by downsampling the 16 kHz output (8k→16k→8k round-trip) collapses all telephony scores to ~1.1 regardless of noise level. Scoring at the 8 kHz intermediate stage gives proper stratification.

**MuLawCompanding + Normalize:** `telecom` inserts `Normalize(p=1.0)` immediately before `MuLawCompanding`. HuggingFace speech datasets (e.g., FLEURS) often have very low peak amplitude (~0.001-0.02). G.711 mu-law uses all 256 quantization levels only when the input is near full-scale; without normalization, low-amplitude signals collapse into the few quantization steps near zero. Normalizing to ±1 before companding ensures consistent quantization behavior.

`MuLawCompanding` is implemented via a `soundfile` WAV round-trip with `subtype='ULAW'`, which uses libsndfile's ITU-T G.711 segmented algorithm — byte-identical to real G.711 PCMU hardware and VoIP stacks (Asterisk, FreeSWITCH). This replaced an earlier smooth-log NumPy approximation that did not match the true piecewise-linear segmented encoding of the spec.

## Input Normalization — Global Pipeline Decision

`pipeline.py` peak-normalizes every input sample to amplitude 1.0 immediately after resampling to 16 kHz, before any preset transforms run:

```python
peak = np.abs(ref_16k).max()
if peak > 1e-9:
    ref_16k = ref_16k / peak
```

**Why:** HuggingFace datasets often have peaks as low as 0.001–0.02. Without normalization, `AddBackgroundNoise` (relative SNR mode) scales noise proportional to that tiny signal RMS — both speech and noise end up inaudible, and 16-bit PCM quantization noise dominates. Peak normalization guarantees all presets receive a full-scale signal.

**Safety:** The same normalized `ref_16k` is used as both the transform input and the PESQ/SNR reference, so all quality metrics remain valid relative comparisons. The mid-chain `Normalize` inside `telecom.yaml` (before `MuLawCompanding`) is still needed separately — the bandpass filter removes energy and that step re-normalizes before companding.

**`noise` also pre-normalizes:** `noise.yaml` adds a `Normalize` as its first transform. This handles the `noise_reverb` compound case: `RoomSimulator` can attenuate the signal by ~10× at large mic distances; without the mid-chain normalize, `AddBackgroundNoise` would see the attenuated level and mix noise too quietly. All compound presets using `noise` inherit this fix automatically.

`transforms.py` auto-detects this split: if the last transform is `Resample(16000)`, it creates a `scoring` Compose (all-but-last) alongside the `full` Compose.

## Dataset Loading

Uses `datasets` with `Audio(decode=False)` + manual `soundfile` decoding — avoids the `torchcodec` requirement introduced in `datasets` 4.x.

## Output Format

`metadata.jsonl` — one JSON object per generated file, following the HuggingFace [AudioFolder](https://huggingface.co/docs/datasets/audio_dataset#audiofolder) convention so the output directory is directly loadable with `datasets.load_dataset("audiofolder", data_dir="./out")`.

```json
{
  "file_name": "audio/common_voice_en_23136613_telecom.wav",
  "source": "common_voice_en_23136613.mp3",
  "dataset": "google/fleurs",
  "language": "en-US",
  "preset": "telecom",
  "transcript": "...",
  "snr_db": 1.8,
  "pesq_mos": 2.86,
  "nisqa_mos": 2.14,
  "nisqa_noisiness": 1.93,
  "nisqa_discontinuity": 2.41,
  "nisqa_coloration": 1.87,
  "nisqa_loudness": 2.3
}
```

File naming: `{original_stem}_{preset_name}.wav` where `original_stem` is derived from `sample["audio"]["path"]` (sanitized to `[a-z0-9_]`). Falls back to `sample_{i:04d}` if no path is available. Collisions resolved with `_1`, `_2`, … suffixes.

NISQA is non-intrusive (no reference needed). It scores the final 16 kHz output using `torchmetrics.functional.audio.nisqa.non_intrusive_speech_quality_assessment`. Model weights (~50 MB) are cached in `~/.torchmetrics/NISQA/` and loaded once per session via `@lru_cache`. Pass `--no-nisqa` to skip. All five dimensions are `null` when skipped.

## Verification

```bash
uv run noisekit list-presets --verbose

# Codec-only presets (no noise dir needed)
uv run noisekit generate \
  --dataset google/fleurs \
  --config en_us --split test \
  --samples 3 --presets clean_reference telecom low_bitrate \
  --output ./test_out --seed 42
cat test_out/metadata.jsonl

# New atomic presets — no external dependencies
uv run noisekit generate \
  --dataset google/fleurs --config en_us --split test \
  --samples 3 --presets clipping \
  --no-nisqa --output ./test_atomic --seed 42

# noise — auto-downloads MUSAN noise-only clips on first run
uv run noisekit generate \
  --dataset google/fleurs --config en_us --split test \
  --samples 3 --presets noise \
  --output ./test_noise --seed 42

# Compound presets (auto-downloads MUSAN noise on first run)
uv run noisekit generate \
  --dataset google/fleurs --config en_us --split test \
  --samples 3 --presets noise_telecom \
  --no-nisqa --output ./test_compound --seed 42

# clipping_telecom — no noise dir needed
uv run noisekit generate \
  --dataset google/fleurs --config en_us --split test \
  --samples 3 --presets clipping_telecom \
  --no-nisqa --output ./test_clipping_telecom --seed 42

# Far-field reverb
uv run noisekit generate \
  --dataset google/fleurs --config en_us --split test \
  --samples 3 --presets reverb noise_reverb \
  --no-nisqa --output ./test_reverb --seed 42

# noise with your own noise corpus (skips auto-download)
uv run noisekit generate \
  --dataset google/fleurs --config en_us --split test \
  --samples 3 --presets noise \
  --noise-dir ~/datasets/musan/noise \
  --output ./test_noise --seed 42
```

Expected PESQ spread: clean ~4.6, telecom ~2.5-3.5 (NB), low_bitrate ~1.5-2.5 (WB), noise ~1.0-2.5 (WB), clipping ~2.0-3.5 (WB), reverb ~2.0-3.5 (WB).

Compound preset PESQ: noise_telecom ~1.5-2.5 (NB), clipping_telecom ~1.0-2.5 (NB), noise_reverb ~1.0-2.5 (WB).

Expected NISQA spread: clean ~4.0-4.5, degraded presets ~1.5-3.0. NISQA model weights (~50 MB) are downloaded on first run.
