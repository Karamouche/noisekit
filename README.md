<div align="center">
  <img src="assets/banner.svg" alt="noisekit" width="800"/>

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![built with audiomentations](https://img.shields.io/badge/built%20with-audiomentations-orange)](https://github.com/iver56/audiomentations)

</div>

<br/>

Generate degraded speech datasets for noise-robust ASR benchmarking.

Takes a clean HuggingFace speech dataset, applies real-world degradation presets via [audiomentations](https://github.com/iver56/audiomentations), and scores each output with PESQ, SNR, and NISQA, producing a JSONL manifest ready for noise-robustness benchmarking.

Six atomic degradation scenarios are built in: telephony (G.711 + low-bitrate codec), wideband codec compression, ambient noise, clipping distortion, and far-field reverb. Atomic presets compose into compound multi-condition scenarios.

> [!NOTE]
> Degradations are programmatically simulated. Scores may not generalize to genuine production recordings; validate final benchmarks on annotated real-world data.

## How it works

```mermaid
flowchart LR
    A[("HuggingFace\nDataset")] --> B["noisekit generate"]
    B --> C["7 atomic presets\ncodec · noise · reverb\ndropout · clipping"]
    B --> D["3 compound presets\nmulti-condition chains"]
    C & D --> E[("WAVs + metadata.jsonl\nPESQ · SNR · NISQA")]
```

## Install

No installation needed. Run directly with `uvx`:

```bash
uvx noisekit --help
```

Or install for development:

```bash
git clone https://github.com/Karamouche/noisekit.git
cd noisekit
uv sync
uv run noisekit --help
```

## Usage

### Generate a degraded dataset

```bash
uvx noisekit generate \
  --dataset google/fleurs \
  --config en_us \
  --split test \
  --samples 300 \
  --preset telecom \
  --preset low_bitrate \
  --output ./benchmark_dataset \
  --seed 42
```

`--preset` is repeatable: pass it once per preset.

For `noise`, you can supply your own background-noise WAVs with `--noise-dir` (e.g. [MUSAN](https://www.openslr.org/17/), [DEMAND](https://zenodo.org/record/1227121), or [FSD50K](https://zenodo.org/record/4060432)):

```bash
uvx noisekit generate \
  --dataset google/fleurs --config en_us --split test \
  --samples 300 --preset noise \
  --noise-dir ~/datasets/musan/noise \
  --output ./benchmark_dataset --seed 42
```

Output:

```
benchmark_dataset/
├── metadata.jsonl          # one entry per generated file (AudioFolder format)
└── audio/
    ├── sample_0000_telecom.wav
    ├── sample_0001_low_bitrate.wav
    └── ...
```

The output is directly loadable as a HuggingFace dataset:

```python
from datasets import load_dataset
ds = load_dataset("audiofolder", data_dir="./benchmark_dataset")
```

Each `metadata.jsonl` entry:

```json
{
  "file_name": "audio/sample_0042_telecom.wav",
  "source": "common_voice_en_23136613.mp3",
  "dataset": "google/fleurs",
  "language": "en-US",
  "preset": "telecom",
  "transcript": "the cat sat on the mat",
  "snr_db": 5.2,
  "pesq_mos": 2.78,
  "nisqa_mos": 2.14,
  "nisqa_noisiness": 1.93,
  "nisqa_discontinuity": 2.41,
  "nisqa_coloration": 1.87,
  "nisqa_loudness": 2.3
}
```

### Score an existing audio folder

```bash
# File stats only (duration, RMS, peak)
uvx noisekit score ./audio_folder --output scores.json

# With PESQ + SNR (requires matching reference files)
uvx noisekit score ./audio_folder --reference-dir ./clean_audio --output scores.json

# Skip NISQA (faster, no model download)
uvx noisekit score ./audio_folder --no-nisqa --output scores.json
```

### List available presets

```bash
uvx noisekit list-presets
uvx noisekit list-presets --verbose   # show full transform stack
```

## Presets

Nine built-in presets: six atomic scenarios, three compound multi-condition presets, and a clean reference control. None use synthetic white noise; codec artifacts, real ambient recordings, and room simulation produce the degradation instead.

### Atomic presets

| Preset            | Description                                                                               | PESQ       |
| ----------------- | ----------------------------------------------------------------------------------------- | ---------- |
| `clean_reference` | Minimal processing (PESQ ceiling / control)                                               | 4.0-4.5    |
| `telecom`         | G.711-style call: 8 kHz bandpass + mu-law companding (ITU-T G.711) + 16-32 kbps MP3 codec | NB 3.5-4.5 |
| `low_bitrate`     | Wideband audio crushed by 16-32 kbps MP3 compression                                      | WB 1.5-2.5 |
| `noise`           | Real ambient noise from `--noise-dir` mixed in at SNR 5-15 dB                             | WB 1.0-2.5 |
| `clipping`        | Microphone overload: clips the loudest 10-25% of samples                                  | WB 2.0-3.5 |
| `reverb`          | Far-field room reverb at 1-3 m mic distance                                               | WB 2.0-3.5 |

`telecom` is scored with PESQ narrowband at 8 kHz (before the final upsample); all other presets are scored wideband at 16 kHz.

All dependencies, including `pyroomacoustics` (used by `reverb`), are bundled with no extra install needed.

`noise` accepts a `--noise-dir` pointing at a directory of background-noise WAVs (e.g. MUSAN, DEMAND, FSD50K). If omitted, noisekit auto-downloads a small MUSAN noise-only subset (~20 files, ~120 MB) to `~/.cache/noisekit/noise/musan_ambient/` on first use.

### Compound presets

Compound presets chain two atomic presets together. Noise is applied first (acoustic environment), then codec or dropout (digital processing on the already-degraded signal).

| Preset             | Chain                  | Noise source                   | PESQ       |
| ------------------ | ---------------------- | ------------------------------ | ---------- |
| `noise_telecom`    | `noise` → `telecom`    | `--noise-dir` or auto-download | NB 1.5-2.5 |
| `clipping_telecom` | `clipping` → `telecom` | (none)                         | NB 1.0-2.5 |
| `noise_reverb`     | `noise` → `reverb`     | `--noise-dir` or auto-download | WB 1.0-2.5 |

You can also define your own compound preset with a `chain:` key in a YAML file:

```yaml
name: my_compound
description: "Noisy environment then telephony codec"
chain:
  - noise
  - telecom
```

### Custom presets

Pass your own YAML file with `--preset-file`:

```bash
uvx noisekit generate \
  --dataset google/fleurs \
  --samples 100 \
  --preset-file ./my_preset.yaml \
  --output ./output
```

Preset format:

```yaml
name: my_preset
description: "Custom telephony simulation"
transforms:
  - type: Resample
    parameters:
      min_sample_rate: 8000
      max_sample_rate: 8000
    p: 1.0
  - type: Mp3Compression
    parameters:
      min_bitrate: 16
      max_bitrate: 32
      backend: lameenc
    p: 1.0
  - type: Resample
    parameters:
      min_sample_rate: 16000
      max_sample_rate: 16000
    p: 1.0
```

Any transform from [audiomentations](https://github.com/iver56/audiomentations) is supported. Use `${NOISE_DIR}` as a placeholder for `--noise-dir` inside your preset YAML. Use `chain:` instead of `transforms:` to compose built-in atomic presets sequentially.

## Requirements

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) for `uvx` usage
- No system dependencies: MP3 encoding uses pure-Python `lameenc`, no ffmpeg needed
