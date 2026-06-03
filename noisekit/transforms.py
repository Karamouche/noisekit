from __future__ import annotations

import io
from pathlib import Path
from typing import NamedTuple

import audiomentations
import numpy as np
import soundfile as sf
import yaml
from audiomentations.core.transforms_interface import BaseWaveformTransform


class _SRTrackingCompose:
    """Drop-in for audiomentations.Compose that propagates the effective sample
    rate between transforms.

    audiomentations.Compose passes the *original* sample_rate to every transform
    in the loop, so a Resample(8000) followed by Resample(16000) becomes a no-op
    on the second step (it sees sample_rate==target and skips). We fix this by
    detecting length changes — a length change means a Resample ran, so we update
    the effective rate from the length ratio before calling the next transform.
    """

    def __init__(self, transforms: list[BaseWaveformTransform]) -> None:
        self._transforms = transforms

    def __call__(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        current_sr = sample_rate
        for t in self._transforms:
            prev_len = len(samples)
            samples = t(samples, current_sr)
            if len(samples) != prev_len and prev_len > 0:
                current_sr = round(current_sr * len(samples) / prev_len)
        return samples


class PresetTransforms(NamedTuple):
    full: _SRTrackingCompose
    # Same chain but without the final "restoration" Resample, for PESQ scoring
    scoring: _SRTrackingCompose | None
    # Sample rate at which `scoring` outputs audio (None if no split)
    scoring_sr: int | None


_NOISE_DIR_PLACEHOLDER = "${NOISE_DIR}"


def _resolve_params(t: dict, noise_dir: Path | None) -> dict:
    params = dict(t.get("parameters", {}))
    for k, v in params.items():
        if v == _NOISE_DIR_PLACEHOLDER:
            if noise_dir is None:
                raise ValueError(
                    f"Transform '{t['type']}' parameter '{k}' requires --noise-dir, "
                    f"but none was provided. Point it at a directory of background "
                    f"noise WAVs (e.g. MUSAN, DEMAND, or FSD50K)."
                )
            params[k] = str(noise_dir)
    return params


class MuLawCompanding(BaseWaveformTransform):
    """G.711 mu-law companding via libsndfile's ULAW codec (true ITU-T G.711 segmented algorithm).

    Uses a soundfile WAV round-trip (subtype='ULAW') — byte-identical to real G.711 PCMU hardware.
    Input should be full-scale (use Normalize beforehand).
    """

    supports_multichannel = True

    def apply(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        buf = io.BytesIO()
        # soundfile expects (samples, channels); audiomentations uses (channels, samples)
        sf.write(buf, samples.T if samples.ndim == 2 else samples, sample_rate, subtype="ULAW", format="WAV")
        buf.seek(0)
        decoded, _ = sf.read(buf, dtype="float32")
        return decoded.T if decoded.ndim == 2 else decoded


def preset_requires_noise_dir(name: str, preset_file: Path | None = None) -> bool:
    """Peek at a preset YAML and report whether it references ${NOISE_DIR}."""
    path = preset_file if preset_file is not None else Path(__file__).parent / "presets" / f"{name}.yaml"
    if not path.exists():
        return False
    cfg = yaml.safe_load(path.read_text())
    for t in cfg.get("transforms", []):
        for v in t.get("parameters", {}).values():
            if v == _NOISE_DIR_PLACEHOLDER:
                return True
    return any(preset_requires_noise_dir(chained_name, preset_file) for chained_name in cfg.get("chain", []))


def _collect_t_configs(cfg: dict, preset_file: Path | None, noise_dir: Path | None) -> list[dict]:
    """Resolve a preset config to a flat list of transform dicts.

    Atomic presets return cfg['transforms'] directly. Compound presets
    (chain: [name, ...]) load each named preset's transforms and concatenate
    them. Nesting chains inside chains is not supported.
    """
    if "chain" in cfg and "transforms" in cfg:
        raise ValueError(
            f"Preset '{cfg.get('name', '?')}' defines both 'chain' and 'transforms'. Use one or the other."
        )
    if "transforms" in cfg:
        return list(cfg["transforms"])
    if "chain" not in cfg:
        raise ValueError(f"Preset '{cfg.get('name', '?')}' has neither 'transforms' nor 'chain' key.")
    presets_dir = Path(__file__).parent / "presets"
    combined: list[dict] = []
    for chained_name in cfg["chain"]:
        if preset_file is not None:
            candidate = preset_file.parent / f"{chained_name}.yaml"
            chained_path = candidate if candidate.exists() else presets_dir / f"{chained_name}.yaml"
        else:
            chained_path = presets_dir / f"{chained_name}.yaml"
        if not chained_path.exists():
            raise FileNotFoundError(f"Chained preset '{chained_name}' not found at {chained_path}.")
        chained_cfg = yaml.safe_load(chained_path.read_text())
        if "chain" in chained_cfg:
            raise ValueError(f"Chained preset '{chained_name}' is itself a compound preset. Nesting not supported.")
        combined.extend(chained_cfg.get("transforms", []))
    return combined


_CUSTOM_TRANSFORMS: dict[str, type] = {
    "MuLawCompanding": MuLawCompanding,
}


def _make_transform(t: dict, noise_dir: Path | None = None) -> BaseWaveformTransform:
    cls_name = t["type"]
    if cls_name in _CUSTOM_TRANSFORMS:
        cls = _CUSTOM_TRANSFORMS[cls_name]
    elif hasattr(audiomentations, cls_name):
        cls = getattr(audiomentations, cls_name)
    else:
        raise ValueError(
            f"Unknown transform: '{cls_name}'. "
            f"Valid custom transforms: {sorted(_CUSTOM_TRANSFORMS)}. "
            f"Check the audiomentations docs for built-in transform names."
        )
    params = _resolve_params(t, noise_dir)
    p = float(t.get("p", 1.0))
    return cls(p=p, **params)


def load_preset(
    name: str,
    preset_file: Path | None = None,
    noise_dir: Path | None = None,
) -> PresetTransforms:
    path = preset_file if preset_file is not None else Path(__file__).parent / "presets" / f"{name}.yaml"

    if not path.exists():
        raise FileNotFoundError(f"Preset '{name}' not found. Run 'noisekit list-presets' to see available presets.")

    cfg = yaml.safe_load(path.read_text())
    t_configs = _collect_t_configs(cfg, preset_file, noise_dir)

    full = _SRTrackingCompose([_make_transform(t, noise_dir) for t in t_configs])

    # Detect pattern: last transform is Resample back to 16 kHz.
    # In that case we split: scoring uses all-but-last (at 8 kHz), output uses full.
    # This avoids the PESQ collapse caused by double-resampling (8k→16k→8k).
    last = t_configs[-1]
    if last["type"] == "Resample" and last.get("parameters", {}).get("min_sample_rate") == 16000 and len(t_configs) > 1:
        scoring_configs = t_configs[:-1]
        scoring = _SRTrackingCompose([_make_transform(t, noise_dir) for t in scoring_configs])
        # Infer scoring SR from the first Resample in the scoring chain
        scoring_sr = 16000
        for t in scoring_configs:
            if t["type"] == "Resample":
                scoring_sr = int(t["parameters"].get("min_sample_rate", 16000))
                break
        return PresetTransforms(full, scoring, scoring_sr)

    return PresetTransforms(full, None, None)


def list_builtin_presets() -> list[dict]:
    presets_dir = Path(__file__).parent / "presets"
    return [yaml.safe_load(f.read_text()) for f in sorted(presets_dir.glob("*.yaml"))]
