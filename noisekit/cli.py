from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="noisekit",
    help="Generate noisy speech datasets for ASR benchmark studies.",
    add_completion=False,
)
console = Console()


@app.command()
def generate(
    dataset: Annotated[str, typer.Option(help="HuggingFace dataset name (e.g. google/fleurs)")],
    samples: Annotated[int, typer.Option(help="Number of source samples to process")] = 100,
    preset: Annotated[
        list[str] | None,
        typer.Option(help="Preset to apply. Repeatable: --preset telecom --preset clean_reference."),
    ] = None,
    output: Annotated[
        Path | None, typer.Option(help="Output directory. Omit to auto-create ./output/<timestamp>/")
    ] = None,
    seed: Annotated[int, typer.Option(help="Random seed for dataset shuffling")] = 42,
    split: Annotated[str, typer.Option(help="Dataset split (e.g. train, validation, test)")] = "train",
    config: Annotated[str | None, typer.Option(help="HuggingFace dataset config name")] = None,
    preset_file: Annotated[Path | None, typer.Option(help="Path to a custom preset YAML file")] = None,
    noise_dir: Annotated[
        Path | None,
        typer.Option(
            "--noise-dir",
            help=(
                "Directory of background-noise WAVs (e.g. MUSAN, DEMAND, FSD50K). "
                "Used by the noise preset and compound noise presets. "
                "If omitted, a small MUSAN noise-only subset (~20 files, ~120 MB) "
                "is auto-downloaded to ~/.cache/noisekit/noise/musan_ambient/ on first use."
            ),
        ),
    ] = None,
    nisqa: Annotated[
        bool, typer.Option("--nisqa/--no-nisqa", help="Compute NISQA scores (downloads ~50 MB model on first use)")
    ] = True,
    transcript_column: Annotated[
        str | None,
        typer.Option(
            "--transcript-column",
            help=(
                "Dataset column to use as the transcript. "
                "Defaults to the first non-empty value among: text, sentence, transcription, normalized_text."
            ),
        ),
    ] = None,
) -> None:
    """Generate a degraded speech dataset by applying audio presets to a clean source dataset."""
    console.print(
        "[bold yellow]⚠  Synthetic pipeline[/bold yellow] — outputs approximate real-world degradation "
        "but are not a substitute for true annotated production audio. "
        "Validate final benchmarks on real degraded recordings.",
        style="yellow",
    )
    from .pipeline import run_generate

    resolved_output = output if output is not None else Path("./output") / datetime.now().strftime("%Y-%m-%d_%H%M%S")

    run_generate(
        dataset=dataset,
        samples=samples,
        presets=list(preset) if preset else [],
        output=resolved_output,
        seed=seed,
        split=split,
        config=config,
        preset_file=preset_file,
        noise_dir=noise_dir,
        nisqa=nisqa,
        transcript_column=transcript_column,
    )


@app.command()
def score(
    input_dir: Annotated[Path, typer.Argument(help="Directory containing WAV files to score")],
    reference_dir: Annotated[
        Path | None,
        typer.Option(help="Directory with matching reference WAVs (enables PESQ + SNR scoring)"),
    ] = None,
    output: Annotated[Path, typer.Option(help="Output JSON path for scores")] = Path("./scores.json"),
    nisqa: Annotated[
        bool, typer.Option("--nisqa/--no-nisqa", help="Compute NISQA scores (downloads ~50 MB model on first use)")
    ] = True,
) -> None:
    """Compute quality scores (PESQ, SNR, NISQA) for an existing audio folder."""
    from .pipeline import run_score

    if not input_dir.exists():
        console.print(f"[red]Input directory not found: {input_dir}[/red]")
        raise typer.Exit(1)

    run_score(input_dir=input_dir, reference_dir=reference_dir, output=output, nisqa=nisqa)


@app.command("list-presets")
def list_presets(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show full transform list for each preset")] = False,
) -> None:
    """Print all available built-in presets and their transform stacks."""
    from .transforms import list_builtin_presets

    presets = list_builtin_presets()

    table = Table(title="Built-in Presets", show_lines=True)
    table.add_column("Name", style="bold cyan", no_wrap=True)
    table.add_column("Description")
    if verbose:
        table.add_column("Transforms")

    for p in presets:
        if "chain" in p:
            transforms_str = "chain: " + " → ".join(p["chain"])
        else:
            transforms_str = " → ".join(f"{t['type']}(p={t.get('p', 1.0)})" for t in p.get("transforms", []))
        if verbose:
            table.add_row(p["name"], p["description"], transforms_str)
        else:
            table.add_row(p["name"], p["description"])

    console.print(table)
    console.print(f"\n[dim]{len(presets)} built-in presets. Pass a custom preset with --preset-file ./my.yaml[/dim]")
