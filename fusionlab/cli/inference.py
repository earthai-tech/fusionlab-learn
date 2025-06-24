
"""
Run an *already-trained* model (described by a run_manifest.json)
on a fresh CSV and produce forecasts + figures.
"""

from __future__ import annotations
import sys
# import json
import pathlib
import click

from fusionlab.tools.app.inference import PredictionPipeline
from fusionlab.tools.app.config    import SubsConfig


def _echo(msg: str, *, stream=sys.stdout) -> None:
    click.echo(msg, file=stream, nl=True)


class _ClickLogger:
    """Log-sink handed to PredictionPipeline – optionally writes a file and
    shows a crude progress bar (stderr)."""

    def __init__(self, log_file: str | None, show_progress: bool) -> None:
        self._fp = open(log_file, "w", encoding="utf-8") if log_file else None
        self._progress = show_progress

    def __call__(self, msg: str) -> None:             # → pipeline.log_callback
        _echo(msg)
        if self._fp:
            self._fp.write(msg + "\n"); self._fp.flush()

    # simple 0-100 % bar (▮ = 4 %)
    def progress(self, pct: int) -> None:             # → SubsConfig.progress
        if self._progress:
            bar = "▮" * (pct // 4)
            sys.stderr.write(f"\r[{bar:<25}] {pct:3d}%"); sys.stderr.flush()
            if pct == 100:
                sys.stderr.write("\n")

    def close(self) -> None:
        if self._fp: self._fp.close()


@click.command("inference")
@click.option(
    "-m", "--manifest", "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    required=True,
    help="Path to *run_manifest.json* created during training."
)
@click.option(
    "-c", "--csv", "csv_path",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    required=True,
    help="CSV file with new data to forecast."
)
@click.option(
    "--log-file", type=click.Path(dir_okay=False, writable=True, path_type=pathlib.Path),
    help="Duplicate console output to this file."
)
@click.option(
    "--progress/--no-progress", default=True,
    help="Render a simple progress bar in stderr (default: on)."
)
def inference_group(
    manifest_path: pathlib.Path,
    csv_path: pathlib.Path,
    log_file: pathlib.Path | None,
    progress: bool,
) -> None:
    """Top-level *inference* sub-command attached to `fusionlab-learn`."""
    logger = _ClickLogger(str(log_file) if log_file else None, progress)

    try:
        pipe = PredictionPipeline(
            manifest_path=str(manifest_path),
            log_callback=logger,
        )

        # feed GUI-style progress updates into our bar
        cfg: SubsConfig = pipe.config
        cfg.progress_callback = logger.progress

        pipe.run(validation_data_path=str(csv_path))

        _echo("\n✅  Inference completed successfully.")
    except Exception as exc:
        _echo(f"\n❌  Inference failed: {exc}", stream=sys.stderr)
        sys.exit(1)
    finally:
        logger.close()
