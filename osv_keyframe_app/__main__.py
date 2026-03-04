"""CLI entry point: python -m osv_keyframe_app --config config.yaml"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="osv_keyframe_app",
        description="OSV Keyframe Extractor - Extract keyframes from dual-fisheye OSV video",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--osv",
        help="Path to OSV file (overrides config.osv_path)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory (overrides config.output_dir)",
    )
    parser.add_argument(
        "--no-colmap", action="store_true",
        help="Skip COLMAP processing even if enabled in config",
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Launch GUI mode",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    _setup_logging(args.verbose)

    from osv_keyframe_app.config import AppConfig
    config = AppConfig.from_yaml(args.config)

    # Apply CLI overrides
    if args.osv:
        config.osv_path = args.osv
    if args.output:
        config.output_dir = args.output
    if args.no_colmap:
        config.colmap.enabled = False

    if args.gui:
        from osv_keyframe_app.gui.app import launch_gui
        launch_gui(config)
        return

    # CLI mode
    osv_path = config.osv_path
    if not osv_path:
        print("Error: --osv or config.osv_path is required", file=sys.stderr)
        sys.exit(1)

    if not Path(osv_path).exists():
        print(f"Error: OSV file not found: {osv_path}", file=sys.stderr)
        sys.exit(1)

    from osv_keyframe_app.pipeline import Pipeline

    def progress_callback(progress: float, message: str) -> None:
        bar_len = 30
        filled = int(bar_len * progress)
        bar = "=" * filled + "-" * (bar_len - filled)
        print(f"\r[{bar}] {progress:.0%} {message}", end="", flush=True)
        if progress >= 1.0:
            print()

    pipeline = Pipeline(config)
    result = pipeline.run(osv_path, on_progress=progress_callback)

    # Optional COLMAP
    if config.colmap.enabled:
        from osv_keyframe_app.colmap_runner import ColmapRunner
        runner = ColmapRunner(config.colmap, config.projection)
        output_dir = Path(config.output_dir)

        print("\nRunning COLMAP SfM...")
        runner.run_sfm(
            output_dir / "sfm" / "images",
            output_dir / config.colmap.workspace,
        )

        print("Running COLMAP 3DGS registration...")
        runner.run_gs_registration(
            output_dir / "gs" / "images",
            output_dir / config.colmap.workspace,
        )

    print(f"\nDone! Output: {config.output_dir}")
    print(f"  Metrics: {result.metrics_csv}")
    print(f"  Manifest: {result.manifest_csv}")
    if result.selection:
        print(f"  SfM frames: {result.selection.sfm_count}")
        print(f"  3DGS frames: {result.selection.gs_count}")


if __name__ == "__main__":
    main()
