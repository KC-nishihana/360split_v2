# COLMAP Setup Guide (360Split)

## macOS (officially verified)

1. Install system dependencies.

```bash
brew update
brew install colmap ffmpeg
```

2. Create Python environment.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Verify tools.

```bash
colmap -h
python main.py --help
```

4. Run CLI with COLMAP backend.

```bash
python main.py --cli input.mp4 \
  --pose-backend colmap \
  --colmap-path colmap \
  --pose-export-format metashape
```

## Linux (reference)

- Install `colmap`, `ffmpeg`, `opencv` runtime deps using your package manager.
- Use the same Python setup and CLI validation as macOS.

## Windows (reference)

- Install COLMAP from official release binaries and add `colmap.exe` to `PATH`.
- Install FFmpeg and add it to `PATH`.
- Create venv and install requirements.

## Troubleshooting

- `COLMAP is not installed or not in PATH`
  - Run `which colmap` (macOS/Linux) or `where colmap` (Windows).
  - Or pass `--colmap-path /absolute/path/to/colmap`.

- `mapper did not produce sparse/0`
  - Reduce frame count.
  - Ensure enough overlap/texture.
  - Check `${workspace}/colmap_pose.log`.

- `produced no camera poses in images.txt`
  - Verify input images are not near-duplicates.
  - Try lower thresholds for keyframe selection.

- GUI run fails on COLMAP backend
  - Open log panel and search `COLMAP_LAST_ERROR`.
  - Confirm `pose_backend`, `colmap_path`, and workspace settings.
