"""
OSMO 360 .OSV
- left/right video extract
- left frames export (png)
- IMU(TimeStamp + Accel/Gyro) extract from Protobuf via exiftool -ee3
- (optional) align IMU samples to left video frame PTS by nearest timestamp

必要:
  - ffmpeg, ffprobe (PATH)
  - exiftool (PATH)
  - pip: opencv-python

修正版: exiftoolの出力形式（TimeStampとAccelerometerが別ブロック）に対応
"""

import os
import re
import csv
import subprocess
from pathlib import Path
from bisect import bisect_left

# ====== Settings ======
osv = "CAM_20260205143223_0028_D.OSV"
left_mp4 = "left_eye.mp4"
right_mp4 = "right_eye.mp4"

left_img_dir = Path("left_images")
left_img_dir.mkdir(exist_ok=True)

imu_dump_txt = Path("imu_dump.txt")
imu_csv = Path("imu_timeseries.csv")
frame_pts_csv = Path("frame_pts.csv")
imu_frames_csv = Path("imu_frames.csv")

# map indexは環境で変わるので、まずは0:v:0, 0:v:1で分離するのが安全
LEFT_MAP = "0:v:0"
RIGHT_MAP = "0:v:1"

# ====== (1) 映像ストリーム分離 ======
# 既に生成済みの場合はスキップ
if not Path(left_mp4).exists():
    print(f"[INFO] Extracting left video stream...")
    subprocess.run([
        "ffmpeg", "-y", "-i", osv,
        "-map", LEFT_MAP, "-c", "copy", left_mp4
    ], check=True)
else:
    print(f"[SKIP] {left_mp4} already exists")

if not Path(right_mp4).exists():
    print(f"[INFO] Extracting right video stream...")
    subprocess.run([
        "ffmpeg", "-y", "-i", osv,
        "-map", RIGHT_MAP, "-c", "copy", right_mp4
    ], check=True)
else:
    print(f"[SKIP] {right_mp4} already exists")

# ====== (2) フレーム抽出（左のみ） ======
# 既に生成済みの場合はスキップ
frame_count = len(list(left_img_dir.glob("frame_*.png")))
if frame_count > 0:
    print(f"[SKIP] {frame_count} frames already exist in {left_img_dir}")
else:
    print(f"[INFO] Extracting frames from {left_mp4}...")
    import cv2

    cap = cv2.VideoCapture(left_mp4)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {left_mp4}")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(left_img_dir / f"frame_{idx:06d}.png"), frame)
        idx += 1
    cap.release()
    print(f"[OK] Extracted {idx} frames -> {left_img_dir}")

# ====== (3) メタデータ抽出（TimeStamp + IMU） ======
# 既に生成済みの場合はスキップ
if not imu_dump_txt.exists():
    print(f"[INFO] Extracting IMU metadata...")
    cmd = [
        "exiftool",
        "-ee3",
        "-api", "Unknown=2",
        "-G", "-a", "-s", "-n",
        "-Protobuf:TimeStamp",
        "-Protobuf:Accelerometer*",
        "-Protobuf:AngularVelocity*",
        osv
    ]
    dump = subprocess.check_output(cmd, text=True, errors="ignore")
    imu_dump_txt.write_text(dump, encoding="utf-8")
    print(f"[OK] Wrote IMU dump -> {imu_dump_txt}")
else:
    print(f"[SKIP] {imu_dump_txt} already exists, reading...")
    dump = imu_dump_txt.read_text(encoding="utf-8")

# ====== パース処理（修正版） ======
# exiftoolの出力はTimeStampとAccelerometerが別々のブロックに分かれている
# そのため、別々に抽出してから配列インデックスで結合する

print(f"[INFO] Parsing IMU data...")

line_re = re.compile(r"^\[Protobuf\]\s+(.+?)\s+:\s+(.*)$")

# (1) TimeStampを全て抽出
timestamps = []
for line in dump.splitlines():
    m = line_re.match(line)
    if m and m.group(1).strip() == "TimeStamp":
        try:
            timestamps.append(float(m.group(2).strip()))
        except ValueError:
            pass

print(f"[INFO] Found {len(timestamps)} TimeStamp entries")

# (2) Accelerometerを3軸セットで抽出
accel_data = []
temp_accel = {}
for line in dump.splitlines():
    m = line_re.match(line)
    if not m:
        continue
    tag = m.group(1).strip()
    val = m.group(2).strip()

    if tag in ("AccelerometerX", "AccelerometerY", "AccelerometerZ"):
        try:
            temp_accel[tag] = float(val)
            # X,Y,Z が揃ったら保存
            if len(temp_accel) == 3:
                accel_data.append(temp_accel.copy())
                temp_accel = {}
        except ValueError:
            pass

print(f"[INFO] Found {len(accel_data)} Accelerometer entries (XYZ sets)")

# (3) AngularVelocityを3軸セットで抽出
gyro_data = []
temp_gyro = {}
for line in dump.splitlines():
    m = line_re.match(line)
    if not m:
        continue
    tag = m.group(1).strip()
    val = m.group(2).strip()

    if tag in ("AngularVelocityX", "AngularVelocityY", "AngularVelocityZ"):
        try:
            temp_gyro[tag] = float(val)
            # X,Y,Z が揃ったら保存
            if len(temp_gyro) == 3:
                gyro_data.append(temp_gyro.copy())
                temp_gyro = {}
        except ValueError:
            pass

print(f"[INFO] Found {len(gyro_data)} AngularVelocity entries (XYZ sets)")

# (4) 全てを結合（配列インデックスで紐付け）
# TimeStampとAccelerometerの数は一致しているはず
# AngularVelocityは存在しない場合もある
records = []
min_len = min(len(timestamps), len(accel_data))

if min_len == 0:
    raise RuntimeError("No valid IMU records found. Check exiftool output format.")

# TimeStamp0を計算（動画PTSと合わせるため）
t0 = timestamps[0]

for i in range(min_len):
    record = {
        "TimeStamp": timestamps[i],
        "TimeStamp0": timestamps[i] - t0,
        "AccelerometerX": accel_data[i]["AccelerometerX"],
        "AccelerometerY": accel_data[i]["AccelerometerY"],
        "AccelerometerZ": accel_data[i]["AccelerometerZ"],
    }

    # AngularVelocityがある場合は追加
    if i < len(gyro_data):
        record["AngularVelocityX"] = gyro_data[i]["AngularVelocityX"]
        record["AngularVelocityY"] = gyro_data[i]["AngularVelocityY"]
        record["AngularVelocityZ"] = gyro_data[i]["AngularVelocityZ"]

    records.append(record)

print(f"[OK] Parsed {len(records)} IMU records")

# CSV出力（存在しない列は落とす）
cols = [
    "TimeStamp", "TimeStamp0",
    "AccelerometerX", "AccelerometerY", "AccelerometerZ",
    "AngularVelocityX", "AngularVelocityY", "AngularVelocityZ",
]
cols = [c for c in cols if any(c in r for r in records)]

with imu_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(records)

print(f"[OK] Wrote IMU timeseries -> {imu_csv} (rows={len(records)})")

# ====== (4) フレームPTS抽出（左動画） ======
if not frame_pts_csv.exists():
    print(f"[INFO] Extracting frame PTS...")
    with open(frame_pts_csv, "w", encoding="utf-8") as f:
        subprocess.run([
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_frames",
            "-show_entries", "frame=coded_picture_number,best_effort_timestamp_time",
            "-of", "csv=p=0",
            left_mp4
        ], check=True, stdout=f)
    print(f"[OK] Wrote frame PTS -> {frame_pts_csv}")
else:
    print(f"[SKIP] {frame_pts_csv} already exists")

# ====== (5) フレームPTSとIMU(TimeStamp0)を最近傍で紐付け ======
print(f"[INFO] Aligning IMU data with frames...")

imu_t = [r["TimeStamp0"] for r in records]

def nearest_imu_index(t: float) -> int:
    i = bisect_left(imu_t, t)
    if i <= 0:
        return 0
    if i >= len(imu_t):
        return len(imu_t) - 1
    # i-1 と i のどっちが近いか
    return i if (imu_t[i] - t) < (t - imu_t[i-1]) else (i - 1)

# frame_pts.csv の読み込み
# 既存のframe_pts.csvは1列のみ（タイムスタンプのみ）
frame_rows = []
with frame_pts_csv.open("r", encoding="utf-8") as f:
    frame_idx = 0
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        try:
            # 1列目がタイムスタンプ
            t = float(parts[0])
            # frame_noは行番号を使用
            frame_rows.append((frame_idx, t))
            frame_idx += 1
        except (ValueError, IndexError):
            continue

print(f"[INFO] Found {len(frame_rows)} frame PTS entries")

# 出力列
out_cols = [
    "frame_idx",
    "frame_png",
    "pts_sec",
    "imu_idx",
    "imu_TimeStamp0",
    "AccelerometerX", "AccelerometerY", "AccelerometerZ",
    "AngularVelocityX", "AngularVelocityY", "AngularVelocityZ",
]

with imu_frames_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=out_cols)
    w.writeheader()

    for idx, (_, pts) in enumerate(frame_rows):
        imu_idx = nearest_imu_index(pts)
        r = records[imu_idx]

        row = {
            "frame_idx": idx,
            "frame_png": str(left_img_dir / f"frame_{idx:06d}.png"),
            "pts_sec": pts,
            "imu_idx": imu_idx,
            "imu_TimeStamp0": r.get("TimeStamp0"),
            "AccelerometerX": r.get("AccelerometerX"),
            "AccelerometerY": r.get("AccelerometerY"),
            "AccelerometerZ": r.get("AccelerometerZ"),
            "AngularVelocityX": r.get("AngularVelocityX"),
            "AngularVelocityY": r.get("AngularVelocityY"),
            "AngularVelocityZ": r.get("AngularVelocityZ"),
        }
        w.writerow(row)

print(f"[OK] Wrote frame-IMU aligned CSV -> {imu_frames_csv} (rows={len(frame_rows)})")
print(f"\n=== Summary ===")
print(f"Frames extracted: {len(frame_rows)}")
print(f"IMU records: {len(records)}")
print(f"Frame-IMU alignment: {len(frame_rows)} rows")
print(f"\nOutput files:")
print(f"  - {imu_csv}")
print(f"  - {imu_frames_csv}")
print(f"  - {left_img_dir}/")
