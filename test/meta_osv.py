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
# 注意: あなたの元コードは -c copy を2回書いていて1コマンドで2出力にしているが、
# ffmpegは出力ごとにオプションが必要になりやすいので、ここでは2回に分ける（安全策）。
subprocess.run([
    "ffmpeg", "-y", "-i", osv,
    "-map", LEFT_MAP, "-c", "copy", left_mp4
], check=True)

subprocess.run([
    "ffmpeg", "-y", "-i", osv,
    "-map", RIGHT_MAP, "-c", "copy", right_mp4
], check=True)

# ====== (2) フレーム抽出（左のみ） ======
# OpenCVで全フレームPNGは重いので、必要なら間引き(fps)を入れてください。
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
# exiftool -ee3 -api Unknown=2 で Protobuf timed metadata を展開
# zsh対策はsubprocessなので不要
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

# dumpをTimeStamp単位で1レコードにまとめる
line_re = re.compile(r"^\[Protobuf\]\s+(.+?)\s+:\s+(.*)$")

records = []
cur = None

def flush():
    global cur
    if not cur:
        return
    # TimeStamp + Accel3軸が最低限揃っている行だけ採用（品質担保）
    if "TimeStamp" in cur and all(k in cur for k in ("AccelerometerX", "AccelerometerY", "AccelerometerZ")):
        records.append(cur)
    cur = None

for line in dump.splitlines():
    m = line_re.match(line)
    if not m:
        continue
    tag = m.group(1).strip()
    val = m.group(2).strip()

    if tag == "TimeStamp":
        flush()
        cur = {"TimeStamp": float(val)}
        continue

    if cur is None:
        continue

    if tag in ("AccelerometerX", "AccelerometerY", "AccelerometerZ",
               "AngularVelocityX", "AngularVelocityY", "AngularVelocityZ"):
        # 角速度が存在しないOSVもあるので、float変換失敗時は捨てる
        try:
            cur[tag] = float(val)
        except ValueError:
            pass

flush()

if not records:
    raise RuntimeError("No IMU records parsed. exiftool output may not include TimeStamp/Accelerometer tags.")

# TimeStampを0始まりに正規化（動画PTSと合わせやすい）
t0 = records[0]["TimeStamp"]
for r in records:
    r["TimeStamp0"] = r["TimeStamp"] - t0

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
# best_effort_timestamp_time を使う（フレームの秒）
# coded_picture_number はフレーム番号（連番とは限らないが、連結キーとして使える）
subprocess.run([
    "ffprobe",
    "-v", "error",
    "-select_streams", "v:0",
    "-show_frames",
    "-show_entries", "frame=coded_picture_number,best_effort_timestamp_time",
    "-of", "csv=p=0",
    left_mp4
], check=True, stdout=open(frame_pts_csv, "w", encoding="utf-8"))

print(f"[OK] Wrote frame PTS -> {frame_pts_csv}")

# ====== (5) フレームPTSとIMU(TimeStamp0)を最近傍で紐付け ======
# ここで「TimeStamp0（秒）」と「frame_pts（秒）」を同一基準にする。
# ※ OSVによってはTimeStampが別基準の可能性があるが、あなたのログでは秒オーダーで単調増加。
#    まずはTimeStamp0で合わせ、ズレる場合はオフセット/スケール補正を入れる。
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
# 形式: coded_picture_number,best_effort_timestamp_time
frame_rows = []
with frame_pts_csv.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            frame_no = int(parts[0])
            t = float(parts[1])
        except ValueError:
            continue
        frame_rows.append((frame_no, t))

# 出力: frame_index(=画像連番), pts_sec, imu_index, imu_TimeStamp0, accel/gyro...
# 画像は frame_000000.png のように idx で保存したので idx を採用する
# PTSのフレーム番号とOpenCVのidxが一致しないケースもあるので、ここは「順番」を優先して idx を使う
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

    # frame_rows の順序はPTS順（基本）なので、先頭から idx を付ける
    for idx2, (_, pts) in enumerate(frame_rows):
        imu_idx = nearest_imu_index(pts)
        r = records[imu_idx]

        row = {
            "frame_idx": idx2,
            "frame_png": str(left_img_dir / f"frame_{idx2:06d}.png"),
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

print(f"[OK] Wrote frame-IMU aligned CSV -> {imu_frames_csv}")
