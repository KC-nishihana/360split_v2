"""
camd boxの詳細解析

camd = Camera Metadata
このboxにはカメラの詳細なメタデータが含まれている可能性があります。
GPS、ジャイロ、その他のセンサーデータなど。
"""

import struct
from pathlib import Path

osv_file = Path("CAM_20260205143223_0028_D.OSV")

# camd boxの位置（前のスクリプトの結果から）
camd_position = 0x6432FD29
camd_size = 2529934

print(f"[INFO] Analyzing camd box...")
print(f"[INFO] Position: 0x{camd_position:X}")
print(f"[INFO] Size: {camd_size:,} bytes ({camd_size / 1024:.1f} KB)")
print()

with open(osv_file, 'rb') as f:
    # camd boxに移動
    f.seek(camd_position)

    # ヘッダーを読む（最初の8バイト）
    header = f.read(16)
    size = struct.unpack('>I', header[:4])[0]
    box_type = header[4:8]

    print(f"Box type: {box_type}")
    print(f"Box size: {size:,} bytes")
    print()

    # データの最初の部分を読む
    preview_size = min(2048, camd_size - 16)
    data = f.read(preview_size)

    # パターンを検索
    patterns = {
        b'GPS': 'GPS data',
        b'gps': 'GPS data (lowercase)',
        b'Gyro': 'Gyroscope',
        b'gyro': 'Gyroscope (lowercase)',
        b'Angular': 'Angular velocity',
        b'Mag': 'Magnetometer',
        b'mag': 'Magnetometer (lowercase)',
        b'Compass': 'Compass',
        b'Pressure': 'Barometer/Pressure',
        b'pressure': 'Barometer/Pressure (lowercase)',
        b'Altitude': 'Altitude',
        b'altitude': 'Altitude (lowercase)',
        b'Latitude': 'Latitude',
        b'latitude': 'Latitude (lowercase)',
        b'Longitude': 'Longitude',
        b'longitude': 'Longitude (lowercase)',
        b'Speed': 'Speed',
        b'speed': 'Speed (lowercase)',
        b'Accel': 'Accelerometer',
        b'accel': 'Accelerometer (lowercase)',
    }

    print("="*80)
    print("Pattern Search Results")
    print("="*80)

    found_patterns = []
    for pattern, description in patterns.items():
        count = data.count(pattern)
        if count > 0:
            print(f"✓ {description}: {count} occurrences")
            found_patterns.append(description)

    if not found_patterns:
        print("❌ No common sensor keywords found")

    print()
    print("="*80)
    print("Data Preview (first 1024 bytes)")
    print("="*80)

    # ASCII表示
    print("\nASCII representation:")
    for i in range(0, min(1024, len(data)), 64):
        chunk = data[i:i+64]
        ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
        print(f"  {i:04X}: {ascii_str}")

    print("\nHEX dump (first 512 bytes):")
    for i in range(0, min(512, len(data)), 16):
        chunk = data[i:i+16]
        hex_str = ' '.join(f'{b:02X}' for b in chunk)
        ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
        print(f"  {i:04X}: {hex_str:<48} {ascii_str}")

    # バイナリデータの統計
    print()
    print("="*80)
    print("Binary Statistics")
    print("="*80)

    # バイト値の分布
    byte_counts = {}
    for b in data[:1024]:
        byte_counts[b] = byte_counts.get(b, 0) + 1

    # 最も頻出するバイト
    top_bytes = sorted(byte_counts.items(), key=lambda x: -x[1])[:10]
    print("\nMost common bytes (first 1024 bytes):")
    for byte_val, count in top_bytes:
        char = chr(byte_val) if 32 <= byte_val < 127 else f'\\x{byte_val:02X}'
        print(f"  0x{byte_val:02X} ({char}): {count} times")

    # データ構造の推測
    print()
    print("="*80)
    print("Possible Data Structure")
    print("="*80)

    # 繰り返しパターンを探す
    print("\nSearching for repeating patterns...")

    # サンプル間隔をチェック（30Hzなら約2310サンプル）
    expected_samples = 2310  # 77秒 × 30Hz
    bytes_per_sample = camd_size // expected_samples

    print(f"  Expected samples (30Hz, 77s): {expected_samples}")
    print(f"  Bytes per sample: {bytes_per_sample}")

    if bytes_per_sample > 0:
        print(f"\n  → If camd contains sensor data at 30Hz:")
        print(f"     Each sample might be ~{bytes_per_sample} bytes")

    # Track 4-7の情報を確認
    print()
    print("="*80)
    print("Additional Tracks (from previous analysis)")
    print("="*80)
    print("  Track 3: Audio (AAC)")
    print("  Track 4: gmhd (Generic Media Header) - Unknown type")
    print("  Track 5: gmhd (Generic Media Header) - Unknown type")
    print("  Track 6: gmhd (Generic Media Header) - Unknown type")
    print("  Track 7: gmhd (Generic Media Header) - Unknown type")
    print()
    print("  → Tracks 4-7 likely contain sensor data (Protobuf format)")
    print("  → Use ffprobe -show_streams to inspect these tracks")

print()
print("="*80)
print("Recommendations")
print("="*80)
print("\n1. Install exiftool to extract Protobuf data:")
print("   sudo apt-get install exiftool")
print("   exiftool -ee3 -Protobuf:all CAM_20260205143223_0028_D.OSV")
print()
print("2. Check track 4-7 for sensor data:")
print("   ffprobe -show_streams -select_streams 4 CAM_20260205143223_0028_D.OSV")
print()
print("3. Extract camd box for detailed analysis:")
print("   dd if=CAM_20260205143223_0028_D.OSV of=camd_data.bin \\")
print(f"      bs=1 skip={camd_position + 8} count={camd_size - 8}")
