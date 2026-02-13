"""
OSVファイルのメタデータを詳細にチェック

MP4コンテナ内のboxを解析して、自己位置推定に使えるデータを探す：
- GPS情報
- ジャイロスコープ（角速度）
- 磁気センサー
- 気圧センサー
- その他のセンサーデータ
"""

import struct
from pathlib import Path

osv_file = Path("CAM_20260205143223_0028_D.OSV")

# MP4のbox type定義
BOX_TYPES = {
    b'ftyp': 'File Type',
    b'moov': 'Movie',
    b'mdat': 'Media Data',
    b'udta': 'User Data',
    b'meta': 'Metadata',
    b'mvhd': 'Movie Header',
    b'trak': 'Track',
    b'mdia': 'Media',
    b'minf': 'Media Information',
    b'stbl': 'Sample Table',
    b'camm': 'Camera Motion Metadata',  # カメラモーションメタデータ
    b'free': 'Free Space',
    b'skip': 'Skip',
    b'wide': 'Wide',
    b'uuid': 'UUID',
    b'gps ': 'GPS Data',
}

def read_box_header(f):
    """MP4 boxのヘッダーを読む"""
    data = f.read(8)
    if len(data) < 8:
        return None, None, None

    size = struct.unpack('>I', data[:4])[0]
    box_type = data[4:8]

    # サイズが1の場合は64bit拡張サイズ
    if size == 1:
        size = struct.unpack('>Q', f.read(8))[0]
        header_size = 16
    else:
        header_size = 8

    return size, box_type, header_size

def explore_boxes(f, indent=0, max_depth=10, parent_size=None, max_boxes=1000):
    """MP4 boxを再帰的に探索"""
    results = []

    if indent > max_depth:
        return results

    start_pos = f.tell()
    box_count = 0

    while box_count < max_boxes:
        if parent_size and (f.tell() - start_pos) >= parent_size - 8:
            break

        size, box_type, header_size = read_box_header(f)

        if size is None:
            break

        box_count += 1

        box_name = BOX_TYPES.get(box_type, box_type.decode('latin1', errors='ignore'))

        # 興味深いboxを記録
        info = {
            'type': box_type,
            'name': box_name,
            'size': size,
            'position': f.tell() - header_size,
            'indent': indent
        }

        # 特定のboxタイプをハイライト
        if box_type in [b'udta', b'meta', b'camm', b'uuid', b'gps ']:
            info['highlight'] = True
            # データの一部を読む
            current_pos = f.tell()
            preview_size = min(256, size - header_size)
            if preview_size > 0:
                info['data_preview'] = f.read(preview_size)
            f.seek(current_pos)

        results.append(info)

        # コンテナboxの場合は再帰的に探索
        if box_type in [b'moov', b'trak', b'mdia', b'minf', b'stbl', b'udta', b'meta']:
            # 子boxを探索
            child_start = f.tell()
            child_results = explore_boxes(f, indent + 1, max_depth, size - header_size, max_boxes)
            results.extend(child_results)
            # 次のboxに移動
            f.seek(child_start + (size - header_size))
        else:
            # 次のboxに移動
            if size > header_size:
                f.seek(f.tell() + size - header_size)

        # ファイル終端チェック
        current_pos = f.tell()

        if size == 0:
            break

    return results

print(f"[INFO] Analyzing {osv_file}...")
print(f"[INFO] File size: {osv_file.stat().st_size / (1024**3):.2f} GB")
print()

with open(osv_file, 'rb') as f:
    boxes = explore_boxes(f, max_depth=15)

print("="*80)
print("MP4 Box Structure")
print("="*80)

# 全boxの構造を表示
for box in boxes:
    indent_str = "  " * box['indent']
    highlight = " ⭐" if box.get('highlight') else ""
    print(f"{indent_str}[{box['name']}] size={box['size']:,} bytes @ 0x{box['position']:X}{highlight}")

print()
print("="*80)
print("Interesting Boxes (Potential Position Data)")
print("="*80)

# 興味深いboxを詳細表示
interesting = [b for b in boxes if b.get('highlight')]

if interesting:
    for box in interesting:
        print(f"\n📦 {box['name']} ({box['type']})")
        print(f"   Size: {box['size']:,} bytes")
        print(f"   Position: 0x{box['position']:X}")

        if 'data_preview' in box:
            preview = box['data_preview']
            print(f"   Data preview (first {len(preview)} bytes):")

            # ASCII表示
            ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in preview)
            print(f"   ASCII: {ascii_str[:80]}")

            # Hex表示
            hex_str = ' '.join(f'{b:02X}' for b in preview[:64])
            print(f"   HEX: {hex_str}")

            # 特定のパターンを検索
            if b'GPS' in preview or b'gps' in preview:
                print("   ⭐ GPS data detected!")
            if b'Gyro' in preview or b'gyro' in preview or b'Angular' in preview:
                print("   ⭐ Gyroscope data detected!")
            if b'Mag' in preview or b'mag' in preview or b'Compass' in preview:
                print("   ⭐ Magnetometer data detected!")
            if b'Pressure' in preview or b'pressure' in preview or b'Altitude' in preview:
                print("   ⭐ Barometer data detected!")
else:
    print("\n❌ No special metadata boxes found (udta, meta, camm, uuid, gps)")

# 統計情報
print("\n" + "="*80)
print("Summary")
print("="*80)
print(f"Total boxes found: {len(boxes)}")
print(f"Interesting boxes: {len(interesting)}")
print(f"\nBox type distribution:")
box_types = {}
for box in boxes:
    box_types[box['name']] = box_types.get(box['name'], 0) + 1

for box_type, count in sorted(box_types.items(), key=lambda x: -x[1])[:20]:
    print(f"  {box_type}: {count}")

print("\n[INFO] To extract detailed sensor data, use:")
print("  exiftool -ee3 -api Unknown=2 -G -a -s CAM_20260205143223_0028_D.OSV")
print("\n[NOTE] If gyroscope or GPS data exists, it may be in Protobuf format")
print("[NOTE] Use: exiftool -ee3 -Protobuf:all CAM_20260205143223_0028_D.OSV")
