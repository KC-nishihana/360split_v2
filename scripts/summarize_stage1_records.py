import json
from collections import Counter
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def num_list(values):
    return [value for value in values if isinstance(value, (int, float))]


def percentile(values, p):
    sorted_values = sorted(num_list(values))
    if not sorted_values:
        return None
    k = (len(sorted_values) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def corr(x_values, y_values):
    x = num_list(x_values)
    y = num_list(y_values)
    m = min(len(x), len(y))
    x = x[:m]
    y = y[:m]
    if m < 2:
        return None

    mx = statistics.fmean(x)
    my = statistics.fmean(y)
    sx = sum((v - mx) ** 2 for v in x)
    sy = sum((v - my) ** 2 for v in y)
    if sx == 0 or sy == 0:
        return 0.0

    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return cov / (sx**0.5 * sy**0.5)


def fmt(value, digits=6):
    return 'N/A' if value is None else f'{value:.{digits}f}'


def main():
    base = Path('/Users/satoshi/01_development/360split')
    jsonl_path = base / 'logs/after_rig_on/stage1_records.jsonl'
    out_dir = base / 'logs/after_rig_on'
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with jsonl_path.open('r', encoding='utf-8') as input_file:
        for line in input_file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    n = len(records)
    if n == 0:
        raise SystemExit('No records found')

    frame_idx = [r.get('frame_index') for r in records]
    timestamps = [r.get('timestamp') for r in records]
    quality = [r.get('quality') for r in records]
    quality_a = [r.get('quality_lens_a') for r in records]
    quality_b = [r.get('quality_lens_b') for r in records]
    sky_a = [r.get('sky_ratio_lens_a') for r in records]
    sky_b = [r.get('sky_ratio_lens_b') for r in records]
    pass_flags = [bool(r.get('is_pass')) for r in records]
    drop_reason = [r.get('drop_reason', 'unknown') for r in records]

    q = num_list(quality)
    qa = num_list(quality_a)
    qb = num_list(quality_b)

    pass_count = sum(pass_flags)
    fail_count = n - pass_count
    pass_rate = pass_count / n * 100

    reason_counts = Counter(drop_reason)
    dom_counts = Counter(r.get('lr_dominant_lens', 'unknown') for r in records)

    corr_q_sky_a = corr(quality, sky_a)
    corr_q_sky_b = corr(quality, sky_b)

    plt.style.use('default')

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(frame_idx, quality, label='quality', linewidth=1.0)
    if qa:
        ax.plot(frame_idx, quality_a, label='quality_lens_a', linewidth=0.8, alpha=0.8)
    if qb:
        ax.plot(frame_idx, quality_b, label='quality_lens_b', linewidth=0.8, alpha=0.8)
    ax.set_title('Stage1 Quality over Frame')
    ax.set_xlabel('frame_index')
    ax.set_ylabel('quality')
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'stage1_quality_timeseries.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(q, bins=40, alpha=0.85)
    ax.set_title('Quality Distribution')
    ax.set_xlabel('quality')
    ax.set_ylabel('count')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / 'stage1_quality_histogram.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    if num_list(sky_a):
        ax.plot(frame_idx, sky_a, label='sky_ratio_lens_a', linewidth=1.0)
    if num_list(sky_b):
        ax.plot(frame_idx, sky_b, label='sky_ratio_lens_b', linewidth=1.0)
    ax.set_title('Sky Ratio over Frame')
    ax.set_xlabel('frame_index')
    ax.set_ylabel('sky_ratio')
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'stage1_sky_ratio_timeseries.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    items = reason_counts.most_common()
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    ax.bar(labels, vals)
    ax.set_title('Drop Reason Counts')
    ax.set_xlabel('drop_reason')
    ax.set_ylabel('count')
    for i, value in enumerate(vals):
        ax.text(i, value, str(value), ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / 'stage1_drop_reason_counts.png', dpi=160)
    plt.close(fig)

    report_path = out_dir / 'stage1_records_summary.md'
    timestamp_values = num_list(timestamps)
    start_ts = timestamp_values[0] if timestamp_values else None
    end_ts = timestamp_values[-1] if timestamp_values else None

    with report_path.open('w', encoding='utf-8') as output_file:
        output_file.write('# stage1_records.jsonl 集計サマリ\n\n')
        output_file.write('## 1. 基本情報\n')
        output_file.write(f'- レコード数: {n}\n')
        if start_ts is not None and end_ts is not None:
            output_file.write(f'- 期間(秒): {fmt(start_ts, 3)} 〜 {fmt(end_ts, 3)} (約 {fmt(end_ts - start_ts, 3)} 秒)\n')
        output_file.write(f'- Pass: {pass_count} / {n} ({pass_rate:.2f}%)\n')
        output_file.write(f'- Fail: {fail_count} / {n} ({100 - pass_rate:.2f}%)\n')

        output_file.write('\n## 2. Quality統計\n')
        output_file.write(f'- mean: {fmt(statistics.fmean(q))}\n')
        output_file.write(f'- std: {fmt(statistics.pstdev(q))}\n')
        output_file.write(
            f'- min / p25 / median / p75 / max: '
            f'{fmt(min(q))} / {fmt(percentile(q, 0.25))} / {fmt(statistics.median(q))} / {fmt(percentile(q, 0.75))} / {fmt(max(q))}\n'
        )

        if qa and qb:
            output_file.write('\n## 3. レンズ別Quality\n')
            output_file.write(f'- lens_a mean: {fmt(statistics.fmean(qa))} (min={fmt(min(qa))}, max={fmt(max(qa))})\n')
            output_file.write(f'- lens_b mean: {fmt(statistics.fmean(qb))} (min={fmt(min(qb))}, max={fmt(max(qb))})\n')
            output_file.write(f'- quality優位レンズ(カウント): {dict(dom_counts)}\n')

        output_file.write('\n## 4. Drop理由\n')
        for reason, count in reason_counts.most_common():
            output_file.write(f'- {reason}: {count} ({count / n * 100:.2f}%)\n')

        output_file.write('\n## 5. 相関(参考)\n')
        output_file.write(f'- corr(quality, sky_ratio_lens_a): {fmt(corr_q_sky_a, 4)}\n')
        output_file.write(f'- corr(quality, sky_ratio_lens_b): {fmt(corr_q_sky_b, 4)}\n')

        output_file.write('\n## 6. 可視化\n')
        output_file.write('- ![quality timeseries](stage1_quality_timeseries.png)\n')
        output_file.write('- ![quality histogram](stage1_quality_histogram.png)\n')
        output_file.write('- ![sky ratio timeseries](stage1_sky_ratio_timeseries.png)\n')
        output_file.write('- ![drop reason counts](stage1_drop_reason_counts.png)\n')

    print(f'Wrote report: {report_path}')
    print(f'Reason counts: {dict(reason_counts)}')
    print(f'Pass rate: {pass_rate:.4f}%')
    print(
        'Quality mean/std/min/max:',
        statistics.fmean(q),
        statistics.pstdev(q),
        min(q),
        max(q),
    )


if __name__ == '__main__':
    main()
