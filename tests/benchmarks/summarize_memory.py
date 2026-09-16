"""Render isolated-process RSS results as grouped GitHub-flavored Markdown."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

MIB = 1024**2
GROUPS = (
    ('image/', 'Image'),
    ('preprocessing/', 'Preprocessing'),
    ('filtering/', 'Filtering'),
    ('radiomics/', 'Radiomics'),
    ('ibsi/i/', 'IBSI I'),
    ('ibsi/ii/phase_i/', 'IBSI II phase I'),
    ('ibsi/ii/phase_ii/', 'IBSI II phase II'),
)


def group_name(workload_id):
    for prefix, title in GROUPS:
        if workload_id.startswith(prefix):
            return title
    return 'Other'


def _metric(sample, key):
    value = sample[key]
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'Invalid {key} for {sample["workload_id"]}: {value!r}')
    return value / MIB


def _one_value(samples, key):
    values = {sample[key] for sample in samples}
    return str(next(iter(values))) if len(values) == 1 else 'mixed'


def _count(number, noun):
    return f'{number} {noun}{"" if number == 1 else "s"}'


def _summary_data(payload):
    if payload.get('schema_version') != 1 or not payload.get('measurements'):
        raise ValueError('Expected a nonempty schema-version-1 memory report.')
    measurements = payload['measurements']
    if any(sample.get('mode') != 'rss' for sample in measurements):
        raise ValueError('This summary supports RSS measurements only.')

    by_workload = defaultdict(list)
    for sample in measurements:
        by_workload[sample['workload_id']].append(sample)
    counts = {len(samples) for samples in by_workload.values()}
    repeats = str(next(iter(counts))) if len(counts) == 1 else 'varying'
    multi_sample = max(counts) > 1
    sample_count = (
        f'{_count(int(repeats), "sample")} per workload' if repeats != 'varying' else 'varying samples per workload'
    )

    grouped = defaultdict(list)
    for workload_id, samples in by_workload.items():
        peaks = [_metric(sample, 'peak_rss_bytes') for sample in samples]
        setups = [_metric(sample, 'setup_peak_rss_bytes') for sample in samples]
        grouped[group_name(workload_id)].append((workload_id, median(peaks), min(peaks), max(peaks), median(setups)))

    titles = [title for _, title in GROUPS] + ['Other']
    return {
        'count': f'{_count(len(by_workload), "workload")} · {_count(len(measurements), "measurement")} · {sample_count}',
        'commit': _one_value(measurements, 'commit'),
        'platform': _one_value(measurements, 'platform'),
        'python': _one_value(measurements, 'python'),
        'multi_sample': multi_sample,
        'groups': [(title, sorted(grouped[title])) for title in titles if grouped[title]],
    }


def render_summary(payload):
    data = _summary_data(payload)
    lines = [
        '## Peak RSS memory benchmarks',
        '',
        data['count'],
        '',
        f'Commit: `{data["commit"]}` · Platform: `{data["platform"]}` · Python: `{data["python"]}`',
        '',
    ]
    for title, workloads in data['groups']:
        lines.extend([f'### {title} — {_count(len(workloads), "workload")}', ''])
        if data['multi_sample']:
            lines.extend(
                [
                    '| Workload | Median peak RSS (MiB) | Peak range (MiB) | Median setup peak (MiB) |',
                    '|---|---:|---:|---:|',
                ]
            )
        else:
            lines.extend(['| Workload | Peak RSS (MiB) | Setup peak (MiB) |', '|---|---:|---:|'])
        for workload_id, peak, low, high, setup in workloads:
            if data['multi_sample']:
                lines.append(f'| `{workload_id}` | {peak:.1f} | {low:.1f}–{high:.1f} | {setup:.1f} |')
            else:
                lines.append(f'| `{workload_id}` | {peak:.1f} | {setup:.1f} |')
        lines.append('')

    lines.extend(
        [
            'Peak RSS is the process-lifetime high-water mark through the operation, '
            'including imports, inputs, and setup. Setup peak is an earlier high-water mark; '
            'subtracting it from final peak does not measure operation-only allocations.',
            '',
            'These absolute measurements are informational, not memory-regression verdicts.',
            '',
        ]
    )
    return '\n'.join(lines)


def _console_table(headers, rows):
    widths = [max(len(cell) for cell in column) for column in zip(headers, *rows)]

    def format_row(row):
        return '  '.join(
            cell.ljust(width) if index == 0 else cell.rjust(width)
            for index, (cell, width) in enumerate(zip(row, widths))
        )

    return [format_row(headers), format_row(tuple('-' * width for width in widths)), *(format_row(row) for row in rows)]


def render_console_summary(payload):
    data = _summary_data(payload)
    lines = [
        'Peak RSS memory benchmarks',
        data['count'],
        f'Commit: {data["commit"]} | Platform: {data["platform"]} | Python: {data["python"]}',
        '',
    ]
    for title, workloads in data['groups']:
        lines.append(f'{title} ({_count(len(workloads), "workload")})')
        if data['multi_sample']:
            headers = ('Workload', 'Median peak (MiB)', 'Peak range (MiB)', 'Median setup (MiB)')
            rows = [
                (workload_id, f'{peak:.1f}', f'{low:.1f}–{high:.1f}', f'{setup:.1f}')
                for workload_id, peak, low, high, setup in workloads
            ]
        else:
            headers = ('Workload', 'Peak (MiB)', 'Setup peak (MiB)')
            rows = [(workload_id, f'{peak:.1f}', f'{setup:.1f}') for workload_id, peak, _, _, setup in workloads]
        lines.extend(_console_table(headers, rows))
        lines.append('')
    lines.append('Peak includes imports, inputs, and setup; setup peak is not an operation-only allocation baseline.')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path, help='RSS JSON produced by memory.py.')
    args = parser.parse_args()
    print(render_summary(json.loads(args.report.read_text())), end='')


if __name__ == '__main__':
    main()
