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


def render_summary(payload):
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

    lines = [
        '## Peak RSS memory benchmarks',
        '',
        f'{_count(len(by_workload), "workload")} · {_count(len(measurements), "measurement")} · {sample_count}',
        '',
        f'Commit: `{_one_value(measurements, "commit")}` · '
        f'Platform: `{_one_value(measurements, "platform")}` · '
        f'Python: `{_one_value(measurements, "python")}`',
        '',
    ]
    grouped = defaultdict(list)
    for workload_id, samples in by_workload.items():
        grouped[group_name(workload_id)].append((workload_id, samples))

    titles = [title for _, title in GROUPS] + ['Other']
    for title in titles:
        workloads = grouped.get(title)
        if not workloads:
            continue
        lines.extend([f'### {title} — {_count(len(workloads), "workload")}', ''])
        if multi_sample:
            lines.extend(
                [
                    '| Workload | Median peak RSS (MiB) | Peak range (MiB) | Median setup peak (MiB) |',
                    '|---|---:|---:|---:|',
                ]
            )
        else:
            lines.extend(['| Workload | Peak RSS (MiB) | Setup peak (MiB) |', '|---|---:|---:|'])
        for workload_id, samples in sorted(workloads):
            peaks = [_metric(sample, 'peak_rss_bytes') for sample in samples]
            setups = [_metric(sample, 'setup_peak_rss_bytes') for sample in samples]
            if multi_sample:
                lines.append(
                    f'| `{workload_id}` | {median(peaks):.1f} | '
                    f'{min(peaks):.1f}–{max(peaks):.1f} | {median(setups):.1f} |'
                )
            else:
                lines.append(f'| `{workload_id}` | {peaks[0]:.1f} | {setups[0]:.1f} |')
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path, help='RSS JSON produced by memory.py.')
    args = parser.parse_args()
    print(render_summary(json.loads(args.report.read_text())), end='')


if __name__ == '__main__':
    main()
