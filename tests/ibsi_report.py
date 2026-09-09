"""Summarize collected IBSI benchmark results from pytest JUnit XML.

Usage: python tests/ibsi_report.py reports/integration.xml reports/ibsi.md
The report describes executed cases, not certification or untested configurations.
"""

import hashlib
import importlib.metadata
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


def build_report(xml_path):
    root = ET.parse(xml_path).getroot()
    cases = []
    supplemental = []
    for case in root.iter('testcase'):
        name = case.attrib['name']
        is_supplemental = name.startswith('test_ibsi_supplemental_')
        if not is_supplemental and not name.startswith(
            (
                'test_ibsi_i_config_',
                'test_ibsi_i_digital_phantom',
                'test_ibsi_i_diagnostics',
                'test_ibsi_ii_ph_i_',
                'test_ibsi_ii_ph_ii_',
                'test_official_ibsi_suv_dro',
            )
        ):
            continue
        status = 'PASS'
        for element, label in [('skipped', 'SKIP'), ('failure', 'FAIL'), ('error', 'ERROR')]:
            if case.find(element) is not None:
                status = label
        (supplemental if is_supplemental else cases).append((name, status))
    if not cases and not supplemental:
        raise ValueError('No IBSI benchmark cases found in JUnit report')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    lines = [
        '# IBSI benchmark execution report',
        '',
        f'Generated: {datetime.now(timezone.utc).isoformat()}',
        f'Report-generation revision: {revision}' + (' (working tree changes present)' if dirty else ''),
        f'Report-generation environment: Python {platform.python_version()}, {platform.platform()}',
        'Revision, dependency versions and reference fingerprints below describe report generation. '
        'Generate immediately after testing in the same checkout and environment; they do not authenticate '
        'the provenance of an older or imported JUnit file.',
        '',
        'Scope: cases recorded in the supplied JUnit report. PASS does not establish agreement for excluded '
        'features or configurations without references. See docs/ibsi/index.rst for coverage limits.',
        '',
        '| Benchmark case | Result |',
        '|---|---|',
    ]
    for name, status in sorted(cases):
        lines.append(f'| `{name}` | {status} |')
    counts = {status: sum(s == status for _, s in cases) for status in ('PASS', 'FAIL', 'ERROR', 'SKIP')}
    lines += ['', ', '.join(f'{count} {status}' for status, count in counts.items()), '']
    lines += [
        '## Supplemental asset and filter checks',
        '',
        'These checks exercise loading, geometry, and mathematical filter properties. '
        'They are not published IBSI consensus comparisons and are excluded from '
        'the benchmark totals above.',
        '',
        '| Supplemental case | Result |',
        '|---|---|',
    ]
    for name, status in sorted(supplemental):
        lines.append(f'| `{name}` | {status} |')
    supplemental_counts = {
        status: sum(s == status for _, s in supplemental) for status in ('PASS', 'FAIL', 'ERROR', 'SKIP')
    }
    lines += [
        '',
        'Supplemental: ' + ', '.join(f'{count} {status}' for status, count in supplemental_counts.items()),
        '',
    ]
    lines += ['## Dependency versions', '']
    for package in ('numpy', 'scipy', 'SimpleITK', 'pydicom', 'PyWavelets'):
        lines.append(f'- {package}: {importlib.metadata.version(package)}')
    lines += ['', '## Reference fingerprints (SHA-256)', '']
    data_dir = Path(__file__).resolve().parents[1] / 'tests' / 'data'
    paths = sorted((data_dir / 'ibsi_1_reference_data').glob('*.csv'))
    paths += [
        data_dir / relative
        for relative in (
            'ibsi_2_reference_data/reference_feature_values/reference_values.csv',
            'ibsi_ct_radiomics_phantom.zip',
            'ibsi_1_digital_phantom.zip',
            'ibsi_2_digital_phantom.zip',
            'ibsi_2_reference_data/ibsi_2_response_maps.zip',
            'IBSI_SUV.zip',
        )
    ]
    for path in paths:
        lines.append(f'- `{path.relative_to(data_dir)}`: `{hashlib.sha256(path.read_bytes()).hexdigest()}`')
    lines.append('')
    return '\n'.join(lines)


if __name__ == '__main__':
    destination = Path(sys.argv[2])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(build_report(sys.argv[1]))
