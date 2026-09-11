"""Convenient native pytest launcher with thread environment set before imports."""

import subprocess
import sys
from pathlib import Path

from runtime import controlled_environment


def main():
    command = [
        sys.executable,
        '-m',
        'pytest',
        'tests/benchmarks',
        '--benchmark-only',
        '-n',
        '0',
        '--no-cov',
        '--benchmark-columns=median,iqr,mean,stddev,min,max,rounds,iterations',
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.call(command, cwd=Path(__file__).resolve().parents[2], env=controlled_environment()))


if __name__ == '__main__':
    main()
