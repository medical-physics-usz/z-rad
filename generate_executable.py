import sys
from pathlib import Path

import pydicom

HIDDEN_IMPORTS = [
    'pydicom.pixels.decoders.gdcm',
    'pydicom.pixels.decoders.pylibjpeg',
    'pydicom.pixels.decoders.pillow',
    'pydicom.pixels.decoders.pyjpegls',
    'pydicom.pixels.decoders.rle',
    'pydicom.pixels.encoders.gdcm',
    'pydicom.pixels.encoders.pylibjpeg',
    'pydicom.pixels.encoders.native',
    'pydicom.pixels.encoders.pyjpegls',
]

LOGO_DATA_FILES = [
    ('docs/logos/icon.ico', 'docs/logos'),
    ('docs/logos/USZLogo.png', 'docs/logos'),
    ('docs/logos/ZRadLogo.jpg', 'docs/logos'),
]

PROJECT_ROOT = Path(__file__).resolve().parent


def _add_data_arg(source: str | Path, destination: str, separator: str) -> str:
    return f'--add-data={source}{separator}{destination}'


def build_pyinstaller_args(platform: str = sys.platform) -> list[str]:
    """Build PyInstaller arguments for the current release target."""
    is_windows = platform.startswith('win')
    is_macos = platform == 'darwin'
    add_data_sep = ';' if is_windows else ':'
    pydicom_data_dir = Path(pydicom.__file__).parent / 'data'

    args = [
        'main.py',
        '--clean',
        '--noconfirm',
        '--log-level=DEBUG',
        '--specpath=build/pyinstaller',
    ]

    if is_macos:
        args.extend(
            [
                '--windowed',
                '--name=Z-Rad',
                f'--icon={PROJECT_ROOT / "docs/logos/icon.icns"}',
                '--osx-bundle-identifier=ch.usz.medphys.zrad',
            ]
        )
    else:
        args.extend(
            [
                '--onefile',
                '--name=z-rad',
                f'--icon={PROJECT_ROOT / "docs/logos/icon.ico"}',
            ]
        )

    args.extend(
        _add_data_arg(PROJECT_ROOT / source, destination, add_data_sep) for source, destination in LOGO_DATA_FILES
    )
    args.append(_add_data_arg(pydicom_data_dir, 'pydicom/data', add_data_sep))
    args.extend(f'--hidden-import={hidden_import}' for hidden_import in HIDDEN_IMPORTS)
    return args


def main() -> None:
    import PyInstaller.__main__

    pydicom_data_dir = Path(pydicom.__file__).parent / 'data'
    print(f'Pydicom data directory: {pydicom_data_dir}')
    PyInstaller.__main__.run(build_pyinstaller_args())


if __name__ == '__main__':
    main()
