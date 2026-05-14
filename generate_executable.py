import os
import sys

import PyInstaller.__main__
import pydicom

# Detect platform
is_windows = sys.platform.startswith('win')
is_mac = sys.platform.startswith('darwin')

# Choose icon format based on OS
icon_path = 'docs/logos/icon.icns' if is_mac else 'docs/logos/icon.ico'

# Choose add-data separator based on OS
add_data_sep = ';' if is_windows else ':'

# Get the path to the pydicom data directory
pydicom_data_dir = os.path.join(os.path.dirname(pydicom.__file__), 'data')

# Print pydicom data directory for verification
print(f"Pydicom data directory: {pydicom_data_dir}")

PyInstaller.__main__.run([
    'main.py',
    '--onefile',
    f'--icon={icon_path}',
    f'--add-data=docs/logos/icon.ico{add_data_sep}docs/logos',
    f'--add-data=docs/logos/USZLogo.png{add_data_sep}docs/logos',
    f'--add-data=docs/logos/ZRadLogo.jpg{add_data_sep}docs/logos',
    f'--add-data={pydicom_data_dir}{add_data_sep}pydicom/data',
    '--hidden-import=pydicom.pixels.decoders.gdcm',
    '--hidden-import=pydicom.pixels.decoders.pylibjpeg',
    '--hidden-import=pydicom.pixels.decoders.pillow',
    '--hidden-import=pydicom.pixels.decoders.pyjpegls',
    '--hidden-import=pydicom.pixels.decoders.rle',
    '--hidden-import=pydicom.pixels.encoders.gdcm',
    '--hidden-import=pydicom.pixels.encoders.pylibjpeg',
    '--hidden-import=pydicom.pixels.encoders.native',
    '--hidden-import=pydicom.pixels.encoders.pyjpegls',
    '--log-level=DEBUG',
    '--clean',
])
