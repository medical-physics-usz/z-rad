import PyInstaller.__main__

PyInstaller.__main__.run([
    'main.py',
    '--onefile',
    '--icon=documentation\\logos\\icon.ico',
    '--add-data=documentation\\logos\\icon.ico;documentation\\logos',
    '--add-data=documentation\\logos\\USZLogo.png;documentation\\logos',
    '--add-data=documentation\\logos\\ZRadLogo.jpg;documentation\\logos',
    '--hidden-import=pydicom.encoders.gdcm',
    '--hidden-import=pydicom.encoders.pylibjpeg',
    '--log-level=DEBUG',
])
