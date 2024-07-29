import PyInstaller.__main__

PyInstaller.__main__.run([
    'main.py',
    '--onefile',
    '--icon=doc\\logos\\icon.ico',
    '--add-data=doc\\logos\\icon.ico;doc\\logos',
    '--add-data=doc\\logos\\USZLogo.png;doc\\logos',
    '--add-data=doc\\logos\\ZRadLogo.jpg;doc\\logos',
    '--hidden-import=pydicom.encoders.gdcm',
    '--hidden-import=pydicom.encoders.pylibjpeg',
    '--log-level=DEBUG',
])
