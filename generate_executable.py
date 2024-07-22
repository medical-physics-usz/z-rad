import PyInstaller.__main__

PyInstaller.__main__.run([
    'main.py',
    '--distpath=exec',
    '--onefile',
    '--add-data=USZLogo.png;.',
    '--add-data=icon.ico;.',
    '--add-data=ZRadLogo.jpg;.',
    '--hidden-import=pydicom.encoders.gdcm',
    '--hidden-import=pydicom.encoders.pylibjpeg',
])
