import PyInstaller.__main__

PyInstaller.__main__.run([
    'main.py',
    '--distpath=exec',
    '--onefile',
    '--add-data=logo.png;.',
    '--add-data=icon.ico;.',
    '--add-data=MainLogo.jpg;.',
    '--hidden-import=pydicom.encoders.gdcm',
    '--hidden-import=pydicom.encoders.pylibjpeg',
])
