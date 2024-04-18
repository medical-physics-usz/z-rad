import PyInstaller.__main__

PyInstaller.__main__.run([
    'main.py',
    '--distpath=exec',
    '--onefile',
    #'--add-data=config.json;.',
    '--add-data=logo.png;.',
    '--hidden-import=pydicom.encoders.gdcm',
    '--hidden-import=pydicom.encoders.pylibjpeg',
])