import PyInstaller.__main__

PyInstaller.__main__.run([
    'main_texture.py',
    '--onefile',
    '--add-data=config.txt;.',
    '--add-data=LogoUSZ.png;.',
    '--add-data=feature_names_2D.txt;.',
    '--hidden-import=vtkmodules',
    '--hidden-import=vtkmodules.all',
])
