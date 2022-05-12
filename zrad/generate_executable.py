import PyInstaller.__main__

PyInstaller.__main__.run([
    'main_texture.py',
    '--distpath=exec',
    '--onefile',
    '--add-data=config.txt;.',
    '--add-data=LogoUSZ.png;.',
    '--add-data=feature_names_2D.txt;.',
    '--hidden-import=vtkmodules',
    '--hidden-import=vtkmodules.all',
    '--hidden-import=sklearn.utils._typedefs',
    '--hidden-import=sklearn.neighbors._partition_nodes',
    '--hidden-import=pydicom.encoders.gdcm',
    '--hidden-import=pydicom.encoders.pylibjpeg',
])
