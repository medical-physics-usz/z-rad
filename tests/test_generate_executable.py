import importlib.util
from pathlib import Path


def _load_generate_executable_module():
    module_path = Path(__file__).resolve().parents[1] / 'generate_executable.py'
    spec = importlib.util.spec_from_file_location('generate_executable', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generate_executable = _load_generate_executable_module()


def test_windows_pyinstaller_args_preserve_onefile_executable():
    args = generate_executable.build_pyinstaller_args('win32')

    project_root = Path(__file__).resolve().parents[1]

    assert 'main.py' in args
    assert '--onefile' in args
    assert '--windowed' not in args
    assert '--name=z-rad' in args
    assert f'--icon={project_root / "docs/logos/icon.ico"}' in args
    assert '--noconfirm' in args
    assert '--specpath=build/pyinstaller' in args
    assert f'--add-data={project_root / "docs/logos/icon.ico"};docs/logos' in args
    assert any(arg.startswith('--add-data=') and arg.endswith(';pydicom/data') for arg in args)


def test_macos_pyinstaller_args_create_app_bundle():
    args = generate_executable.build_pyinstaller_args('darwin')

    project_root = Path(__file__).resolve().parents[1]

    assert 'main.py' in args
    assert '--windowed' in args
    assert '--onefile' not in args
    assert '--name=Z-Rad' in args
    assert f'--icon={project_root / "docs/logos/icon.icns"}' in args
    assert '--osx-bundle-identifier=ch.usz.medphys.zrad' in args
    assert '--noconfirm' in args
    assert '--specpath=build/pyinstaller' in args
    assert f'--add-data={project_root / "docs/logos/icon.ico"}:docs/logos' in args
    assert any(arg.startswith('--add-data=') and arg.endswith(':pydicom/data') for arg in args)
