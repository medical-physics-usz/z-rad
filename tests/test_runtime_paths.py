import sys
from pathlib import Path

from zrad import toolbox_logic


def test_runtime_data_dir_uses_cwd_when_not_frozen(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delattr(sys, "frozen", raising=False)

    assert toolbox_logic.get_runtime_data_dir() == tmp_path
    assert toolbox_logic.get_config_path() == tmp_path / "config.json"
    assert toolbox_logic.get_logs_dir() == tmp_path / "logs"


def test_runtime_data_dir_uses_macos_app_support_when_frozen(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    app_support_dir = tmp_path / "Library" / "Application Support" / "Z-Rad"

    assert toolbox_logic.get_runtime_data_dir() == app_support_dir
    assert toolbox_logic.get_config_path() == app_support_dir / "config.json"
    assert toolbox_logic.get_logs_dir() == app_support_dir / "logs"


def test_logger_creates_log_file_under_frozen_macos_app_support(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    toolbox_logic.close_all_loggers()

    try:
        logger = toolbox_logic.get_logger("runtime-path-test")
        logger.info("log path smoke test")
    finally:
        toolbox_logic.close_all_loggers()

    log_file = tmp_path / "Library" / "Application Support" / "Z-Rad" / "logs" / "runtime-path-test.log"
    assert log_file.is_file()
