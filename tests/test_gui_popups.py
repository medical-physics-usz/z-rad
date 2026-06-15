import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PyQt5.QtWidgets import QApplication, QMessageBox  # noqa: E402

from zrad.gui.toolbox_gui import CustomErrorBox, CustomInfoBox, CustomWarningBox  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def captured_window_titles(monkeypatch):
    titles = {}
    set_window_title = QMessageBox.setWindowTitle

    def capture_title(box, title):
        titles[box] = title
        return set_window_title(box, title)

    monkeypatch.setattr(QMessageBox, "setWindowTitle", capture_title)
    return titles


def test_info_popup_uses_information_severity(qapp, captured_window_titles):
    box = CustomInfoBox("Processing finished!")

    assert box.icon() == QMessageBox.Information
    assert captured_window_titles[box] == "Info!"
    assert box.text() == "Processing finished!"


def test_warning_popup_uses_warning_severity(qapp, captured_window_titles):
    box = CustomWarningBox("Select Input Imaging Data Type")

    assert box.icon() == QMessageBox.Warning
    assert captured_window_titles[box] == "Warning!"
    assert box.text() == "Select Input Imaging Data Type"


def test_error_popup_uses_error_severity(qapp, captured_window_titles):
    box = CustomErrorBox("Error reading NIfTI mask")

    assert box.icon() == QMessageBox.Critical
    assert captured_window_titles[box] == "Error!"
    assert box.text() == "Error reading NIfTI mask"
