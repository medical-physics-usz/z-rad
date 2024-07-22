import os
import sys
from multiprocessing import cpu_count

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import QLineEdit, QLabel, QPushButton, QComboBox, QCheckBox, QMessageBox, QWidget


def data_io(parent):
    # Load Directory Button and Label
    parent.load_dir_button = CustomButton(
        'Input Directory',
        20, 50, 160, 50, parent,
        style=True)
    parent.load_dir_label = CustomTextField(
        '',
        200, 50, 1000, 50,
        parent,
        style=True)
    parent.load_dir_label.setAlignment(Qt.AlignCenter)
    parent.load_dir_button.clicked.connect(lambda: parent.open_directory(key=True))

    parent.input_imaging_mod_combo_box = CustomBox(
        200, 140, 160, 50, parent,
        item_list=[
            "Imaging Modality:", "CT", "MR", "PT"
        ]
    )

    #  Start and Stop Folder TextFields and Labels
    parent.start_folder_label = CustomLabel(
        'Start Folder:',
        400, 140, 100, 50, parent,
        style="color: white;"
    )
    parent.start_folder_text_field = CustomTextField(
        "",
        500, 140, 60, 50, parent
    )

    for pos_x in [570, 770, 1080]:

        CustomInfo(
            ' i',
            'Use Start and Stop folders if only all the folders in the directory are integers.'
                    '\nUsers can specify the folders of interest in the List of Folders (e.g. 1, 2a, 3_1), '
                    'even if folder names are non-integer.'
                    '\nKeep Start, Stop, and List of Folders empty to run ALL folders in the Input Directory, '
                    'even if folder names are non-integer.',
            pos_x, 140, 14, 14, parent
        )

    parent.stop_folder_label = CustomLabel(
        'Stop Folder:',
        600, 140, 100, 50, parent,
        style="color: white;")
    parent.stop_folder_text_field = CustomTextField(
        "",
        700, 140, 60, 50, parent
    )
    # List of Patient Folders TextField and Label
    parent.list_of_patient_folders_label = CustomLabel(
        'List of Folders:',
        800, 140, 110, 50, parent,
        style="color: white;"
    )
    parent.list_of_patient_folders_text_field = CustomTextField(
        "E.g. 1, 5, 10, 34...",
        920, 140, 150, 50, parent)

    # Number of Threads ComboBox
    no_of_threads = ['Threads:']
    for core in range(cpu_count()):
        if core == 0:
            no_of_threads.append(str(core + 1) + " thread")
        else:
            no_of_threads.append(str(core + 1) + " threads")
    parent.number_of_threads_combo_box = CustomBox(
        1100, 140, 100, 50, parent,
        item_list=no_of_threads
    )

    # Save Directory Button and Label
    parent.save_dir_button = CustomButton(
        'Output Directory',
        20, 220, 160, 50, parent,
        style=True)
    parent.save_dir_label = CustomTextField(
        '',
        200, 220, 1000, 50,
        parent,
        style=True)
    parent.save_dir_label.setAlignment(Qt.AlignCenter)
    parent.save_dir_button.clicked.connect(lambda: parent.open_directory(key=False))


class CustomButton(QPushButton):

    def __init__(self, text: str, pos_x: int, pos_y: int, size_x: int, size_y: int, parent: QWidget, style: bool = True):
        super().__init__(text, parent)
        self.setGeometry(pos_x, pos_y, size_x, size_y)
        if style:
            self.setStyleSheet("QPushButton {"
                               "background-color: #4CAF50; "
                               "color: white; "
                               "border: none; "
                               "border-radius: 25px;"
                               "}"
                               "QPushButton:hover {"
                               "background-color: green;"
                               "}"
                               )


class CustomBox(QComboBox):

    def __init__(self, pos_x: int, pos_y: int, size_x: int, size_y: int, parent: QWidget, item_list: list):
        super().__init__(parent)
        for item in item_list:
            self.addItem(item)
        self.setGeometry(pos_x, pos_y, size_x, size_y)
        self.setStyleSheet("QComboBox:hover {"
                           "background-color: #27408B;"
                           "color: yellow; "
                           "}"
                           )


class CustomLabel(QLabel):

    def __init__(self, text: str, pos_x: int, pos_y: int, size_x: int, size_y: int, parent: QWidget,
                 style: str = "background-color: white; color: black; border: none; border-radius: 25px;"):
        super().__init__(text, parent)
        self.setGeometry(pos_x, pos_y, size_x, size_y)
        self.setStyleSheet(style)


class CustomInfo(QLabel):

    def __init__(self, text: str, info_tip: str, pos_x: int, pos_y: int, size_x: int, size_y: int, parent: QWidget):
        super().__init__(text, parent)
        self.setGeometry(pos_x, pos_y, size_x, size_y)
        self.setStyleSheet("background-color: white; color: black; border: none; border-radius: 7px;")
        self.setToolTip(info_tip)


class CustomEditLine(QLineEdit):

    def __init__(self, text: str, pos_x: int, pos_y: int, size_x: int, size_y: int, parent: QWidget):
        super().__init__(parent)
        self.setPlaceholderText(text)
        self.setGeometry(pos_x, pos_y, size_x, size_y)
        self.setStyleSheet("background-color: white; color: black;")


class CustomTextField(QLineEdit):

    def __init__(self, text: str, pos_x: int, pos_y: int, size_x: int, size_y: int, parent: QWidget, style: bool = False):
        super().__init__(parent)
        self.setPlaceholderText(text)
        self.setGeometry(pos_x, pos_y, size_x, size_y)
        if style:
            self.setStyleSheet("background-color: white; color: black; border: none; border-radius: 25px;")
        else:
            self.setStyleSheet("background-color: white; color: black;")


class CustomCheckBox(QCheckBox):

    def __init__(self, text: str, pos_x: int, pos_y: int, size_x: int, size_y: int, parent: QWidget):
        super().__init__(text, parent)
        self.setGeometry(pos_x, pos_y, size_x, size_y)
        self.setStyleSheet("""
            QCheckBox::indicator {
                width: 25px;
                height: 25px;
                border: 1px solid black;
                background-color: rgba(255, 255, 255, 128); /* Semi-transparent white background */
            }
            QCheckBox::indicator:checked {
                border: 1px solid black;
                background-color: rgba(144, 238, 144, 255); /* Semi-transparent white background */
            }
        """)


class CustomWarningBox(QMessageBox):

    def __init__(self, text: str, warning: bool = True):
        super().__init__()
        self.warning_key = warning
        self.setup_message_box(text)

    def setup_message_box(self, text: str):
        if self.warning_key:
            self.setIcon(QMessageBox.Warning)
            self.setWindowTitle('Warning!')
            self.setText(text)
            self.setStandardButtons(QMessageBox.Retry)
        else:
            self.setIcon(QMessageBox.Information)
            self.setWindowTitle('Help & Support')
            self.setText(text)
            self.setStandardButtons(QMessageBox.Close)

        self.setStyleSheet("QPushButton {"
                           "background-color: #FFD700;"
                           "color: black;"
                           "border-style: solid;"
                           "border-width: 2px;"
                           "border-radius: 10px;"
                           "border-color: #606060;"
                           "font: bold 16px;"
                           "padding: 10px;"
                           "}"
                           "QPushButton:hover {"
                           "background-color: #FF8C00;"
                           "}"
                           "QPushButton:pressed {"
                           "background-color: #505050;"
                           "}"
                           )
        font = QFont('Verdana')
        self.setFont(font)

    def response(self) -> bool:
        get_response = self.exec_()
        return get_response == QMessageBox.Retry


def resource_path(relative_path: str) -> str:
    try:
        base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def add_logo_to_tab(tab: QWidget):
    logo_label = QLabel(tab)
    logo_pixmap = QPixmap(resource_path('logo.png'))
    desired_width = 300
    desired_height = 150
    logo_pixmap = logo_pixmap.scaled(desired_width, desired_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    logo_label.setPixmap(logo_pixmap)
    logo_label.setGeometry(20, 620 - logo_pixmap.height(),
                           logo_pixmap.width(), logo_pixmap.height())
