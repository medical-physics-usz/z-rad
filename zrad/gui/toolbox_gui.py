import os
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtWidgets import QLineEdit, QLabel, QPushButton, QComboBox, QCheckBox, QMessageBox


def adjust_fonts():
    screen = QGuiApplication.primaryScreen()
    height = screen.size().height()
    font_size = int(height * 0.013)
    return font_size

class CustomButton(QPushButton):

    def __init__(self, text, font, pos_x, pos_y, size_x, size_y, parent, style=True, run=False):
        super().__init__(text, parent)
        if run:
            self.setFont(QFont('Arial', 20))
        else:
            self.setFont(QFont('Arial', adjust_fonts()))
        self.setGeometry(pos_x, pos_y, size_x, size_y)
        if style:
            self.setStyleSheet("QPushButton {" 
                               "background-color: #4CAF50; "
                               "color: white; "
                               "border: none; "
                               "border-radius: 25px;"
                               "}"
                               "QPushButton:hover {"
                               "  background-color: green;"
                               "}"
                               )


class CustomBox(QComboBox):

    def __init__(self, font, pos_x, pos_y, size_x, size_y, parent, item_list):
        super().__init__(parent)
        for item in item_list:
            self.addItem(item)
        self.setGeometry(pos_x, pos_y, size_x, size_y)
        self.setFont(QFont('Arial', adjust_fonts()))
        self.setStyleSheet("QComboBox:hover {"
                           "background-color: #27408B;"
                           "color: yellow; "
                           "}"
                           )


class CustomLabel(QLabel):

    def __init__(self, text, font, pos_x, pos_y, size_x, size_y, parent,
                 style="background-color: white; color: black; border: none; border-radius: 25px;"):
        super().__init__(text, parent)
        self.setFont(QFont('Arial', adjust_fonts()+1))
        self.setGeometry(pos_x, pos_y, size_x, size_y)
        self.setStyleSheet(style)


class CustomEditLine(QLineEdit):

    def __init__(self, text, font, pos_x, pos_y, size_x, size_y, parent):
        super().__init__(parent)
        self.setPlaceholderText(text)
        self.setGeometry(pos_x, pos_y, size_x, size_y)
        self.setStyleSheet(
            "background-color: white; color: black;")
        self.PrefixTextField.setFont(QFont('Arial', adjust_fonts()))


class CustomTextField(QLineEdit):

    def __init__(self, text, font, pos_x, pos_y, size_x, size_y, parent, style=False):
        super().__init__(parent)
        self.setPlaceholderText(text)
        self.setGeometry(pos_x, pos_y, size_x, size_y)
        if style:
            self.setStyleSheet("background-color: white; color: black; border: none; border-radius: 25px;")
        else:
            self.setStyleSheet(
                "background-color: white; color: black;")
        self.setFont(QFont('Arial', adjust_fonts()))


class CustomCheckBox(QCheckBox):

    def __init__(self, text, font, pos_x, pos_y, size_x, size_y, parent):
        super().__init__(text, parent)
        self.setGeometry(pos_x, pos_y, size_x, size_y)
        self.setFont(QFont('Arial', adjust_fonts()+2))
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
    def __init__(self, text, warning=True):
        super().__init__()
        self.warning_key = warning
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
                           "  background-color: #FFD700;"  
                           "  color: black;"
                           "  border-style: solid;"
                           "  border-width: 2px;"
                           "  border-radius: 10px;"
                           "  border-color: #606060;"
                           "  font: bold 16px;"
                           "  padding: 10px;"
                           "}"
                           "QPushButton:hover {"
                           "  background-color: #FF8C00;"
                           "}"
                           "QPushButton:pressed {"
                           "  background-color: #505050;"
                           "}"
                           )
        font = QFont('Arial')
        font.setPointSize(adjust_fonts())
        self.setFont(font)

    def response(self):
        get_response = self.exec_()
        return get_response == QMessageBox.Retry


def resource_path(relative_path):
    try:
        base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def add_logo_to_tab(tab):
    logo_label = QLabel(tab)
    logo_pixmap = QPixmap(resource_path('logo.png'))
    desired_width = 300
    desired_height = 150
    logo_pixmap = logo_pixmap.scaled(desired_width, desired_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    logo_label.setPixmap(logo_pixmap)
    logo_label.setGeometry(1750 - logo_pixmap.width(), 660 - logo_pixmap.height(),
                           logo_pixmap.width(), logo_pixmap.height())
