from PyQt5.QtWidgets import (QLineEdit, QLabel, QPushButton, QComboBox, QCheckBox)
from PyQt5.QtGui import QFont


class CustomButton(QPushButton):

    def __init__(self, text, font, pos_x, pos_y, size_x, size_y, parent, style):
        super().__init__(text, parent)
        self.setFont(QFont('SansSerif', font))
        self.setGeometry(pos_x, pos_y, size_x, size_y)
        if style:
            self.setStyleSheet(style)


class CustomBox(QComboBox):

    def __init__(self, font, pos_x, pos_y, size_x, size_y, parent, item_list):
        super().__init__(parent)
        for item in item_list:
            self.addItem(item)
        self.setGeometry(pos_x, pos_y, size_x, size_y)
        self.setFont(QFont('SansSerif', font))


class CustomLabel(QLabel):

    def __init__(self, text, font, pos_x, pos_y, size_x, size_y, parent,
                 style="background-color: white; color: black; border: none; border-radius: 25px;"):
        super().__init__(text, parent)
        self.setFont(QFont('SansSerif', font))
        self.setGeometry(pos_x, pos_y, size_x, size_y)
        self.setStyleSheet(style)


class CustomEditLine(QLineEdit):

    def __init__(self, text, font, pos_x, pos_y, size_x, size_y, parent):
        super().__init__(parent)
        self.setPlaceholderText(text)
        self.setGeometry(pos_x, pos_y, size_x, size_y)
        self.setStyleSheet(
            "background-color: white; color: black;")  # Set background color to white and text color to black
        self.PrefixTextField.setFont(QFont('SansSerif', font))


class CustomTextField(QLineEdit):

    def __init__(self, text, font, pos_x, pos_y, size_x, size_y, parent):
        super().__init__(parent)
        self.setPlaceholderText(text)
        self.setGeometry(pos_x, pos_y, size_x, size_y)
        self.setStyleSheet(
            "background-color: white; color: black;")  # Set background color to white and text color to black
        self.setFont(QFont('SansSerif', font))


class CustomCheckBox(QCheckBox):

    def __init__(self, text, font, pos_x, pos_y, size_x, size_y, parent):
        super().__init__(text, parent)
        self.setGeometry(pos_x, pos_y, size_x, size_y)
        self.setFont(QFont('SansSerif', font))
        self.setStyleSheet("""
                    QCheckBox::indicator {
                        width: 25px;
                        height: 25px;
                        border: 1px solid black;
                        background-color: rgba(255, 255, 255, 128); /* Semi-transparent white background */
                    }
                    QCheckBox::indicator:checked {
                        image: url(graphics/blue-check-mark.png);
                        border: 1px solid black;
                        background-color: rgba(144, 238, 144, 255); /* Semi-transparent white background */
                    }
                """)
