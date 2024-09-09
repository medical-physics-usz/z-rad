from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QLineEdit, QLabel, QPushButton, QComboBox, QCheckBox, QMessageBox, QWidget


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
            self.setStandardButtons(QMessageBox.Ok)
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
        return get_response == QMessageBox.Ok


class CustomInfoBox(QMessageBox):

    def __init__(self, text: str, info: bool = True):
        super().__init__()
        self.info_key = info
        self.setup_message_box(text)

    def setup_message_box(self, text: str):
        if self.info_key:
            self.setIcon(QMessageBox.Warning)
            self.setWindowTitle('Info!')
            self.setText(text)
            self.setStandardButtons(QMessageBox.Ok)
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
        return get_response == QMessageBox.Ok
