from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QLineEdit, QLabel, QPushButton, QComboBox, QCheckBox, QMessageBox


class CustomButton(QPushButton):

    def __init__(self, text, font, pos_x, pos_y, size_x, size_y, parent, style=True):
        super().__init__(text, parent)
        self.setFont(QFont('Arial', font))
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
        self.setFont(QFont('Arial', font))
        self.setStyleSheet("QComboBox:hover {"
                           "background-color: #27408B;"
                           "color: yellow; "
                           "}"
                           )


class CustomLabel(QLabel):

    def __init__(self, text, font, pos_x, pos_y, size_x, size_y, parent,
                 style="background-color: white; color: black; border: none; border-radius: 25px;"):
        super().__init__(text, parent)
        self.setFont(QFont('Arial', font))
        self.setGeometry(pos_x, pos_y, size_x, size_y)
        self.setStyleSheet(style)


class CustomEditLine(QLineEdit):

    def __init__(self, text, font, pos_x, pos_y, size_x, size_y, parent):
        super().__init__(parent)
        self.setPlaceholderText(text)
        self.setGeometry(pos_x, pos_y, size_x, size_y)
        self.setStyleSheet(
            "background-color: white; color: black;")
        self.PrefixTextField.setFont(QFont('Arial', font))


class CustomTextField(QLineEdit):

    def __init__(self, text, font, pos_x, pos_y, size_x, size_y, parent):
        super().__init__(parent)
        self.setPlaceholderText(text)
        self.setGeometry(pos_x, pos_y, size_x, size_y)
        self.setStyleSheet(
            "background-color: white; color: black;")
        self.setFont(QFont('Arial', font))


class CustomCheckBox(QCheckBox):

    def __init__(self, text, font, pos_x, pos_y, size_x, size_y, parent):
        super().__init__(text, parent)
        self.setGeometry(pos_x, pos_y, size_x, size_y)
        self.setFont(QFont('Arial', font))
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
    def __init__(self, text):
        super().__init__()
        self.setIcon(QMessageBox.Warning)
        self.setWindowTitle('Warning!')
        self.setText(text)
        self.setStandardButtons(QMessageBox.Retry)

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
        font.setPointSize(14)
        self.setFont(font)

    def response(self):
        get_response = self.exec_()
        return get_response == QMessageBox.Retry
