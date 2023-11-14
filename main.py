import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QAction)
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QStyleFactory
from PyQt5.QtGui import QPalette, QColor

from gui.prep_tab import PreprocessingTab

class ZRAD(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Configure the window
        self.setWindowTitle('Z-RAD')
        self.setGeometry(0, 0, 1800, 750)


        # Create a tab widget
        self.tab_widget = QTabWidget(self)
        self.setCentralWidget(self.tab_widget)

        # Create the tabs
        self.tab_one = PreprocessingTab()
        self.tab_one.initUI()

        # Add the tabs to the tab widget
        self.tab_widget.addTab(self.tab_one, "Resampling")

        self.createMenu()  # Initialize the menu

        # Connect to the signal for tab changes
        self.tab_widget.currentChanged.connect(self.onTabChanged)

        self.tab_widget.setStyleSheet("background-color: #00008B;")  # Set background

        # Show the window
        self.show()

    def createMenu(self):
        self.menubar = self.menuBar()
        self.fileMenu = self.menubar.addMenu('File')

        # Load and Save Settings Actions
        self.loadAction = QAction('Load Input', self)
        self.saveAction = QAction('Save Input', self)

        # Add actions to the menu
        self.fileMenu.addAction(self.loadAction)
        self.fileMenu.addAction(self.saveAction)

        # Separator
        self.fileMenu.addSeparator()

        # Exit Action
        exitAction = QAction('Exit', self)
        exitAction.triggered.connect(self.close)
        self.fileMenu.addAction(exitAction)

        # Initially connect to first tab's functions
        self.loadAction.triggered.connect(self.tab_one.load_input_data)
        self.saveAction.triggered.connect(self.tab_one.save_input_data)

    def onTabChanged(self, index):
        # Disconnect the previous signals to avoid overlap
        self.loadAction.triggered.disconnect()
        self.saveAction.triggered.disconnect()

        # Depending on the index of the selected tab, connect to the appropriate functions
        if index == 0:  # Preprocessing Tab
            self.loadAction.triggered.connect(self.tab_one.load_input_data)
            self.saveAction.triggered.connect(self.tab_one.save_input_data)

        # Update action text based on the current tab
        current_tab_name = self.tab_widget.tabText(index)
        self.loadAction.setText('Load Input')
        self.saveAction.setText('Save Input')

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Set the style and palette
    app.setStyle(QStyleFactory.create('Fusion'))
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.ButtonText, Qt.white)
    app.setPalette(palette)

    ex = ZRAD()
    sys.exit(app.exec_())
