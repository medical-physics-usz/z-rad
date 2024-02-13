import sys
from PyQt5.QtGui import QFont
# Import required PyQt5 modules for GUI creation
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QAction
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QStyleFactory

from zrad.gui.filt_tab import FilteringTab
from zrad.gui.prep_tab import PreprocessingTab
from zrad.gui.rad_tab import RadiomicsTab


class Zrad(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize GUI components
        self.tab_one = None
        self.tab_two = None
        self.tab_three = None
        self.tab_widget = None
        self.menubar = None
        self.file_menu = None
        self.load_action = None
        self.save_action = None
        self.exit_action = None

    def add_logo_to_tab(self, tab):
        logo_label = QLabel(tab)
        logo_pixmap = QPixmap('zrad/logo/logo.png')
        desiredWidth = 300  # Set this to your desired width
        desiredHeight = 150  # Set this to your desired height
        logo_pixmap = logo_pixmap.scaled(desiredWidth, desiredHeight, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo_label.setPixmap(logo_pixmap)
        logo_label.setGeometry(1750 - logo_pixmap.width(), 660 - logo_pixmap.height(),
                               logo_pixmap.width(), logo_pixmap.height())

    def init_gui(self):
        """
        Initialize the main GUI components.
        """
        # Set window title and geometry
        self.setWindowTitle('Z-rad-V8')
        self.setGeometry(0, 0, 1800, 750)

        # Create and set the central tab widget
        self.tab_widget = QTabWidget(self)
        self.setCentralWidget(self.tab_widget)

        # Initialize and add tabs
        self.tab_one = PreprocessingTab()
        self.tab_one.init_tab()
        self.tab_two = FilteringTab()
        self.tab_two.init_tab()
        self.tab_three = RadiomicsTab()
        self.tab_three.init_tab()

        # Add tabs to the tab widget
        self.tab_widget.addTab(self.tab_one, "Resampling")
        self.tab_widget.addTab(self.tab_two, "Filtering")
        self.tab_widget.addTab(self.tab_three, "Radiomics")

        self.add_logo_to_tab(self.tab_one)
        self.add_logo_to_tab(self.tab_two)
        self.add_logo_to_tab(self.tab_three)

        # Create the menu bar
        self.create_menu()

        # Connect to tab change signal
        self.tab_widget.currentChanged.connect(self.tab_changed)

        # Set style for the tab widget
        self.tab_widget.setStyleSheet("background-color: #00008B;")

        # Display the main window
        self.show()

    def create_menu(self):
        """
        Create and configure the menu bar.
        """
        # Initialize the menu bar
        self.menubar = self.menuBar()
        self.file_menu = self.menubar.addMenu('File')

        # Create actions for loading and saving
        self.load_action = QAction('Load Input', self)
        self.save_action = QAction('Save Input', self)

        # Add actions to the File menu
        self.file_menu.addAction(self.load_action)
        self.file_menu.addAction(self.save_action)

        self.load_action.setShortcut('Ctrl+O')
        self.save_action.setShortcut('Ctrl+S')

        # Add a separator
        self.file_menu.addSeparator()

        # Create and add the exit action
        self.exit_action = QAction('Exit', self)
        self.exit_action.triggered.connect(self.close)
        self.file_menu.addAction(self.exit_action)

        # Connect actions to the first tab's functions
        self.load_action.triggered.connect(self.tab_one.load_input_data)
        self.save_action.triggered.connect(self.tab_one.save_input_data)

    def tab_changed(self, index):
        """
        Handle the change of tabs.
        """
        # Disconnect previous signals
        self.load_action.triggered.disconnect()
        self.save_action.triggered.disconnect()

        # Connect actions to the appropriate tab functions
        if index == 0:  # Preprocessing Tab
            self.load_action.triggered.connect(self.tab_one.load_input_data)
            self.save_action.triggered.connect(self.tab_one.save_input_data)
        elif index == 1:  # Filtering Tab
            self.load_action.triggered.connect(self.tab_two.load_input_data)
            self.save_action.triggered.connect(self.tab_two.save_input_data)
        elif index == 2:  # Radiomics Tab
            self.load_action.triggered.connect(self.tab_three.load_input_data)
            self.save_action.triggered.connect(self.tab_three.save_input_data)

        # Update action text
        self.load_action.setText('Load Input')
        self.save_action.setText('Save Input')


if __name__ == '__main__':
    # Initialize and configure the application
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))

    # Set application palette for styling
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.ButtonText, Qt.white)
    app.setPalette(palette)

    app.setFont(QFont('Arial', 10))

    # Create and display the main window
    ex = Zrad()
    ex.init_gui()
    sys.exit(app.exec_())