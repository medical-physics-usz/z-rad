import os
import sys
from multiprocessing import freeze_support

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QAction, QStyleFactory, QScrollArea, QWidget, \
    QVBoxLayout, QSplashScreen, QLabel

from zrad import __version__
from zrad.gui.filt_tab import FilteringTab
from zrad.gui.prep_tab import PreprocessingTab
from zrad.gui.rad_tab import RadiomicsTab
from zrad.gui.toolbox_gui import CustomWarningBox

WINDOW_TITLE = f"Z-Rad v{__version__}"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
BACKGROUND_COLOR = "#005ea8"
FONT_FAMILY = 'Verdana'
FONT_SIZE = 14


def resource_path(relative_path: str) -> str:
    try:
        base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def add_logo_to_tab(tab: QWidget):
    logo_label = QLabel(tab)
    logo_pixmap = QPixmap(resource_path('docs/logos/USZLogo.png'))
    desired_width = 300
    desired_height = 150
    logo_pixmap = logo_pixmap.scaled(desired_width, desired_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    logo_label.setPixmap(logo_pixmap)
    logo_label.setGeometry(20, 620 - logo_pixmap.height(),
                           logo_pixmap.width(), logo_pixmap.height())


class ZRad(QMainWindow):
    def __init__(self):
        super().__init__()
        self.tab_widget = None
        self.tabs = None
        self.menubar = None
        self.file_menu = None
        self.help_menu = None
        self.load_action = None
        self.save_action = None
        self.help_action = None
        self.exit_action = None
        # self.help_documentation_action = None
        self.help_git_action = None
        self.help_contact_action = None
        self.help_about_action = None

        self.init_gui()

    def init_gui(self):
        """
        Initialize the main GUI components.
        """
        self.setWindowTitle(WINDOW_TITLE)
        self.setGeometry(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)

        self.init_tabs()
        self.create_menu()
        self.setWindowIcon(QIcon(resource_path("docs/logos/icon.ico")))

    def init_tabs(self):
        """
        Initialize and add tabs to the main window.
        """
        self.tab_widget = QTabWidget(self)
        self.setCentralWidget(self.tab_widget)

        self.tabs = [
            ("Preprocessing", PreprocessingTab()),
            ("Filtering", FilteringTab()),
            ("Radiomics", RadiomicsTab())
        ]

        for title, tab in self.tabs:
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            add_logo_to_tab(tab)
            scroll_area.setWidget(tab)
            scrollable_widget = QWidget()
            scrollable_layout = QVBoxLayout()
            scrollable_layout.addWidget(scroll_area)
            scrollable_widget.setLayout(scrollable_layout)
            self.tab_widget.addTab(scrollable_widget, title)

        self.tab_widget.currentChanged.connect(self.tab_changed)
        self.tab_widget.setStyleSheet(f"background-color: {BACKGROUND_COLOR};")

    def create_menu(self):
        """
        Create and configure the menu bar.
        """
        self.menubar = self.menuBar()
        self.file_menu = self.menubar.addMenu('File')
        self.help_menu = self.menubar.addMenu('Help')

        self.load_action = QAction('Load Input', self)
        self.save_action = QAction('Save Input', self)
        self.exit_action = QAction('Exit', self)
        # self.help_documentation_action = QAction('Documentation', self)
        self.help_git_action = QAction('GitHub', self)
        self.help_contact_action = QAction('Contact Us', self)
        self.help_about_action = QAction('About', self)

        self.load_action.setShortcut('Ctrl+O')
        self.save_action.setShortcut('Ctrl+S')
        self.exit_action.triggered.connect(self.close)

        # self.help_documentation_action.triggered.connect(self.display_documentation)
        self.help_git_action.triggered.connect(self.display_github)
        self.help_contact_action.triggered.connect(self.display_contact)
        self.help_about_action.triggered.connect(self.display_about)

        self.file_menu.addAction(self.load_action)
        self.file_menu.addAction(self.save_action)
        self.file_menu.addAction(self.exit_action)
        self.file_menu.addSeparator()
        # self.help_menu.addAction(self.help_documentation_action)
        self.help_menu.addAction(self.help_git_action)
        self.help_menu.addAction(self.help_contact_action)
        self.help_menu.addAction(self.help_about_action)

        self.load_action.triggered.connect(self.tabs[0][1].load_settings)
        self.save_action.triggered.connect(self.tabs[0][1].save_settings)

    @staticmethod
    def display_documentation():
        """
        Display documentation.
        """
        text = f'{WINDOW_TITLE} \n\nAll relevant information about Z-Rad: https://github.com/radiomics-usz/zrad'
        CustomWarningBox(text, False).response()

    @staticmethod
    def display_github():
        """
        Display GitHub link.
        """
        text = f'{WINDOW_TITLE} \n\nhttps://github.com/radiomics-usz/zrad'
        CustomWarningBox(text, False).response()

    @staticmethod
    def display_contact():
        """
        Display contact info.
        """
        text = f'{WINDOW_TITLE} \n\nContact Us: zrad@usz.ch'
        CustomWarningBox(text, False).response()

    @staticmethod
    def display_about():
        """
        Display info about Z-Rad.
        """
        text = (f'{WINDOW_TITLE} \n\nDeveloped at the University Hospital Zurich '
                'by the Department of Radiation Oncology')
        CustomWarningBox(text, False).response()

    def tab_changed(self, index: int):
        """
        Handle the change of tabs.
        """
        self.load_action.triggered.disconnect()
        self.save_action.triggered.disconnect()

        tab = self.tabs[index][1]
        self.load_action.triggered.connect(tab.load_settings)
        self.save_action.triggered.connect(tab.save_settings)

        self.load_action.setText('Load Input')
        self.save_action.setText('Save Input')


def main():
    if sys.platform.startswith('win'):
        freeze_support()

    # Set high DPI scaling attributes before creating the QApplication instance
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.ButtonText, Qt.white)
    app.setPalette(palette)

    # Set font size in pixels
    font = QFont(FONT_FAMILY)
    font.setPixelSize(FONT_SIZE)
    app.setFont(font)
    app.setWindowIcon(QIcon(resource_path("docs/logos/icon.ico")))

    splash_pix = QPixmap(resource_path("docs/logos/ZRadLogo.jpg"))
    splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    splash.setWindowFlag(Qt.FramelessWindowHint)  # No window frame
    splash.show()

    # Simulate some initialization time (3 seconds)
    QTimer.singleShot(1100, splash.close)

    exe = ZRad()
    exe.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
