# main.py

import sys
# Import system-specific parameters and functions (used to exit the app properly)

from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QMessageBox
# Import required Qt classes:
# - QApplication: manages the GUI application's control flow
# - QMainWindow: main application window container
# - QTabWidget: widget that provides tabbed interface
# - QMessageBox: used to display error messages

# Import the 4 tabs (each tab is defined in its own file)
from codeAZ.tab_predict import PredictorTab
from codeAZ.tab_corr import CorrelationTab
from codeAZ.tab_map import MapTab
from codeAZ.tab_explore import ExplorationTabWidget


class MainWindow(QMainWindow):
    """
    Main application window.
    Contains a QTabWidget with four functional tabs:
    - Prediction
    - Correlation
    - Map
    - Exploration
    """

    def __init__(self):
        super().__init__()  # Initialize QMainWindow

        # Set window title displayed in the title bar
        self.setWindowTitle("US Real Estate – Dashboard")

        # Set initial window size (width, height)
        self.resize(1300, 850)

        # Create tab container
        tabs = QTabWidget()

        # Add each functional tab to the tab widget
        tabs.addTab(PredictorTab(), "Prediction")
        tabs.addTab(CorrelationTab(), "Correlation")
        tabs.addTab(MapTab(), "Map")
        tabs.addTab(ExplorationTabWidget(), "Exploration")

        # Set the tab widget as the central widget of the main window
        self.setCentralWidget(tabs)


# Entry point of the application
if __name__ == "__main__":
    # Create the QApplication instance (required for any Qt app)
    app = QApplication(sys.argv)

    try:
        # Create main window instance
        w = MainWindow()

        # Show the main window
        w.show()

        # Start the Qt event loop
        sys.exit(app.exec())

    except Exception as e:
        # If an error occurs at startup, show a critical error dialog
        QMessageBox.critical(None, "Startup error", str(e))

        # Exit with error code
        sys.exit(1)
