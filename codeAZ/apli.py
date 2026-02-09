# main.py
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QMessageBox

# Import the 4 tabs (each tab is defined in its own file)
from tab_predict import PredictorTab
from tab_corr import CorrelationTab
from tab_map import MapTab
from tab_explore import ExplorationTabWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("US Real Estate – Dashboard")
        self.resize(1300, 850)

        tabs = QTabWidget()
        tabs.addTab(PredictorTab(), "Prediction")
        tabs.addTab(CorrelationTab(), "Correlation")
        tabs.addTab(MapTab(), "Map")
        tabs.addTab(ExplorationTabWidget(), "Exploration")

        self.setCentralWidget(tabs)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    try:
        w = MainWindow()
        w.show()
        sys.exit(app.exec())
    except Exception as e:
        QMessageBox.critical(None, "Startup error", str(e))
        sys.exit(1)
