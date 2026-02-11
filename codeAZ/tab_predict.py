# tab_predict.py
import joblib
import pandas as pd
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QLabel, QPushButton, QGroupBox, QMessageBox
)
from PySide6.QtCore import Qt

from model import parse_us_address, predict_interval_from_constraints

MODEL_PATH = r"C:\Users\pc\Downloads\projet PAI\housing_pipe.joblib"
DATA_PATH  = r"C:\Users\pc\Downloads\projet PAI\projet PAI\American_Housing_Data_20231209.csv"


def _money(x):
    return "—" if x is None else f"{x:,.0f} $"


def _opt_int(s):
    try:
        return int(s) if s.strip() else None
    except:
        return None


class PredictorTab(QWidget):
    def __init__(self):
        super().__init__()

        bundle = joblib.load(MODEL_PATH)
        self.pipe = bundle["pipe"]
        self.feature_cols = bundle["feature_cols"]
        self.use_log = bundle["use_log"]

        self.df = pd.read_csv(DATA_PATH)
        self.df = self.df.dropna(subset=["Price", "Beds", "Baths", "Living Space", "City", "State"])

        root = QVBoxLayout(self)

        # Address
        addr_box = QGroupBox("Address (optional)")
        addr_lay = QHBoxLayout(addr_box)
        self.addr = QLineEdit()
        self.addr.setPlaceholderText("1600 Pennsylvania Avenue NW, Washington, DC")
        btn_parse = QPushButton("Parse")
        btn_parse.clicked.connect(self.parse_address)
        addr_lay.addWidget(self.addr)
        addr_lay.addWidget(btn_parse)
        root.addWidget(addr_box)

        # Inputs
        form_box = QGroupBox("Inputs")
        form = QFormLayout(form_box)
        self.city = QLineEdit()
        self.state = QLineEdit()
        self.zipc = QLineEdit()
        self.beds = QLineEdit()
        self.baths = QLineEdit()
        self.space = QLineEdit()

        form.addRow("City*", self.city)
        form.addRow("State*", self.state)
        form.addRow("Zip Code", self.zipc)
        form.addRow("Beds", self.beds)
        form.addRow("Baths", self.baths)
        form.addRow("Living Space", self.space)
        root.addWidget(form_box)

        # Buttons
        row = QHBoxLayout()
        btn = QPushButton("Predict")
        btn.clicked.connect(self.predict)
        row.addStretch()
        row.addWidget(btn)
        root.addLayout(row)

        # Output
        out = QGroupBox("Result")
        f = QFormLayout(out)
        self.lbl_med = QLabel("—")
        self.lbl_low = QLabel("—")
        self.lbl_high = QLabel("—")
        f.addRow("Median", self.lbl_med)
        f.addRow("Low (80%)", self.lbl_low)
        f.addRow("High (80%)", self.lbl_high)
        root.addWidget(out)

    def parse_address(self):
        try:
            info = parse_us_address(self.addr.text())
            self.city.setText(info["City"])
            self.state.setText(info["State"])
            self.zipc.setText(info["Zip Code"])
        except Exception as e:
            QMessageBox.critical(self, "Parse error", str(e))

    def predict(self):
        if not self.city.text() or not self.state.text():
            QMessageBox.warning(self, "Error", "City and State required")
            return

        constraints = {
            "City": self.city.text(),
            "State": self.state.text(),
            "Zip Code": self.zipc.text() or None,
            "Beds": _opt_int(self.beds.text()),
            "Baths": _opt_int(self.baths.text()),
            "Living Space": _opt_int(self.space.text()),
        }

        low, med, high, _ = predict_interval_from_constraints(
            self.pipe, self.df, self.feature_cols, constraints, use_log=self.use_log
        )

        self.lbl_med.setText(_money(med))
        self.lbl_low.setText(_money(low))
        self.lbl_high.setText(_money(high))
