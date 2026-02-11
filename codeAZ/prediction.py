# app.py
import sys
import joblib
import pandas as pd
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QLabel, QPushButton, QGroupBox, QMessageBox
)
from PySide6.QtCore import Qt

from model import parse_us_address, predict_interval_from_constraints


def _money(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "â€”"
    return f"{x:,.0f} $"


def _to_optional_int(s: str):
    s = (s or "").strip()
    if not s:
        return None
    try:
        return int(float(s))
    except:
        return None


class HousingPredictorUI(QWidget):
    def __init__(self, pipe, df_train, feature_cols, use_log=True):
        super().__init__()
        self.pipe = pipe
        self.df_train = df_train
        self.feature_cols = feature_cols
        self.use_log = use_log

        self.setWindowTitle("Housing Price Predictor")
        self.setMinimumWidth(650)

        root = QVBoxLayout(self)

        # Address group
        g_addr = QGroupBox("Address (optional)")
        addr_layout = QVBoxLayout(g_addr)

        row1 = QHBoxLayout()
        self.addr_edit = QLineEdit()
        self.addr_edit.setPlaceholderText("100 W 2nd St, Boston, MA 02127, United States")
        self.parse_btn = QPushButton("Parse")
        self.parse_btn.clicked.connect(self.on_parse_address)

        row1.addWidget(self.addr_edit, 1)
        row1.addWidget(self.parse_btn, 0)
        addr_layout.addLayout(row1)
        root.addWidget(g_addr)

        # Features group
        g_feat = QGroupBox("Inputs")
        form = QFormLayout(g_feat)

        self.city_edit = QLineEdit()
        self.state_edit = QLineEdit()
        self.zip_edit = QLineEdit()

        self.beds_edit = QLineEdit()
        self.baths_edit = QLineEdit()
        self.living_edit = QLineEdit()
        self.zpop_edit = QLineEdit()

        self.beds_edit.setPlaceholderText("Optional")
        self.baths_edit.setPlaceholderText("Optional")
        self.living_edit.setPlaceholderText("Optional")
        self.zpop_edit.setPlaceholderText("Optional")

        form.addRow("City *", self.city_edit)
        form.addRow("State *", self.state_edit)
        form.addRow("Zip Code", self.zip_edit)
        form.addRow("Beds", self.beds_edit)
        form.addRow("Baths", self.baths_edit)
        form.addRow("Living Space", self.living_edit)
        form.addRow("Zip Code Population", self.zpop_edit)
        root.addWidget(g_feat)

        # Buttons
        row_btn = QHBoxLayout()
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.on_clear)

        self.predict_btn = QPushButton("Predict")
        self.predict_btn.clicked.connect(self.on_predict)

        row_btn.addStretch(1)
        row_btn.addWidget(self.clear_btn)
        row_btn.addWidget(self.predict_btn)
        root.addLayout(row_btn)

        # Output
        g_out = QGroupBox("Result")
        out_layout = QFormLayout(g_out)

        self.used_rows_lbl = QLabel("â€”")
        self.med_lbl = QLabel("â€”")
        self.low_lbl = QLabel("â€”")
        self.high_lbl = QLabel("â€”")

        for lbl in [self.used_rows_lbl, self.med_lbl, self.low_lbl, self.high_lbl]:
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)

        out_layout.addRow("Matching rows used", self.used_rows_lbl)
        out_layout.addRow("Median prediction", self.med_lbl)
        out_layout.addRow("80% interval (low)", self.low_lbl)
        out_layout.addRow("80% interval (high)", self.high_lbl)
        root.addWidget(g_out)

        note = QLabel("(*) City + State required. Add Beds/Baths/Living Space to narrow the interval.")
        note.setStyleSheet("opacity: 0.75;")
        root.addWidget(note)

    def on_parse_address(self):
        addr = self.addr_edit.text().strip()
        if not addr:
            QMessageBox.information(self, "Address", "Type an address first.")
            return
        try:
            info = parse_us_address(addr)
            if info.get("City"):
                self.city_edit.setText(info["City"])
            if info.get("State"):
                self.state_edit.setText(info["State"])
            if info.get("Zip Code"):
                self.zip_edit.setText(info["Zip Code"])
        except Exception as e:
            QMessageBox.critical(self, "Parse error", str(e))

    def build_constraints(self):
        city = self.city_edit.text().strip()
        state = self.state_edit.text().strip()

        if not city or not state:
            raise ValueError("City and State are required.")

        constraints = {"City": city, "State": state}

        zipc = self.zip_edit.text().strip()
        if zipc:
            constraints["Zip Code"] = zipc

        beds = _to_optional_int(self.beds_edit.text())
        baths = _to_optional_int(self.baths_edit.text())
        living = _to_optional_int(self.living_edit.text())
        zpop = _to_optional_int(self.zpop_edit.text())

        if beds is not None:
            constraints["Beds"] = beds
        if baths is not None:
            constraints["Baths"] = baths
        if living is not None:
            constraints["Living Space"] = living
        if zpop is not None:
            constraints["Zip Code Population"] = zpop

        return constraints

    def on_predict(self):
        try:
            constraints = self.build_constraints()
            low, med, high, n_used = predict_interval_from_constraints(
                self.pipe, self.df_train, self.feature_cols, constraints,
                n_samples=1000, alpha=0.20, use_log=self.use_log, random_state=42
            )
            self.used_rows_lbl.setText(str(n_used))
            self.med_lbl.setText(_money(med))
            self.low_lbl.setText(_money(low))
            self.high_lbl.setText(_money(high))
        except Exception as e:
            QMessageBox.critical(self, "Prediction error", str(e))

    def on_clear(self):
        self.addr_edit.clear()
        self.city_edit.clear()
        self.state_edit.clear()
        self.zip_edit.clear()
        self.beds_edit.clear()
        self.baths_edit.clear()
        self.living_edit.clear()
        self.zpop_edit.clear()
        self.used_rows_lbl.setText("â€”")
        self.med_lbl.setText("â€”")
        self.low_lbl.setText("â€”")
        self.high_lbl.setText("â€”")


def load_everything():
    from pathlib import Path
        MODEL_PATH = Path(__file__).resolve().parent.parent / "housing_pipe.joblib"
    from pathlib import Path
        DATA_PATH = Path(__file__).resolve().parent.parent / "American_Housing_Data_20231209.csv"

    bundle = joblib.load(MODEL_PATH)
    pipe = bundle["pipe"]
    feature_cols = bundle["feature_cols"]
    use_log = bundle["use_log"]

    df = pd.read_csv(DATA_PATH)

    if "Zip Code" in df.columns:
        df["Zip Code"] = (
            df["Zip Code"].astype(str)
            .str.replace(" ", "", regex=False)
            .str.replace(r"\.0$", "", regex=True)
        )

    for col in ["Price", "Beds", "Baths", "Living Space", "Zip Code Population"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Price", "Beds", "Baths", "Living Space", "City", "State"])
    return pipe, df, feature_cols, use_log


if __name__ == "__main__":
    pipe, df, feature_cols, use_log = load_everything()

    app = QApplication(sys.argv)
    w = HousingPredictorUI(pipe, df, feature_cols, use_log=use_log)
    w.show()
    sys.exit(app.exec())
