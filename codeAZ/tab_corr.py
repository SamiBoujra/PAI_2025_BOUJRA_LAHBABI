# tab_corr.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QSpinBox, QCheckBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

DATA_PATH = r"C:\Users\pc\Downloads\projet PAI\projet PAI\American_Housing_Data_20231209.csv"


class CorrelationTab(QWidget):
    def __init__(self):
        super().__init__()

        df = pd.read_csv(DATA_PATH)

        # numeric only
        self.num = df.select_dtypes(include=["number"]).copy()
        self.cols = list(self.num.columns)

        root = QVBoxLayout(self)

        # Controls
        ctrl = QHBoxLayout()
        self.cb_x = QComboBox(); self.cb_x.addItems(self.cols)
        self.cb_y = QComboBox(); self.cb_y.addItems(self.cols)
        self.cb_y.setCurrentIndex(1 if len(self.cols) > 1 else 0)

        self.spin_sample = QSpinBox()
        self.spin_sample.setRange(1000, 200000)
        self.spin_sample.setValue(20000)  # default sample size

        self.chk_scatter = QCheckBox("Show scatter")
        self.chk_scatter.setChecked(True)

        self.btn = QPushButton("Compute")
        self.btn.clicked.connect(self.update_plot)

        ctrl.addWidget(QLabel("X:"))
        ctrl.addWidget(self.cb_x)
        ctrl.addWidget(QLabel("Y:"))
        ctrl.addWidget(self.cb_y)
        ctrl.addWidget(QLabel("Sample:"))
        ctrl.addWidget(self.spin_sample)
        ctrl.addWidget(self.chk_scatter)
        ctrl.addWidget(self.btn)

        ctrl.addStretch(1)
        root.addLayout(ctrl)

        self.lbl = QLabel("Press Compute")
        self.lbl.setStyleSheet("font-size: 14px;")
        root.addWidget(self.lbl)

        self.fig = plt.figure(figsize=(7, 5))
        self.canvas = FigureCanvas(self.fig)
        root.addWidget(self.canvas)

        # IMPORTANT: do NOT compute at startup (keeps UI fast)
        # self.update_plot()

    def update_plot(self):
        x = self.cb_x.currentText()
        y = self.cb_y.currentText()
        if x == y:
            self.lbl.setText("Choose two different columns.")
            return

        # Take a random sample to make it fast
        n = min(self.spin_sample.value(), len(self.num))
        df_s = self.num[[x, y]].dropna()

        if len(df_s) == 0:
            self.lbl.setText("No valid rows after dropna().")
            return

        if len(df_s) > n:
            df_s = df_s.sample(n=n, random_state=42)

        # Ensure float (avoid weird dtypes)
        xs = pd.to_numeric(df_s[x], errors="coerce")
        ys = pd.to_numeric(df_s[y], errors="coerce")
        m = xs.notna() & ys.notna()
        xs = xs[m]; ys = ys[m]

        if len(xs) < 3:
            self.lbl.setText("Not enough data points.")
            return

        # Fast Pearson correlation (pandas)
        r = float(xs.corr(ys))

        self.lbl.setText(f"Pearson r({x}, {y}) = {r:.4f}   (n={len(xs):,})")

        # Plot
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        if self.chk_scatter.isChecked():
            # cap points drawn (plotting can be slow)
            max_plot = 15000
            if len(xs) > max_plot:
                idx = np.random.default_rng(42).choice(len(xs), size=max_plot, replace=False)
                xs_plot = xs.iloc[idx]
                ys_plot = ys.iloc[idx]
            else:
                xs_plot, ys_plot = xs, ys

            ax.scatter(xs_plot, ys_plot, s=6)
            ax.set_title(f"{x} vs {y} (sampled)")
            ax.set_xlabel(x)
            ax.set_ylabel(y)
        else:
            ax.text(0.5, 0.5, f"Pearson r = {r:.4f}\n(n={len(xs):,})",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()

        self.fig.tight_layout()
        self.canvas.draw()
