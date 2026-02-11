# tab_corr.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QSpinBox, QCheckBox, QGroupBox, QPlainTextEdit
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


# ✅ CSV path: relative to project root (no hardcoded C:\...)
# Assumes the CSV is at the project root: American_Housing_Data_20231209.csv
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "American_Housing_Data_20231209.csv"


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Try to find a column in df matching any candidate names (case-insensitive).
    We also accept loose matches (spaces/underscores).
    """
    cols = list(df.columns)
    norm = {c: "".join(str(c).lower().replace("_", " ").split()) for c in cols}

    cand_norm = ["".join(x.lower().replace("_", " ").split()) for x in candidates]

    # exact normalized match
    for c in cols:
        if norm[c] in cand_norm:
            return c

    # contains match
    for c in cols:
        for cn in cand_norm:
            if cn and cn in norm[c]:
                return c

    return None


class CorrelationTab(QWidget):
    """
    Tab that provides:
    - E6: correlation between two numeric columns + optional scatter plot
    - E8: summary indicators (mean price by city / zip / income bins)
    """

    def __init__(self):
        super().__init__()

        # Load full dataset once
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"CSV not found: {DATA_PATH}\n"
                f"Put American_Housing_Data_20231209.csv at project root: {PROJECT_ROOT}"
            )

        self.df = pd.read_csv(DATA_PATH)

        # Detect key columns for E8 (robust to naming differences)
        self.col_price = _find_col(self.df, ["price", "saleprice", "sale price", "listprice", "list price"])
        self.col_city = _find_col(self.df, ["city", "ville"])
        self.col_zip = _find_col(self.df, ["zip", "zipcode", "zip code", "postal", "postalcode", "postal code"])
        self.col_income = _find_col(self.df, ["medianincome", "median income", "income", "revenu", "revenu median"])

        # Prepare numeric df for correlation
        self.num = self.df.select_dtypes(include=["number"]).copy()
        self.cols = list(self.num.columns)

        root = QVBoxLayout(self)

        # -------------------------
        # (A) CORRELATION CONTROLS (E6)
        # -------------------------
        corr_group = QGroupBox("Corrélations (E6)")
        corr_layout = QVBoxLayout(corr_group)

        ctrl = QHBoxLayout()
        self.cb_x = QComboBox()
        self.cb_x.addItems(self.cols)

        self.cb_y = QComboBox()
        self.cb_y.addItems(self.cols)
        self.cb_y.setCurrentIndex(1 if len(self.cols) > 1 else 0)

        self.spin_sample = QSpinBox()
        self.spin_sample.setRange(1000, 200000)
        self.spin_sample.setValue(20000)

        self.chk_scatter = QCheckBox("Afficher nuage de points")
        self.chk_scatter.setChecked(True)

        self.btn_corr = QPushButton("Calculer corrélation")
        self.btn_corr.clicked.connect(self.update_plot)

        ctrl.addWidget(QLabel("X:"))
        ctrl.addWidget(self.cb_x)
        ctrl.addWidget(QLabel("Y:"))
        ctrl.addWidget(self.cb_y)
        ctrl.addWidget(QLabel("Échantillon:"))
        ctrl.addWidget(self.spin_sample)
        ctrl.addWidget(self.chk_scatter)
        ctrl.addWidget(self.btn_corr)
        ctrl.addStretch(1)

        corr_layout.addLayout(ctrl)

        self.lbl_corr = QLabel("Cliquez sur « Calculer corrélation »")
        self.lbl_corr.setStyleSheet("font-size: 14px;")
        corr_layout.addWidget(self.lbl_corr)

        self.fig = plt.figure(figsize=(7, 5))
        self.canvas = FigureCanvas(self.fig)
        corr_layout.addWidget(self.canvas)

        root.addWidget(corr_group)

        # -------------------------
        # (B) KPI / INDICATORS (E8)
        # -------------------------
        kpi_group = QGroupBox("Indicateurs statistiques (E8)")
        kpi_layout = QVBoxLayout(kpi_group)

        kpi_ctrl = QHBoxLayout()
        self.spin_topn = QSpinBox()
        self.spin_topn.setRange(3, 50)
        self.spin_topn.setValue(10)

        self.btn_kpi = QPushButton("Calculer indicateurs")
        self.btn_kpi.clicked.connect(self.update_kpis)

        kpi_ctrl.addWidget(QLabel("Top N:"))
        kpi_ctrl.addWidget(self.spin_topn)
        kpi_ctrl.addWidget(self.btn_kpi)
        kpi_ctrl.addStretch(1)

        kpi_layout.addLayout(kpi_ctrl)

        self.kpi_text = QPlainTextEdit()
        self.kpi_text.setReadOnly(True)
        self.kpi_text.setStyleSheet("font-family: Consolas, monospace; font-size: 12px;")
        self.kpi_text.setPlaceholderText("Cliquez sur « Calculer indicateurs »")
        kpi_layout.addWidget(self.kpi_text)

        root.addWidget(kpi_group)

        # Note: we don't auto-run at startup to keep UI fast

    # -------------------------
    # E6: correlation plot
    # -------------------------
    def update_plot(self):
        x = self.cb_x.currentText()
        y = self.cb_y.currentText()

        if x == y:
            self.lbl_corr.setText("Choisissez deux colonnes différentes.")
            return

        df_s = self.num[[x, y]].dropna()
        if len(df_s) == 0:
            self.lbl_corr.setText("Aucune ligne valide après dropna().")
            return

        n = min(self.spin_sample.value(), len(df_s))
        if len(df_s) > n:
            df_s = df_s.sample(n=n, random_state=42)

        xs = pd.to_numeric(df_s[x], errors="coerce")
        ys = pd.to_numeric(df_s[y], errors="coerce")
        m = xs.notna() & ys.notna()
        xs = xs[m]
        ys = ys[m]

        if len(xs) < 3:
            self.lbl_corr.setText("Pas assez de points pour calculer la corrélation.")
            return

        r = float(xs.corr(ys))
        self.lbl_corr.setText(f"Pearson r({x}, {y}) = {r:.4f}   (n={len(xs):,})")

        self.fig.clear()
        ax = self.fig.add_subplot(111)

        if self.chk_scatter.isChecked():
            max_plot = 15000
            if len(xs) > max_plot:
                idx = np.random.default_rng(42).choice(len(xs), size=max_plot, replace=False)
                xs_plot = xs.iloc[idx]
                ys_plot = ys.iloc[idx]
            else:
                xs_plot, ys_plot = xs, ys

            ax.scatter(xs_plot, ys_plot, s=6)
            ax.set_title(f"{x} vs {y} (échantillonné)")
            ax.set_xlabel(x)
            ax.set_ylabel(y)
        else:
            ax.text(
                0.5, 0.5,
                f"Pearson r = {r:.4f}\n(n={len(xs):,})",
                ha="center", va="center",
                transform=ax.transAxes
            )
            ax.set_axis_off()

        self.fig.tight_layout()
        self.canvas.draw()

    # -------------------------
    # E8: indicators (mean price by city/zip/income bins)
    # -------------------------
    def update_kpis(self):
        topn = int(self.spin_topn.value())

        # Check detected columns
        missing = []
        if self.col_price is None:
            missing.append("Price")
        if self.col_city is None:
            missing.append("City")
        if self.col_zip is None:
            missing.append("Zip Code")
        if self.col_income is None:
            missing.append("Median Income")

        if missing:
            self.kpi_text.setPlainText(
                "Impossible de calculer E8 : colonnes non trouvées automatiquement.\n"
                f"Manquantes: {', '.join(missing)}\n\n"
                "➡️ Solution: vérifie les noms exacts des colonnes dans le CSV et\n"
                "mets-les en dur dans le code (col_price/col_city/col_zip/col_income)."
            )
            return

        df = self.df.copy()

        # Clean numeric price & income
        df[self.col_price] = pd.to_numeric(df[self.col_price], errors="coerce")
        df[self.col_income] = pd.to_numeric(df[self.col_income], errors="coerce")

        df = df.dropna(subset=[self.col_price])
        if len(df) == 0:
            self.kpi_text.setPlainText("Aucune valeur de prix exploitable (après nettoyage).")
            return

        # KPI 0: overall mean price
        mean_price = df[self.col_price].mean()

        # KPI 1: mean price by city
        by_city = (
            df.dropna(subset=[self.col_city])
              .groupby(self.col_city)[self.col_price]
              .mean()
              .sort_values(ascending=False)
              .head(topn)
        )

        # KPI 2: mean price by zip
        by_zip = (
            df.dropna(subset=[self.col_zip])
              .groupby(self.col_zip)[self.col_price]
              .mean()
              .sort_values(ascending=False)
              .head(topn)
        )

        # KPI 3: mean price by income bins
        # bins are generic; adjust later if needed
        bins = [0, 40000, 60000, 80000, 100000, 150000, 250000, float("inf")]
        labels = ["<40k", "40-60k", "60-80k", "80-100k", "100-150k", "150-250k", "250k+"]

        df_income = df.dropna(subset=[self.col_income]).copy()
        if len(df_income) > 0:
            df_income["Income_Bin"] = pd.cut(
                df_income[self.col_income],
                bins=bins,
                labels=labels,
                right=False
            )

            by_income = (
                df_income.dropna(subset=["Income_Bin"])
                         .groupby("Income_Bin")[self.col_price]
                         .mean()
                         .sort_values(ascending=False)
            )
        else:
            by_income = pd.Series(dtype=float)

        # Build text report
        lines: list[str] = []
        lines.append("=== E8 : Indicateurs statistiques ===")
        lines.append(f"Colonnes détectées :")
        lines.append(f"- Prix: {self.col_price}")
        lines.append(f"- Ville: {self.col_city}")
        lines.append(f"- Code postal: {self.col_zip}")
        lines.append(f"- Revenu médian: {self.col_income}")
        lines.append("")
        lines.append(f"Prix moyen global = {mean_price:,.0f}".replace(",", " "))
        lines.append("")

        lines.append(f"Top {topn} villes (prix moyen) :")
        if len(by_city) == 0:
            lines.append("  (aucune donnée ville)")
        else:
            for k, v in by_city.items():
                lines.append(f"  - {k}: {v:,.0f}".replace(",", " "))
        lines.append("")

        lines.append(f"Top {topn} codes postaux (prix moyen) :")
        if len(by_zip) == 0:
            lines.append("  (aucune donnée code postal)")
        else:
            for k, v in by_zip.items():
                lines.append(f"  - {k}: {v:,.0f}".replace(",", " "))
        lines.append("")

        lines.append("Prix moyen par tranche de revenu médian :")
        if len(by_income) == 0:
            lines.append("  (aucune donnée revenu médian)")
        else:
            for k, v in by_income.items():
                lines.append(f"  - {k}: {v:,.0f}".replace(",", " "))

        self.kpi_text.setPlainText("\n".join(lines))
