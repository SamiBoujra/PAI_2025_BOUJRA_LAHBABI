# tab_explore.py
from __future__ import annotations

from pathlib import Path
from typing import List, Any

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QSortFilterProxyModel
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableView, QLabel, QLineEdit,
    QPushButton, QFileDialog, QFormLayout, QDoubleSpinBox, QComboBox,
    QSplitter, QMessageBox, QSpinBox
)

# --------------------------- Config ---------------------------
from pathlib import Path
DATA_PATH = Path(__file__).resolve().parent.parent / "American_Housing_Data_20231209.csv"

EXPECTED_COLUMNS = [
    "Zip Code", "Price", "Beds", "Baths", "Living Space", "Address",
    "City", "State", "Zip Code Population", "Zip Code Density", "County",
    "Median Household Income", "Latitude", "Longitude",
]

# --------------------- Modèle pandas -> Qt --------------------
class PandasModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df.reset_index(drop=True)

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else self._df.shape[1]

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid():
            return None
        if role in (Qt.DisplayRole, Qt.EditRole):
            val = self._df.iat[index.row(), index.column()]
            if isinstance(val, (int, np.integer)):
                return f"{int(val):,}".replace(",", " ")
            if isinstance(val, (float, np.floating)):
                s = f"{float(val):,.2f}".replace(",", " ")
                return s.replace(".00", "")
            return str(val)
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole) -> Any:
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section])
        return str(section + 1)

    def dataframe(self) -> pd.DataFrame:
        return self._df


# -------------------- Proxy de filtrage Qt --------------------
class RealEstateFilterProxy(QSortFilterProxyModel):
    def __init__(self):
        super().__init__()
        self.setDynamicSortFilter(True)
        self.setFilterCaseSensitivity(Qt.CaseInsensitive)

        self.min_price = None
        self.max_price = None
        self.min_space = None
        self.max_space = None
        self.min_beds = None
        self.max_beds = None
        self.min_income = None
        self.max_income = None
        self.city_substr = ""
        self.state_exact = ""
        self.search_text = ""

    def _col(self, name: str) -> int:
        model = self.sourceModel()
        if model is None:
            return -1
        try:
            return list(model.dataframe().columns).index(name)
        except ValueError:
            return -1

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        model: PandasModel = self.sourceModel()
        if model is None:
            return True
        df = model.dataframe()

        def val(col_name):
            col_idx = self._col(col_name)
            if col_idx < 0:
                return None
            return df.iat[source_row, col_idx]

        price = val("Price")
        if isinstance(price, (int, float, np.integer, np.floating)):
            if self.min_price is not None and price < self.min_price:
                return False
            if self.max_price is not None and price > self.max_price:
                return False

        space = val("Living Space")
        if isinstance(space, (int, float, np.integer, np.floating)):
            if self.min_space is not None and space < self.min_space:
                return False
            if self.max_space is not None and space > self.max_space:
                return False

        beds = val("Beds")
        if isinstance(beds, (int, float, np.integer, np.floating)):
            if self.min_beds is not None and beds < self.min_beds:
                return False
            if self.max_beds is not None and beds > self.max_beds:
                return False

        income = val("Median Household Income")
        if isinstance(income, (int, float, np.integer, np.floating)):
            if self.min_income is not None and income < self.min_income:
                return False
            if self.max_income is not None and income > self.max_income:
                return False

        city = str(val("City") or "")
        if self.city_substr and self.city_substr.lower() not in city.lower():
            return False

        state = str(val("State") or "")
        if self.state_exact and state != self.state_exact:
            return False

        address = str(val("Address") or "")
        if self.search_text and self.search_text.lower() not in address.lower():
            return False

        return True


# ------------------------ UI Exploration ----------------------
class ExplorationTab(QWidget):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df

        self.model = PandasModel(self.df)
        self.proxy = RealEstateFilterProxy()
        self.proxy.setSourceModel(self.model)

        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setStretchLastSection(True)

        self._build_filters()

        splitter = QSplitter()
        left = QWidget()
        left.setLayout(self.filters_layout)
        splitter.addWidget(left)
        splitter.addWidget(self.table)
        splitter.setSizes([340, 900])

        main_layout = QVBoxLayout()
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

        self.apply_filters()

    def _build_filters(self):
        self.filters_layout = QFormLayout()

        self.spin_min_price = QDoubleSpinBox(); self.spin_min_price.setRange(0, 1e9); self.spin_min_price.setPrefix("Min $"); self.spin_min_price.setDecimals(0)
        self.spin_max_price = QDoubleSpinBox(); self.spin_max_price.setRange(0, 1e9); self.spin_max_price.setPrefix("Max $"); self.spin_max_price.setDecimals(0)
        self.filters_layout.addRow(QLabel("Prix ($) :"), self._row(self.spin_min_price, self.spin_max_price))

        self.spin_min_space = QDoubleSpinBox(); self.spin_min_space.setRange(0, 1e6); self.spin_min_space.setPrefix("Min "); self.spin_min_space.setDecimals(0)
        self.spin_max_space = QDoubleSpinBox(); self.spin_max_space.setRange(0, 1e6); self.spin_max_space.setPrefix("Max "); self.spin_max_space.setDecimals(0)
        self.filters_layout.addRow(QLabel("Surface (ft²) :"), self._row(self.spin_min_space, self.spin_max_space))

        self.spin_min_beds = QSpinBox(); self.spin_min_beds.setRange(0, 50); self.spin_min_beds.setPrefix("Min ")
        self.spin_max_beds = QSpinBox(); self.spin_max_beds.setRange(0, 50); self.spin_max_beds.setPrefix("Max ")
        self.filters_layout.addRow(QLabel("Chambres (Beds) :"), self._row(self.spin_min_beds, self.spin_max_beds))

        self.spin_min_income = QDoubleSpinBox(); self.spin_min_income.setRange(0, 1e7); self.spin_min_income.setPrefix("Min $"); self.spin_min_income.setDecimals(0)
        self.spin_max_income = QDoubleSpinBox(); self.spin_max_income.setRange(0, 1e7); self.spin_max_income.setPrefix("Max $"); self.spin_max_income.setDecimals(0)
        self.filters_layout.addRow(QLabel("Revenu médian ($) :"), self._row(self.spin_min_income, self.spin_max_income))

        self.edit_city = QLineEdit(); self.edit_city.setPlaceholderText("Contient… (ex: New York)")
        self.filters_layout.addRow(QLabel("Ville (contient) :"), self.edit_city)

        self.combo_state = QComboBox(); self.combo_state.addItem("")
        if "State" in self.df.columns:
            states = sorted(map(str, self.df["State"].dropna().unique()))
            self.combo_state.addItems(states)
        self.filters_layout.addRow(QLabel("État (exact) :"), self.combo_state)

        self.edit_search = QLineEdit(); self.edit_search.setPlaceholderText("Recherche dans Address…")
        self.filters_layout.addRow(QLabel("Recherche (Address) :"), self.edit_search)

        self.btn_apply = QPushButton("Appliquer les filtres")
        self.btn_apply.clicked.connect(self.apply_filters)

        self.btn_reset = QPushButton("Réinitialiser")
        self.btn_reset.clicked.connect(self.reset_filters)

        row_btns = QWidget()
        h = QHBoxLayout(row_btns)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.btn_apply)
        h.addWidget(self.btn_reset)
        self.filters_layout.addRow(row_btns)

        self.lbl_count = QLabel("")
        self.filters_layout.addRow(QLabel("Résultats :"), self.lbl_count)

        self.btn_export = QPushButton("Exporter CSV (filtré)")
        self.btn_export.clicked.connect(self.export_csv)
        self.filters_layout.addRow(self.btn_export)

        self.edit_city.returnPressed.connect(self.apply_filters)
        self.edit_search.returnPressed.connect(self.apply_filters)

    def _row(self, *widgets) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        for wd in widgets:
            lay.addWidget(wd)
        return w

    def apply_filters(self):
        self.proxy.min_price = self.spin_min_price.value() or None
        self.proxy.max_price = self.spin_max_price.value() or None
        if self.proxy.max_price == 0:
            self.proxy.max_price = None

        self.proxy.min_space = self.spin_min_space.value() or None
        self.proxy.max_space = self.spin_max_space.value() or None
        if self.proxy.max_space == 0:
            self.proxy.max_space = None

        self.proxy.min_income = self.spin_min_income.value() or None
        self.proxy.max_income = self.spin_max_income.value() or None
        if self.proxy.max_income == 0:
            self.proxy.max_income = None

        self.proxy.min_beds = self.spin_min_beds.value() or None
        self.proxy.max_beds = self.spin_max_beds.value() or None
        if self.proxy.max_beds == 0:
            self.proxy.max_beds = None

        self.proxy.city_substr = self.edit_city.text().strip()
        self.proxy.state_exact = self.combo_state.currentText().strip()
        self.proxy.search_text = self.edit_search.text().strip()

        self.proxy.invalidateFilter()
        self.lbl_count.setText(f"{self.proxy.rowCount():,}".replace(",", " "))

    def reset_filters(self):
        for w in [self.spin_min_price, self.spin_max_price, self.spin_min_space, self.spin_max_space,
                  self.spin_min_income, self.spin_max_income]:
            w.setValue(0)
        for w in [self.spin_min_beds, self.spin_max_beds]:
            w.setValue(0)
        self.edit_city.clear()
        self.combo_state.setCurrentIndex(0)
        self.edit_search.clear()
        self.apply_filters()

    def _filtered_dataframe(self) -> pd.DataFrame:
        df_all = self.model.dataframe()
        rows: List[int] = []
        for r in range(self.proxy.rowCount()):
            src_index = self.proxy.mapToSource(self.proxy.index(r, 0))
            rows.append(src_index.row())
        return df_all.iloc[rows].copy()

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Exporter CSV filtré", "filtered_exploration.csv", "CSV (*.csv)")
        if not path:
            return
        df_filtered = self._filtered_dataframe()
        try:
            df_filtered.to_csv(path, index=False)
            QMessageBox.information(self, "Export CSV", f"Export réussi vers:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export CSV", f"Erreur d'export:\n{e}")


def load_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        print("[Avertissement] Colonnes manquantes dans le CSV:", missing)
    return df


class ExplorationTabWidget(QWidget):
    """
    Wrapper 'onglet' qui charge le CSV par défaut.
    Dans main.py tu importes ExplorationTabWidget et tu fais tabs.addTab(ExplorationTabWidget(), "Exploration")
    """
    def __init__(self):
        super().__init__()
        df = load_dataframe(DATA_PATH)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(ExplorationTab(df))
