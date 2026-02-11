# tab_map.py
import pandas as pd
import folium
from folium.plugins import FastMarkerCluster

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QDoubleSpinBox, QSpinBox, QLineEdit, QComboBox, QLabel, QPushButton
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtCore import QUrl


from pathlib import Path
DATA_PATH = Path(__file__).resolve().parent.parent / "American_Housing_Data_20231209.csv"


def fmt_price(x):
    try:
        return f"${float(x):,.0f}"
    except Exception:
        return "$0"


class CartographyDynamic(QWidget):
    """Carte Folium (HTML) mise Ã  jour sur action (bouton Appliquer / Enter)."""

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df

        # ---- Filtres ----
        self.spin_min_price = QDoubleSpinBox()
        self.spin_min_price.setRange(0, 1e8)
        self.spin_min_price.setDecimals(0)
        self.spin_min_price.setPrefix("Min $")

        self.spin_max_price = QDoubleSpinBox()
        self.spin_max_price.setRange(0, 1e8)
        self.spin_max_price.setDecimals(0)
        self.spin_max_price.setPrefix("Max $")

        self.spin_min_beds = QSpinBox()
        self.spin_min_beds.setRange(0, 50)
        self.spin_min_beds.setPrefix("Min Beds ")

        self.spin_max_beds = QSpinBox()
        self.spin_max_beds.setRange(0, 50)
        self.spin_max_beds.setPrefix("Max Beds ")

        self.edit_city = QLineEdit()
        self.edit_city.setPlaceholderText("Ville contientâ€¦ (EntrÃ©e pour appliquer)")
        self.edit_city.returnPressed.connect(self.update_map)

        self.combo_state = QComboBox()
        self.combo_state.addItem("")  # pas de filtre
        if "State" in df.columns:
            states = sorted(map(str, df["State"].dropna().unique()))
            self.combo_state.addItems(states)

        # ---- Bouton appliquer + info ----
        self.btn_apply = QPushButton("Appliquer les filtres")
        self.btn_apply.clicked.connect(self.update_map)

        self.lbl_count = QLabel("")

        # ---- Carte ----
        self.web = QWebEngineView()

        # ---- Layout ----
        form = QFormLayout()
        form.addRow("Prix ($)", self._row(self.spin_min_price, self.spin_max_price))
        form.addRow("Chambres (Beds)", self._row(self.spin_min_beds, self.spin_max_beds))
        form.addRow("Ville", self.edit_city)
        form.addRow("Ã‰tat", self.combo_state)

        lay = QVBoxLayout(self)
        lay.addLayout(form)

        row_btn = QHBoxLayout()
        row_btn.addWidget(self.btn_apply)
        row_btn.addWidget(self.lbl_count)
        row_btn.addStretch(1)
        lay.addLayout(row_btn)

        lay.addWidget(QLabel("Carte des biens filtrÃ©s :"))
        lay.addWidget(self.web, stretch=1)

        # GÃ©nÃ©ration initiale
        self.update_map()

    def _row(self, *widgets):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        for wd in widgets:
            h.addWidget(wd)
        return w

    def filtered_df(self):
        df = self.df

        min_price = self.spin_min_price.value()
        max_price = self.spin_max_price.value()
        min_beds = self.spin_min_beds.value()
        max_beds = self.spin_max_beds.value()
        city = self.edit_city.text().strip()
        state = self.combo_state.currentText().strip()

        if "Price" in df.columns:
            if min_price > 0:
                df = df[df["Price"] >= min_price]
            if max_price > 0:
                df = df[df["Price"] <= max_price]

        if "Beds" in df.columns:
            if min_beds > 0:
                df = df[df["Beds"] >= min_beds]
            if max_beds > 0:
                df = df[df["Beds"] <= max_beds]

        if city and "City" in df.columns:
            df = df[df["City"].astype(str).str.contains(city, case=False, na=False)]

        if state and "State" in df.columns:
            df = df[df["State"].astype(str) == state]

        return df

    def update_map(self):
        """Met Ã  jour la carte Ã  partir du DataFrame filtrÃ© (FAST cluster)."""
        try:
            df_f = self.filtered_df()
            self.lbl_count.setText(f"{len(df_f):,} rÃ©sultats")

            m = folium.Map(location=[39.5, -98.35], zoom_start=4, tiles="CartoDB positron")

            if df_f.empty or not {"Latitude", "Longitude"}.issubset(df_f.columns):
                self.web.setHtml(m.get_root().render(), baseUrl=QUrl("https://localhost/"))
                return

            points = []
            popups = []

            for _, row in df_f.iterrows():
                try:
                    lat = float(row["Latitude"])
                    lon = float(row["Longitude"])
                except Exception:
                    continue

                html = (
                    f"<b>{row.get('Address','')}</b><br>"
                    f"{row.get('City','')}, {row.get('State','')} ({row.get('Zip Code','')})<br>"
                    f"Price: {fmt_price(row.get('Price', 0))}<br>"
                    f"Beds: {row.get('Beds','?')} | Baths: {row.get('Baths','?')} | "
                    f"Living Space: {row.get('Living Space','?')} ftÂ²"
                )

                points.append([lat, lon])
                popups.append(html)

            FastMarkerCluster(data=points, popups=popups).add_to(m)

            self.web.setHtml(m.get_root().render(), baseUrl=QUrl("https://localhost/"))

        except Exception as e:
            print(f"[ERREUR update_map] {e}")


class MapTab(QWidget):
    """Onglet Cartographie, prÃªt pour main.py"""
    def __init__(self):
        super().__init__()
        df = pd.read_csv(DATA_PATH)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(CartographyDynamic(df))
