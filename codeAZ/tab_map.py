# tab_map.py

import pandas as pd
import folium
from folium.plugins import FastMarkerCluster
# Folium sert à générer une carte interactive (HTML/Leaflet).
# FastMarkerCluster permet d'afficher un grand nombre de points rapidement via clustering.

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QDoubleSpinBox, QSpinBox, QLineEdit, QComboBox, QLabel, QPushButton
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtCore import QUrl
# QWebEngineView permet d'afficher une page web (ici l'HTML généré par Folium) dans l'application Qt.

from pathlib import Path

# Chemin vers le CSV (relatif à la racine du projet)
DATA_PATH = Path(__file__).resolve().parent.parent / "American_Housing_Data_20231209.csv"


def fmt_price(x):
    """
    Convertit une valeur en string formatée style "$123,456".
    Utilise un try/except pour éviter de casser l'UI si x est NaN / string / None.
    """
    try:
        return f"${float(x):,.0f}"
    except Exception:
        return "$0"


class CartographyDynamic(QWidget):
    """
    Widget principal de cartographie :
    - panneau de filtres (prix, beds, ville, état)
    - bouton "Appliquer"
    - affichage d'une carte Folium dans un QWebEngineView

    La carte est régénérée "à la demande" (bouton ou Enter),
    ce qui évite de recalculer en continu (plus performant).
    """

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df  # DataFrame complet chargé une seule fois

        # =========================
        # 1) WIDGETS DE FILTRAGE
        # =========================

        # Prix min / max
        self.spin_min_price = QDoubleSpinBox()
        self.spin_min_price.setRange(0, 1e8)
        self.spin_min_price.setDecimals(0)
        self.spin_min_price.setPrefix("Min $")

        self.spin_max_price = QDoubleSpinBox()
        self.spin_max_price.setRange(0, 1e8)
        self.spin_max_price.setDecimals(0)
        self.spin_max_price.setPrefix("Max $")

        # Nombre de chambres (beds) min / max
        self.spin_min_beds = QSpinBox()
        self.spin_min_beds.setRange(0, 50)
        self.spin_min_beds.setPrefix("Min Beds ")

        self.spin_max_beds = QSpinBox()
        self.spin_max_beds.setRange(0, 50)
        self.spin_max_beds.setPrefix("Max Beds ")

        # Filtre ville (contient)
        self.edit_city = QLineEdit()
        self.edit_city.setPlaceholderText("Ville contient… (Entrée pour appliquer)")
        # UX: l'utilisateur peut appuyer sur Enter pour appliquer
        self.edit_city.returnPressed.connect(self.update_map)

        # Filtre état (exact) via combo
        self.combo_state = QComboBox()
        self.combo_state.addItem("")  # valeur vide = pas de filtre
        if "State" in df.columns:
            states = sorted(map(str, df["State"].dropna().unique()))
            self.combo_state.addItems(states)

        # =========================
        # 2) BOUTON + INFOS
        # =========================

        self.btn_apply = QPushButton("Appliquer les filtres")
        self.btn_apply.clicked.connect(self.update_map)

        # Label affichant le nombre de résultats après filtrage
        self.lbl_count = QLabel("")

        # =========================
        # 3) CARTE (WEBVIEW)
        # =========================

        # Affiche l’HTML généré par Folium
        self.web = QWebEngineView()

        # =========================
        # 4) MISE EN PAGE (LAYOUT)
        # =========================

        # FormLayout pour les filtres (joli alignement label/champ)
        form = QFormLayout()
        form.addRow("Prix ($)", self._row(self.spin_min_price, self.spin_max_price))
        form.addRow("Chambres (Beds)", self._row(self.spin_min_beds, self.spin_max_beds))
        form.addRow("Ville", self.edit_city)
        form.addRow("État", self.combo_state)

        lay = QVBoxLayout(self)
        lay.addLayout(form)

        # Ligne bouton + compteur
        row_btn = QHBoxLayout()
        row_btn.addWidget(self.btn_apply)
        row_btn.addWidget(self.lbl_count)
        row_btn.addStretch(1)
        lay.addLayout(row_btn)

        lay.addWidget(QLabel("Carte des biens filtrés :"))
        lay.addWidget(self.web, stretch=1)

        # Génération initiale de la carte au chargement de l'onglet
        self.update_map()

    def _row(self, *widgets):
        """
        Helper UI : met plusieurs widgets sur la même ligne (HBoxLayout).
        """
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        for wd in widgets:
            h.addWidget(wd)
        return w

    def filtered_df(self):
        """
        Applique les filtres sélectionnés sur le DataFrame et renvoie un df filtré.
        Important : on ne modifie pas self.df, on construit un df temporaire.
        """
        df = self.df

        # Récupération des filtres
        min_price = self.spin_min_price.value()
        max_price = self.spin_max_price.value()
        min_beds = self.spin_min_beds.value()
        max_beds = self.spin_max_beds.value()
        city = self.edit_city.text().strip()
        state = self.combo_state.currentText().strip()

        # Filtre prix
        if "Price" in df.columns:
            if min_price > 0:
                df = df[df["Price"] >= min_price]
            if max_price > 0:
                df = df[df["Price"] <= max_price]

        # Filtre beds
        if "Beds" in df.columns:
            if min_beds > 0:
                df = df[df["Beds"] >= min_beds]
            if max_beds > 0:
                df = df[df["Beds"] <= max_beds]

        # Filtre ville (contient, insensible à la casse)
        if city and "City" in df.columns:
            df = df[df["City"].astype(str).str.contains(city, case=False, na=False)]

        # Filtre état (exact)
        if state and "State" in df.columns:
            df = df[df["State"].astype(str) == state]

        return df

    def update_map(self):
        """
        Régénère la carte Folium à partir du DataFrame filtré.
        Utilise FastMarkerCluster pour la performance.

        Étapes :
        1) Filtrer df
        2) Créer une carte centrée US
        3) Construire listes (points + popups HTML)
        4) Ajouter cluster
        5) Injecter HTML dans QWebEngineView
        """
        try:
            df_f = self.filtered_df()
            self.lbl_count.setText(f"{len(df_f):,} résultats")

            # Carte centrée sur les USA (lat/lon moyens) avec style léger
            m = folium.Map(location=[39.5, -98.35], zoom_start=4, tiles="CartoDB positron")

            # Si df vide ou colonnes coordonnées absentes -> on affiche carte vide
            if df_f.empty or not {"Latitude", "Longitude"}.issubset(df_f.columns):
                self.web.setHtml(m.get_root().render(), baseUrl=QUrl("https://localhost/"))
                return

            points = []
            popups = []

            # Pour chaque ligne filtrée : construire un point + popup
            for _, row in df_f.iterrows():
                try:
                    lat = float(row["Latitude"])
                    lon = float(row["Longitude"])
                except Exception:
                    # Ignore les lignes sans coordonnées valides
                    continue

                # Popup HTML affiché lors du clic sur le marqueur
                html = (
                    f"<b>{row.get('Address','')}</b><br>"
                    f"{row.get('City','')}, {row.get('State','')} ({row.get('Zip Code','')})<br>"
                    f"Price: {fmt_price(row.get('Price', 0))}<br>"
                    f"Beds: {row.get('Beds','?')} | Baths: {row.get('Baths','?')} | "
                    f"Living Space: {row.get('Living Space','?')} ft²"
                )

                points.append([lat, lon])
                popups.append(html)

            # Ajout du cluster rapide (beaucoup de points possible)
            FastMarkerCluster(data=points, popups=popups).add_to(m)

            # Injecte la carte HTML dans la WebView.
            # baseUrl évite certains warnings/limitations de chargement.
            self.web.setHtml(m.get_root().render(), baseUrl=QUrl("https://localhost/"))

        except Exception as e:
            # En cas d'erreur, on évite de faire planter l’UI :
            print(f"[ERREUR update_map] {e}")


class MapTab(QWidget):
    """
    Onglet prêt pour main.py :
    - charge le CSV
    - instancie CartographyDynamic(df)
    """
    def __init__(self):
        super().__init__()

        # Charge le dataset
        df = pd.read_csv(DATA_PATH)

        # Place le widget carto en plein onglet
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(CartographyDynamic(df))
