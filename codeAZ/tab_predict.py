# tab_predict.py

import joblib
# joblib sert à charger/sauvegarder des objets Python (ici : pipeline ML + métadonnées)

import pandas as pd
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QLabel, QPushButton, QGroupBox, QMessageBox
)
from PySide6.QtCore import Qt

# Fonctions "métier" venant de ton module model.py :
# - parse_us_address : extrait City/State/Zip d'une adresse
# - predict_interval_from_constraints : retourne un intervalle de prix (low/med/high)
from codeAZ.model import parse_us_address, predict_interval_from_constraints

# ⚠️ Chemins absolus (spécifiques à TON PC) : ça marche chez toi, mais pas sur un autre poste.
MODEL_PATH = r"C:\Users\pc\Downloads\projet PAI\housing_pipe.joblib"
DATA_PATH  = r"C:\Users\pc\Downloads\projet PAI\projet PAI\American_Housing_Data_20231209.csv"


def _money(x):
    """
    Formate une valeur numérique en euros/dollars style "123 456 $".
    Si x est None => affiche un tiret.
    """
    return "—" if x is None else f"{x:,.0f} $"


def _opt_int(s):
    """
    Convertit un texte en int si possible.
    Retourne None si vide ou invalide.
    Exemple:
      "" -> None
      "  " -> None
      "12" -> 12
      "abc" -> None
    """
    try:
        return int(s) if s.strip() else None
    except:
        return None


class PredictorTab(QWidget):
    """
    Onglet "Prediction" :
    - charge le modèle ML (pipeline) depuis un fichier joblib
    - propose une interface utilisateur pour entrer des contraintes (ville, état, etc.)
    - calcule une prédiction sous forme d'intervalle (low/median/high)
    """

    def __init__(self):
        super().__init__()

        # =========================
        # 1) CHARGEMENT DU MODÈLE
        # =========================

        # bundle contient ce que tu as sauvegardé dans model.py:
        # {"pipe": pipe, "feature_cols": feature_cols, "use_log": use_log}
        bundle = joblib.load(MODEL_PATH)

        self.pipe = bundle["pipe"]              # pipeline sklearn: preprocess + modèle xgboost
        self.feature_cols = bundle["feature_cols"]  # liste des features attendues
        self.use_log = bundle["use_log"]        # indique si le modèle prédit log(price)

        # =========================
        # 2) CHARGEMENT DU DATASET
        # =========================
        # Utilisé pour "predict_interval_from_constraints" (échantillonnage de lignes plausibles)
        self.df = pd.read_csv(DATA_PATH)

        # On retire les lignes incomplètes sur les colonnes clés (minimiser les erreurs)
        self.df = self.df.dropna(subset=["Price", "Beds", "Baths", "Living Space", "City", "State"])

        # =========================
        # 3) CONSTRUCTION UI
        # =========================
        root = QVBoxLayout(self)

        # -------------------------
        # (A) Adresse optionnelle
        # -------------------------
        # L'utilisateur peut taper une adresse complète, cliquer "Parse",
        # et City/State/Zip seront remplis automatiquement.
        addr_box = QGroupBox("Address (optional)")
        addr_lay = QHBoxLayout(addr_box)

        self.addr = QLineEdit()
        self.addr.setPlaceholderText("1600 Pennsylvania Avenue NW, Washington, DC")

        btn_parse = QPushButton("Parse")
        btn_parse.clicked.connect(self.parse_address)

        addr_lay.addWidget(self.addr)
        addr_lay.addWidget(btn_parse)
        root.addWidget(addr_box)

        # -------------------------
        # (B) Formulaire inputs
        # -------------------------
        form_box = QGroupBox("Inputs")
        form = QFormLayout(form_box)

        # Champs utilisateur
        self.city = QLineEdit()
        self.state = QLineEdit()
        self.zipc = QLineEdit()
        self.beds = QLineEdit()
        self.baths = QLineEdit()
        self.space = QLineEdit()

        # Ajout dans le formulaire
        # City et State marqués * car obligatoires
        form.addRow("City*", self.city)
        form.addRow("State*", self.state)
        form.addRow("Zip Code", self.zipc)
        form.addRow("Beds", self.beds)
        form.addRow("Baths", self.baths)
        form.addRow("Living Space", self.space)

        root.addWidget(form_box)

        # -------------------------
        # (C) Bouton Predict
        # -------------------------
        row = QHBoxLayout()
        btn = QPushButton("Predict")
        btn.clicked.connect(self.predict)
        row.addStretch()
        row.addWidget(btn)
        root.addLayout(row)

        # -------------------------
        # (D) Zone résultats
        # -------------------------
        out = QGroupBox("Result")
        f = QFormLayout(out)

        self.lbl_med = QLabel("—")
        self.lbl_low = QLabel("—")
        self.lbl_high = QLabel("—")

        f.addRow("Median", self.lbl_med)
        f.addRow("Low (80%)", self.lbl_low)
        f.addRow("High (80%)", self.lbl_high)

        root.addWidget(out)

    # =========================
    # ACTION 1: PARSER ADRESSE
    # =========================
    def parse_address(self):
        """
        Utilise parse_us_address() (module model.py) pour extraire:
        - City
        - State
        - Zip Code
        depuis l'adresse libre saisie.
        """
        try:
            info = parse_us_address(self.addr.text())

            # Remplit automatiquement les champs correspondants
            self.city.setText(info["City"])
            self.state.setText(info["State"])
            self.zipc.setText(info["Zip Code"])

        except Exception as e:
            # Erreur de parsing -> popup
            QMessageBox.critical(self, "Parse error", str(e))

    # =========================
    # ACTION 2: PREDICTION
    # =========================
    def predict(self):
        """
        - Vérifie que City et State existent
        - Construit le dictionnaire constraints
        - Appelle predict_interval_from_constraints()
        - Met à jour l'affichage (low/median/high)
        """

        # Garde-fou : City/State obligatoires
        if not self.city.text() or not self.state.text():
            QMessageBox.warning(self, "Error", "City and State required")
            return

        # Dictionnaire de contraintes.
        # Les champs vides deviennent None.
        # Beds/Baths/Living Space sont convertis en int (ou None si vide/invalide).
        constraints = {
            "City": self.city.text(),
            "State": self.state.text(),
            "Zip Code": self.zipc.text() or None,
            "Beds": _opt_int(self.beds.text()),
            "Baths": _opt_int(self.baths.text()),
            "Living Space": _opt_int(self.space.text()),
        }

        # Appel de la fonction qui:
        # - filtre des lignes proches dans df
        # - échantillonne
        # - fait prédire une distribution
        # - renvoie quantiles (low, median, high)
        low, med, high, _ = predict_interval_from_constraints(
            self.pipe,
            self.df,
            self.feature_cols,
            constraints,
            use_log=self.use_log
        )

        # Mise à jour de l'UI avec format monétaire
        self.lbl_med.setText(_money(med))
        self.lbl_low.setText(_money(low))
        self.lbl_high.setText(_money(high))
