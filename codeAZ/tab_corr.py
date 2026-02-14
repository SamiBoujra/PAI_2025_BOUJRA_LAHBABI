# tab_corr.py
from __future__ import annotations
# Permet d’utiliser des annotations de types avancées (ex: list[str]) sans soucis
# selon la version Python / interprétation future.

from pathlib import Path
# Pour gérer des chemins de fichiers de manière portable (Windows/Linux/Mac)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QSpinBox, QCheckBox, QGroupBox, QPlainTextEdit
)
# Widgets Qt utilisés pour construire l'UI:
# - QWidget : base d’un onglet
# - layouts : organisation verticale/horizontale
# - QLabel : texte simple
# - QComboBox : listes déroulantes
# - QPushButton : boutons
# - QSpinBox : sélecteur d’entier
# - QCheckBox : option booléenne
# - QGroupBox : section encadrée
# - QPlainTextEdit : zone texte multi-ligne (rapport KPI)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# Canvas Matplotlib intégré dans une UI Qt (affichage du graphe dans l’onglet)


# ✅ Chemin CSV portable : relatif au projet (évite les chemins absolus type C:\...)
# Hypothèse : le CSV est à la racine du projet : American_Housing_Data_20231209.csv
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "American_Housing_Data_20231209.csv"


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Tente de retrouver une colonne du DataFrame correspondant à une liste de noms candidats.

    Objectif:
    - Rendre le code robuste aux variations de noms (ex: "Zip Code" vs "zipcode" vs "postal code")
    - Recherche insensible à la casse
    - Tolère espaces / underscores

    Retour:
    - Nom exact de la colonne trouvée dans df
    - None si aucune correspondance
    """
    cols = list(df.columns)

    # Normalisation des colonnes du CSV : minuscules + suppression espaces/underscore
    norm = {c: "".join(str(c).lower().replace("_", " ").split()) for c in cols}

    # Normalisation des candidats de la même façon
    cand_norm = ["".join(x.lower().replace("_", " ").split()) for x in candidates]

    # 1) Match exact sur version normalisée
    for c in cols:
        if norm[c] in cand_norm:
            return c

    # 2) Match "contains" (si le candidat est contenu dans le nom normalisé de la colonne)
    for c in cols:
        for cn in cand_norm:
            if cn and cn in norm[c]:
                return c

    return None


class CorrelationTab(QWidget):
    """
    Onglet "Correlation" avec deux parties principales :

    ✅ (E6) Corrélation :
      - L’utilisateur choisit X et Y parmi les colonnes numériques
      - Calcul de la corrélation de Pearson (r)
      - Option : affichage nuage de points

    ✅ (E8) Indicateurs statistiques :
      - Prix moyen global
      - Top N villes par prix moyen
      - Top N zip codes par prix moyen
      - Prix moyen par tranches de revenu médian (bins)
    """

    def __init__(self):
        super().__init__()

        # -------------------------
        # CHARGEMENT DU CSV
        # -------------------------
        # On charge une seule fois à l'initialisation de l’onglet.
        if not DATA_PATH.exists():
            # Si le fichier n'existe pas : crash contrôlé avec message explicite.
            raise FileNotFoundError(
                f"CSV not found: {DATA_PATH}\n"
                f"Put American_Housing_Data_20231209.csv at project root: {PROJECT_ROOT}"
            )

        self.df = pd.read_csv(DATA_PATH)

        # -------------------------
        # DÉTECTION ROBUSTE DES COLONNES (E8)
        # -------------------------
        # On essaie de retrouver automatiquement les colonnes importantes, même si le CSV
        # n’utilise pas les mêmes noms exacts que le code.
        self.col_price = _find_col(self.df, ["price", "saleprice", "sale price", "listprice", "list price"])
        self.col_city = _find_col(self.df, ["city", "ville"])
        self.col_zip = _find_col(self.df, ["zip", "zipcode", "zip code", "postal", "postalcode", "postal code"])
        self.col_income = _find_col(self.df, ["medianincome", "median income", "income", "revenu", "revenu median"])

        # -------------------------
        # DATAFRAME NUMÉRIQUE POUR LA CORRÉLATION (E6)
        # -------------------------
        # select_dtypes inclut seulement les colonnes déjà reconnues comme numériques.
        self.num = self.df.select_dtypes(include=["number"]).copy()
        self.cols = list(self.num.columns)

        # Layout principal (vertical)
        root = QVBoxLayout(self)

        # ==========================================================
        # (A) ZONE CORRÉLATION (E6)
        # ==========================================================
        corr_group = QGroupBox("Corrélations (E6)")
        corr_layout = QVBoxLayout(corr_group)

        # ---- Barre de contrôle (choix X, Y, taille échantillon, scatter on/off)
        ctrl = QHBoxLayout()

        # ComboBox X : liste des colonnes numériques
        self.cb_x = QComboBox()
        self.cb_x.addItems(self.cols)

        # ComboBox Y : idem
        self.cb_y = QComboBox()
        self.cb_y.addItems(self.cols)
        self.cb_y.setCurrentIndex(1 if len(self.cols) > 1 else 0)  # évite X=Y par défaut si possible

        # QSpinBox : taille max de l’échantillon
        # But : éviter un scatter trop lourd si beaucoup de données
        self.spin_sample = QSpinBox()
        self.spin_sample.setRange(1000, 200000)
        self.spin_sample.setValue(20000)

        # Checkbox : afficher / masquer le nuage de points
        self.chk_scatter = QCheckBox("Afficher nuage de points")
        self.chk_scatter.setChecked(True)

        # Bouton : déclenche le calcul et l'affichage
        self.btn_corr = QPushButton("Calculer corrélation")
        self.btn_corr.clicked.connect(self.update_plot)

        # Ajout des widgets dans la barre
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

        # Label résultat : affiche r et n
        self.lbl_corr = QLabel("Cliquez sur « Calculer corrélation »")
        self.lbl_corr.setStyleSheet("font-size: 14px;")
        corr_layout.addWidget(self.lbl_corr)

        # Figure Matplotlib + Canvas Qt
        self.fig = plt.figure(figsize=(7, 5))
        self.canvas = FigureCanvas(self.fig)
        corr_layout.addWidget(self.canvas)

        root.addWidget(corr_group)

        # ==========================================================
        # (B) ZONE KPI / INDICATEURS (E8)
        # ==========================================================
        kpi_group = QGroupBox("Indicateurs statistiques (E8)")
        kpi_layout = QVBoxLayout(kpi_group)

        kpi_ctrl = QHBoxLayout()

        # Nombre d’éléments à afficher dans les Tops (Top N)
        self.spin_topn = QSpinBox()
        self.spin_topn.setRange(3, 50)
        self.spin_topn.setValue(10)

        # Bouton calcul KPI
        self.btn_kpi = QPushButton("Calculer indicateurs")
        self.btn_kpi.clicked.connect(self.update_kpis)

        kpi_ctrl.addWidget(QLabel("Top N:"))
        kpi_ctrl.addWidget(self.spin_topn)
        kpi_ctrl.addWidget(self.btn_kpi)
        kpi_ctrl.addStretch(1)

        kpi_layout.addLayout(kpi_ctrl)

        # Zone texte rapport
        self.kpi_text = QPlainTextEdit()
        self.kpi_text.setReadOnly(True)
        self.kpi_text.setStyleSheet("font-family: Consolas, monospace; font-size: 12px;")
        self.kpi_text.setPlaceholderText("Cliquez sur « Calculer indicateurs »")
        kpi_layout.addWidget(self.kpi_text)

        root.addWidget(kpi_group)

        # Note : on ne lance rien automatiquement pour éviter de ralentir le démarrage.


    # ==========================================================
    # E6 : CALCUL CORRÉLATION + AFFICHAGE
    # ==========================================================
    def update_plot(self):
        """
        - Récupère les colonnes X et Y choisies.
        - Nettoie les NaN
        - Option : échantillonne pour performance
        - Calcule Pearson r
        - Affiche soit scatter, soit texte
        """
        x = self.cb_x.currentText()
        y = self.cb_y.currentText()

        # Garde-fou : pas de corrélation si même colonne
        if x == y:
            self.lbl_corr.setText("Choisissez deux colonnes différentes.")
            return

        # On ne garde que les 2 colonnes, puis suppression NaN
        df_s = self.num[[x, y]].dropna()
        if len(df_s) == 0:
            self.lbl_corr.setText("Aucune ligne valide après dropna().")
            return

        # Échantillonnage pour limiter le volume
        n = min(self.spin_sample.value(), len(df_s))
        if len(df_s) > n:
            df_s = df_s.sample(n=n, random_state=42)

        # Conversion forcée en numérique (sécurité si colonnes mixtes)
        xs = pd.to_numeric(df_s[x], errors="coerce")
        ys = pd.to_numeric(df_s[y], errors="coerce")
        m = xs.notna() & ys.notna()
        xs = xs[m]
        ys = ys[m]

        # Minimum de points pour corrélation fiable
        if len(xs) < 3:
            self.lbl_corr.setText("Pas assez de points pour calculer la corrélation.")
            return

        # Corrélation de Pearson
        r = float(xs.corr(ys))
        self.lbl_corr.setText(f"Pearson r({x}, {y}) = {r:.4f}   (n={len(xs):,})")

        # Nettoie la figure puis trace
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        if self.chk_scatter.isChecked():
            # Scatter peut être lourd -> on limite à 15000 points max
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
            # Si scatter désactivé -> simple affichage du résultat
            ax.text(
                0.5, 0.5,
                f"Pearson r = {r:.4f}\n(n={len(xs):,})",
                ha="center", va="center",
                transform=ax.transAxes
            )
            ax.set_axis_off()

        self.fig.tight_layout()
        self.canvas.draw()


    # ==========================================================
    # E8 : INDICATEURS / KPI
    # ==========================================================
    def update_kpis(self):
        """
        Produit un rapport texte :
        - Prix moyen global
        - Top N villes (prix moyen)
        - Top N codes postaux (prix moyen)
        - Prix moyen par tranche de revenu médian (bins)
        """
        topn = int(self.spin_topn.value())

        # Vérifie si les colonnes essentielles ont été détectées
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
            # Message utilisateur + suggestion corrective
            self.kpi_text.setPlainText(
                "Impossible de calculer E8 : colonnes non trouvées automatiquement.\n"
                f"Manquantes: {', '.join(missing)}\n\n"
                "➡️ Solution: vérifie les noms exacts des colonnes dans le CSV et\n"
                "mets-les en dur dans le code (col_price/col_city/col_zip/col_income)."
            )
            return

        df = self.df.copy()

        # Nettoyage : conversion en numérique (si prix ou revenu sont stockés en texte)
        df[self.col_price] = pd.to_numeric(df[self.col_price], errors="coerce")
        df[self.col_income] = pd.to_numeric(df[self.col_income], errors="coerce")

        # On garde les lignes où le prix est exploitable
        df = df.dropna(subset=[self.col_price])
        if len(df) == 0:
            self.kpi_text.setPlainText("Aucune valeur de prix exploitable (après nettoyage).")
            return

        # KPI 0 : prix moyen global
        mean_price = df[self.col_price].mean()

        # KPI 1 : prix moyen par ville (Top N)
        by_city = (
            df.dropna(subset=[self.col_city])
              .groupby(self.col_city)[self.col_price]
              .mean()
              .sort_values(ascending=False)
              .head(topn)
        )

        # KPI 2 : prix moyen par code postal (Top N)
        by_zip = (
            df.dropna(subset=[self.col_zip])
              .groupby(self.col_zip)[self.col_price]
              .mean()
              .sort_values(ascending=False)
              .head(topn)
        )

        # KPI 3 : prix moyen par tranche de revenu médian
        # Bins fixés "génériques" (à adapter si ton dataset est différent)
        bins = [0, 40000, 60000, 80000, 100000, 150000, 250000, float("inf")]
        labels = ["<40k", "40-60k", "60-80k", "80-100k", "100-150k", "150-250k", "250k+"]

        df_income = df.dropna(subset=[self.col_income]).copy()
        if len(df_income) > 0:
            # On classe chaque ligne dans une tranche de revenu
            df_income["Income_Bin"] = pd.cut(
                df_income[self.col_income],
                bins=bins,
                labels=labels,
                right=False  # intervalle [a, b)
            )

            by_income = (
                df_income.dropna(subset=["Income_Bin"])
                         .groupby("Income_Bin")[self.col_price]
                         .mean()
                         .sort_values(ascending=False)
            )
        else:
            by_income = pd.Series(dtype=float)

        # Construction d’un rapport texte lisible
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

        # Affichage dans la zone texte
        self.kpi_text.setPlainText("\n".join(lines))
