# 🏙️ US Real Estate Dashboard — PAI 2025–2026

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)](#)
[![PySide6](https://img.shields.io/badge/UI-PySide6-green)](#)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-success)](#)
[![Tests](https://img.shields.io/badge/Tested%20with-pytest-orange)](#)

> **Authors:** Sami BOUJRA · Khalid LAHBABI  
> **Project:** PAI 2025–2026  
> **Title:** Application de visualisation et d’analyse des prix immobiliers dans les grandes villes américaines

---

## 📌 Overview

This project is a **desktop data analysis application** built with **PySide6 (Qt)** that allows interactive exploration and prediction of real estate prices across major U.S. cities.

It combines:

- 📊 Data analysis  
- 🗺️ Geospatial visualization  
- 📈 Statistical correlation analysis  
- 🤖 Machine learning price prediction  
- ✅ Continuous Integration with automated testing  

---

## 📊 Dataset

- **Source:** Kaggle – American House Prices & Demographics  
- **Format:** CSV  
- **Attributes include:**
  - City, State, Zip Code  
  - Price  
  - Beds / Baths  
  - Living Space  
  - Median Household Income  
  - Latitude / Longitude  

The dataset is stored locally and loaded at runtime.

---

## 🧠 Features

### 🔎 Exploration Tab
- Dynamic filtering by:
  - Price range
  - Beds
  - Living space
  - City
  - State
  - Median income
- Sortable table view
- CSV export of filtered results

---

### 🗺️ Cartography Tab
- Interactive map using **Folium**
- Fast marker clustering
- Live filtering
- Displays:
  - Address
  - Price
  - Beds / Baths
  - Living Space

---

### 📈 Correlation Tab
- Pearson correlation computation
- Scatter plot visualization
- Sampling support
- Summary statistics:
  - Mean price by city
  - Mean price by ZIP
  - Mean price by income bracket

---

### 🤖 Prediction Tab
- Address parsing via `usaddress`
- ML model trained with:
  - XGBoost
  - Scikit-learn pipeline
- Returns:
  - Median predicted price
  - 80% prediction interval
  - Number of comparable rows used

---

## 🛠 Tech Stack

| Layer | Technology |
|--------|------------|
| UI | PySide6 (Qt) |
| Data | Pandas, NumPy |
| ML | Scikit-learn, XGBoost |
| Mapping | Folium |
| Visualization | Matplotlib |
| Address Parsing | usaddress |
| Testing | pytest, pytest-qt |
| CI | GitHub Actions |

---

## 🚀 Installation

### 1️⃣ Clone repository

```bash
git clone https://github.com/your-username/PAI_2025_BOUJRA_LAHBABI.git
cd PAI_2025_BOUJRA_LAHBABI
```

### 2️⃣ Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
.venv\Scripts\activate         # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run application

```bash
python codeAZ/apli.py
```

---

## 🧪 Testing

Run locally:

```bash
python -m pytest
```

CI automatically runs tests on:

- Push  
- Pull Request  

Workflow file:

```
.github/workflows/CI.yaml
```

---

## 📂 Project Structure

```
PROJET PAI/
│
├── codeAZ/
│   ├── model.py
│   ├── tab_corr.py
│   ├── tab_explore.py
│   ├── tab_map.py
│   ├── tab_predict.py
│   └── apli.py
│
├── tests/
│   ├── test_helpers.py
│   ├── test_find_col.py
│   ├── test_map_filter.py
│   └── conftest.py
│
├── requirements.txt
├── README.md
└── .github/workflows/CI.yaml
```

---

## 📈 Machine Learning

Model:
- `XGBRegressor`
- Log-price transformation
- Prediction interval via bootstrapped sampling

Metrics:
- MAE
- RMSE
- R²

Model saved as:

```
housing_pipe.joblib
```

---

## 🔒 Continuous Integration

CI ensures:

- All modules import correctly  
- Filtering logic behaves correctly  
- Helper functions validate input properly  
- Map filtering works in headless Qt mode  

Runs on:
- Ubuntu  
- Python 3.13  

---

## 👥 Authors

| Name | Role |
|------|------|
| Sami BOUJRA | Data & Visualization |
| Khalid LAHBABI | Machine Learning & Integration |

---

## 📜 License

MIT License

---

> PAI 2025–2026 — Real Estate Data Analysis Dashboard
