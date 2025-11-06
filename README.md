# 🏙️ USCities House Prices — PAI 2025-2026

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](#)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red?logo=streamlit)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

> **Authors:** [Sami BOUJRA](#) · [Khalid LAHBABI](#)  
> **Project:** PAI 2025–2026  
> **Title:** *Application de visualisation et d’analyse des prix immobiliers dans les grandes villes américaines*

---

## 📚 Table of Contents
- [🎯 Overview](#-overview)
- [🗂 Dataset](#-dataset)
- [✨ Features](#-features)
- [🧰 Tech Stack](#-tech-stack)
- [🚀 Installation](#-installation)
- [🏃‍♂️ Usage](#-usage)
- [📊 App Structure](#-app-structure)
- [⚙️ Configuration](#️-configuration)
- [📈 Machine Learning Module](#-machine-learning-module)
- [📦 Repository Layout](#-repository-layout)
- [👥 Authors](#-authors)
- [📝 License](#-license)

---

## 🎯 Overview

This project is an **interactive data visualization and analysis web app** that explores the **real estate market** in the **50 largest U.S. cities**.  
It integrates **data science**, **geospatial visualization**, and **machine learning** to study how factors like **median income, population density**, and **city size** affect **house prices**.

🧠 **Goal:** Provide a user-friendly tool to visualize housing trends and predict property values based on key features.

---

## 🗂 Dataset

- **Source:** [Kaggle – American House Prices and Demographics of Top Cities](https://www.kaggle.com/datasets/jeremylarcher/american-house-prices-and-demographics-of-top-cities)
- **Format:** CSV  
- **Scope:** 50 major U.S. cities  
- **Records:** Several thousand entries  
- **Attributes:**
  - `Zip Code`, `City`, `State`, `County`
  - `Price`, `Beds`, `Baths`, `Living Space`
  - `Median Income`, `Population`, `Density`
  - `Latitude`, `Longitude`

---

## ✨ Features

| Module | Description |
|:--|:--|
| **Exploration** | Filter and sort property listings by price, area, rooms, city, or income. |
| **Cartography** | Interactive map of properties using Folium or Leafmap. |
| **Correlations** | Visualize relationships (e.g. price vs income) with scatterplots & heatmaps. |
| **Prediction** | Estimate property price using a trained ML model. |
| **Statistics** | View summary indicators by city or ZIP. |
| **Export** | Download filtered data or charts (CSV, PNG). |

---

## 🧰 Tech Stack

| Layer | Tools |
|:--|:--|
| **Frontend / App** | Streamlit, Plotly, Folium / Leafmap |
| **Data Handling** | Pandas, NumPy, GeoPandas |
| **Machine Learning** | Scikit-learn (Random Forest baseline) |
| **Visualization** | Plotly Express, Matplotlib |
| **Environment** | Python 3.10+, `.env` for configuration |

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/USCities-HousePrices.git
cd USCities-HousePrices

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Prepare data
mkdir -p data/raw
# Put the Kaggle dataset at: data/raw/american_house_prices.csv

# 5. (Optional) Train your model
python src/train.py

# 6. Launch the app 🚀
streamlit run app.py
