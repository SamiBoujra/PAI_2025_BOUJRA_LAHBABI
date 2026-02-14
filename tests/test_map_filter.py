import pytest

QtWebEngine = pytest.importorskip("PySide6.QtWebEngineWidgets")

import codeAZ.tab_map as tab_map

def test_map_filtered_df(qapp, monkeypatch, sample_housing_df):
    # Prevent update_map from running during init (avoids folium rendering too)
    monkeypatch.setattr(tab_map.CartographyDynamic, "update_map", lambda self: None)

    w = tab_map.CartographyDynamic(sample_housing_df)

    w.spin_min_price.setValue(300000)
    w.spin_max_price.setValue(700000)
    w.edit_city.setText("port")
    w.combo_state.setCurrentText("Oregon")

    out = w.filtered_df()
    assert out.shape[0] == 1
