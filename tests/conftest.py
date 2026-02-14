import pandas as pd
import pytest

@pytest.fixture
def sample_housing_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Price": [250000, 400000, 600000, 800000],
            "Beds": [2, 3, 4, 4],
            "Baths": [1, 2, 3, 3],
            "Living Space": [900, 1400, 2100, 2600],
            "City": ["Boston", "Boston", "Portland", "Portland"],
            "State": ["MA", "MA", "Oregon", "Oregon"],
            "Zip Code": ["02127", "02128", "97205", "97206"],
            "Latitude": [42.33, 42.36, 45.52, 45.55],
            "Longitude": [-71.05, -71.00, -122.67, -122.64],
            "Address": ["A", "B", "C", "D"],
        }
    )
