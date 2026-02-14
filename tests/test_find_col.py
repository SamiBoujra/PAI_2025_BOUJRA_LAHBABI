import pandas as pd
from codeAZ.tab_corr import _find_col

def test_find_col_matches_variants():
    df = pd.DataFrame({"Zip_Code": [1], "Sale Price": [10], "CITY": ["X"]})
    assert _find_col(df, ["zip code", "zipcode"]) == "Zip_Code"
    assert _find_col(df, ["price"]) == "Sale Price"
    assert _find_col(df, ["city"]) == "CITY"
