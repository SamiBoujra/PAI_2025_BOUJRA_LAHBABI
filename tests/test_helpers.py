import pandas as pd

from codeAZ.tab_predict import _money, _opt_int
from codeAZ.tab_map import fmt_price
from codeAZ.tab_corr import _find_col



def test_opt_int():
    assert _opt_int("") is None
    assert _opt_int("   ") is None
    assert _opt_int("12") == 12
    assert _opt_int("abc") is None


def test_money_format():
    assert _money(None) == "—"
    assert _money(1234.56).endswith("$")
    assert "1" in _money(1234.56)


def test_fmt_price():
    assert fmt_price(1234) == "$1,234"
    assert fmt_price("999") == "$999"
    assert fmt_price(None) == "$0"


def test_find_col_loose_matching():
    df = pd.DataFrame({"Zip_Code": [1], "Sale Price": [10], "CITY": ["X"]})
    assert _find_col(df, ["zip code", "zipcode"]) == "Zip_Code"
    assert _find_col(df, ["price"]) == "Sale Price"
    assert _find_col(df, ["city"]) == "CITY"
