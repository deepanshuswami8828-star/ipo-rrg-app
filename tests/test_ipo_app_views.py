"""
App views of the IPO RRG app (four tabs) via Streamlit AppTest: no exceptions, no errors.
  - Home opens from every possible main file (rrg_app.py, ipo_page.py, streamlit_app.py) with its table of every IPO
  - Stock: an old IPO, one with a split, one renamed after listing, a waiting one (no RRG yet), and the oldest and
    newest ready stock of every tag
  - Compare: every tag and ALL
  - Upcoming: GMP always labelled unofficial
"""
import os
import sys

import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config as cfg

ROOT = cfg.ROOT_DIR


def errors(at):
    return [e.value for e in at.exception] + [e.body for e in at.error]


def stock_script(ticker):
    import streamlit as st
    import app_main
    st.session_state["stock_pick"] = ticker
    app_main.stock_page()


def compare_script(tag):
    import streamlit as st
    import app_main
    st.session_state["cmp_tag"] = tag
    app_main.compare_page()


def upcoming_script():
    import app_main
    app_main.upcoming_page()


@pytest.mark.parametrize("entry", ["rrg_app.py", "ipo_page.py", "streamlit_app.py"])
def test_home_from_every_entry_file(entry):
    at = AppTest.from_file(os.path.join(ROOT, entry), default_timeout=600)
    at.run()
    assert not errors(at), errors(at)
    assert at.title[0].value == "📈 IPO RRG"
    assert at.dataframe[0].value.shape[0] == len(cfg.ALL_STOCKS)        # every IPO in the Home table


STOCK_CASES = ["SWIGGY.NS", "CORDELIA.NS", "SETL.NS", next(iter(cfg.WAITING))]
for _tag in cfg.TAGS[1:]:
    _m = cfg.stocks_for_tag(_tag)
    STOCK_CASES += [_m[0], _m[-1]]


@pytest.mark.parametrize("ticker", sorted(set(STOCK_CASES)))
def test_stock_tab(ticker):
    at = AppTest.from_function(stock_script, args=(ticker,), default_timeout=600)
    at.run()
    assert not errors(at), (ticker, errors(at))
    charts = len(at.get("plotly_chart"))
    if ticker in cfg.STOCKS:
        assert charts == 2, (ticker, charts)              # RRG chart + price chart
    else:
        assert charts == 1 and any("RRG starts after" in i.body for i in at.info), ticker
    assert len(at.dataframe) == 1                          # IPO details with sources


@pytest.mark.parametrize("tag", cfg.TAGS[1:] + ["ALL"])
def test_compare_tab(tag):
    at = AppTest.from_function(compare_script, args=(tag,), default_timeout=600)
    at.run()
    assert not errors(at), errors(at)
    assert len(at.get("plotly_chart")) == 1


def test_upcoming_tab():
    at = AppTest.from_function(upcoming_script, default_timeout=600)
    at.run()
    assert not errors(at), errors(at)
    assert any("UNOFFICIAL" in w.body for w in at.warning)
    assert len(at.dataframe) >= 2                       # GMP (not listed / listed) + latest listings
