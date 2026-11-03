"""
App views of the IPO RRG app via Streamlit AppTest, with no exceptions or errors:
  - every tag: its oldest and newest ready stock on Daily, Weekly, Both, and the tag's All stocks page
  - a stock without enough weekly history shows the 'needs more history' notice on Weekly / Both (not an error)
  - the IPO details page, for ALL and for each tag
(Every stock on every page takes hours; tests/test_ipo_stocks_and_tags.py checks every stock's data instead.)
"""
import os
import sys

import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config as cfg

APP = os.path.join(cfg.ROOT_DIR, "rrg_app.py")
IPO_PAGE = os.path.join(cfg.ROOT_DIR, "ipo_page.py")
CASES = []
for tag in cfg.TAGS[1:]:
    members = cfg.stocks_for_tag(tag)
    for stock in {members[0], members[-1]}:
        for page in ("Daily", "Weekly", "Both"):
            CASES.append((tag, stock, page))
    CASES.append((tag, members[0], "All stocks"))


def errors(at):
    return [e.value for e in at.exception] + [e.body for e in at.error]


@pytest.mark.parametrize("tag,stock,page", CASES)
def test_rrg_views(tag, stock, page):
    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    at.selectbox(key="tag").set_value(tag).run()
    at.selectbox(key="stock").set_value(stock).run()
    at.radio(key="page").set_value(page).run()
    assert not errors(at), (tag, stock, page, errors(at))
    if page in ("Weekly", "Both"):
        notice = [w.body for w in at.warning if "Weekly RRG needs more history" in w.body]
        has_chart = len(at.subheader) > 0
        assert notice or has_chart, (stock, page)


@pytest.mark.parametrize("tag", ["ALL"] + cfg.TAGS[1:])
def test_ipo_page(tag):
    at = AppTest.from_file(IPO_PAGE, default_timeout=600)
    at.run()
    at.selectbox(key="ipo_tag").set_value(tag).run()
    assert not errors(at), errors(at)
    assert len(at.dataframe) >= 4
    assert any("UNOFFICIAL" in w.body for w in at.warning)          # GMP always labelled unofficial
