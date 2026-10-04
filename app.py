"""Fashion AI — the single application entry point.

streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from src.ui.pages import diagnostics_page, outfits_page, recommendations_page, wardrobe_page

st.set_page_config(page_title="Fashion AI Wardrobe", page_icon="👗", layout="wide")

navigation = st.navigation(
    [
        st.Page(wardrobe_page, title="Wardrobe", icon="👕", url_path="wardrobe", default=True),
        st.Page(outfits_page, title="Outfit Generator", icon="✨", url_path="outfits"),
        st.Page(recommendations_page, title="Item Recommendations", icon="🔗", url_path="recommendations"),
        st.Page(diagnostics_page, title="System / Diagnostics", icon="🩺", url_path="diagnostics"),
    ]
)
# Always-visible device indicator, so an accidental CPU launch is obvious.
try:
    from src.ui.state import get_service

    _info = get_service().device_info()
    if _info["active"].startswith("cuda"):
        st.sidebar.caption(f"Running on CUDA · {_info['gpu']}")
    else:
        st.sidebar.caption("Running on CPU (slower) · see Diagnostics")
except Exception:  # the pages themselves report service errors
    pass

navigation.run()
