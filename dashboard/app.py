from pathlib import Path

import streamlit as st

from components.layout.sidebar import render_sidebar

from views.predictor import render as predictor_page
from views.research import render as research_page
from views.about import render as about_page


# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="TechVenture",
    page_icon="📈",
    layout="wide",
)


# --------------------------------------------------
# Load CSS
# --------------------------------------------------

CSS_DIR = Path(__file__).parent / "assets" / "css"

for css in (
    "root.css",
    "components.css",
    "navigation.css",
    "sidebar.css",
):
    with open(CSS_DIR / css, encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True,
        )


# --------------------------------------------------
# Session State
# --------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "Predictor"


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

render_sidebar()


# --------------------------------------------------
# Routing
# --------------------------------------------------

ROUTES = {
    "Predictor": predictor_page,
    "Research": research_page,
    "About": about_page,
}

ROUTES.get(
    st.session_state.page,
    predictor_page,
)()