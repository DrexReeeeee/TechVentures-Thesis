from pathlib import Path

import streamlit as st

from components.navigation.nav_item import render_nav_item


LOGO = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "icons"
    / "default_telp.png"
)

PAGES = [
    ("Predictor", "automation"),
    ("Research", "book_5"),
    ("About", "page_info"),
]


def render_sidebar() -> None:
    """
    Render the application sidebar.
    Handles navigation state internally.
    """

    if "page" not in st.session_state:
        st.session_state.page = "Predictor"

    with st.sidebar:

        # -------------------------
        # Brand
        # -------------------------

        left, right = st.columns([1, 3], gap="small")

        with left:
            st.image(str(LOGO), width=58)

        with right:
            st.markdown(
                """
                <div class="sidebar-brand-text">
                    <div class="sidebar-title">
                        TechVenture
                    </div>
                    <div class="sidebar-subtitle">
                        SFTNN
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # -------------------------
        # Navigation
        # -------------------------

        with st.container(key="sidebar_nav"):

            for page, icon in PAGES:

                active = page == st.session_state.page

                clicked = render_nav_item(
                    label=page,
                    icon=icon,
                    active=active,
                    key=f"nav_{page}",
                )

                if clicked and not active:
                    st.session_state.page = page
                    st.rerun()