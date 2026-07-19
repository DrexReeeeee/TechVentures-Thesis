"""
Reusable page header component.

This component renders a consistent page title and subtitle
across the entire dashboard.
"""

from __future__ import annotations

import streamlit as st


def render_page_header(
    title: str,
    subtitle: str | None = None,
) -> None:
    """
    Render a reusable page header.

    Parameters
    ----------
    title:
        Main page title.

    subtitle:
        Optional descriptive text shown below the title.
    """

    st.markdown(
        f"""
        <div class="page-header">

            <h1 class="page-title">
                {title}
            </h1>

            {
                f'<p class="page-subtitle">{subtitle}</p>'
                if subtitle
                else ""
            }

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()