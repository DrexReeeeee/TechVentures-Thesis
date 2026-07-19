from __future__ import annotations

import streamlit as st


def render_nav_item(
    *,
    label: str,
    icon: str,
    active: bool,
    key: str,
) -> bool:
    """
    Render one sidebar navigation item.
    """

    return st.button(
        label,
        icon=f":material/{icon}:",
        key=key,
        use_container_width=True,
        type="primary" if active else "secondary",
    )