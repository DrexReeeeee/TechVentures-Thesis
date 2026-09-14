import streamlit as st


def render_page_header(
    *,
    title: str,
    subtitle: str,
) -> None:
    """
    Render a consistent page header.
    """

    st.title(title)

    st.caption(subtitle)

    st.write("")