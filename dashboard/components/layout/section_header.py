import streamlit as st


def render_section_header(
    *,
    title: str,
    subtitle: str | None = None,
) -> None:
    """
    Reusable section header.
    """

    st.subheader(title)

    if subtitle:
        st.caption(subtitle)