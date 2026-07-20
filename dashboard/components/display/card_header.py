import streamlit as st


def render_card_header(
    *,
    icon: str,
    title: str,
    subtitle: str,
) -> None:
    """
    Render the reusable card header.
    """

    st.markdown(
        f"""
<div class="card-header">

    <div class="card-icon">

        <span class="material-symbols-outlined">
            {icon}
        </span>

    </div>

    <div class="card-text">

        <div class="card-title">
            {title}
        </div>

        <div class="card-subtitle">
            {subtitle}
        </div>

    </div>

</div>
""",
        unsafe_allow_html=True,
    )