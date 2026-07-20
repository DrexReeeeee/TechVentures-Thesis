import streamlit as st

from components.layout.page_header import render_page_header
from components.layout.section_header import render_section_header
from components.display.preset_card import render_preset_card


def render():

    render_page_header(
        title="Predictor Dashboard",
        subtitle="Predict startup success using the Spatial Feature Transformation Neural Network.",
    )

    render_section_header(
        title="Preset Models",
        subtitle="Quickly evaluate representative startup scenarios.",
    )

    col1, col2, col3 = st.columns(
        3,
        gap="large",
    )

    with col1:

        render_preset_card(
            title="Southeast Asia SaaS",
            subtitle="High growth potential",
            icon="apartment",
            key="preset_sea",
        )

    with col2:

        render_preset_card(
            title="Africa FinTech",
            subtitle="Emerging opportunity",
            icon="public",
            key="preset_africa",
        )

    with col3:

        render_preset_card(
            title="North America AI",
            subtitle="Established market",
            icon="analytics",
            key="preset_na",
        )

    st.divider()

    render_section_header(
        title="Input Mode",
    )

    st.container(border=True).info(
        "Startup input form will be added here."
    )

    st.divider()

    render_section_header(
        title="Prediction Results",
    )

    st.container(border=True).info(
        "Prediction results will appear here."
    )

    st.divider()

    render_section_header(
        title="Explainability",
    )

    st.container(border=True).info(
        "SHAP and Explainable AI visualization."
    )

    st.divider()

    render_section_header(
        title="System Integrity",
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Dataset", "SEA-2026")

    with c2:
        st.metric("Model", "SFTNN")

    with c3:
        st.metric("Status", "Ready")