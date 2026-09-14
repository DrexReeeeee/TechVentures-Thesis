import streamlit as st

from components.layout.page_header import render_page_header
from components.layout.section_header import render_section_header
from components.display.cards.preset_card import render_preset_card
from components.common.startup_form import render_startup_form
from components.common.upload_zone import render_upload_zone
from data.fake_data import get_prediction_result
from components.display.cards.model_result_card import (
    render_model_result_card,
)
from components.display.cards.comparison_card import (
    render_comparison_card,
)
from data.fake_data import get_explainability

from components.display.explainability.regional_adaptation import (
    render_regional_heatmap,
)

from components.display.explainability.feature_attribution import (
    render_integrated_gradients,
)

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
        subtitle="Choose how you'd like to provide startup data.",
    )

    mode = st.segmented_control(
        "",
        options=[
            "Single Startup",
            "Bulk CSV Upload",
        ],
        default="Single Startup",
    )

    st.write("")

    if mode == "Single Startup":

        form_data = render_startup_form()

    else:

        upload_data = render_upload_zone()

    st.divider()

    render_section_header(

        title="Prediction Results",

        subtitle=(
            "Compare the proposed Spatial Feature Transformation "
            "Neural Network with the strongest baseline model."
        ),

    )

    results = get_prediction_result()

    left, right = st.columns(
        2,
        gap="large",
    )

    with left:

        render_model_result_card(
            results["sftnn"],
        )

    with right:

        render_model_result_card(
            results["baseline"],
        )

    st.write("")

    render_comparison_card(
        results["comparison"],
    )

    st.divider()

    render_section_header(

        title="Prediction Insights",

        subtitle=(
            "Powered by Explainable AI through Integrated Gradients "
            "and Spatial Feature Transformation."
        ),

    )

    explainability = get_explainability()

    left, right = st.columns(
        2,
        gap="large",
    )

    with left:

        render_regional_heatmap(
            explainability["regional"],
        )

    with right:

        render_integrated_gradients(
            explainability["integrated_gradients"],
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