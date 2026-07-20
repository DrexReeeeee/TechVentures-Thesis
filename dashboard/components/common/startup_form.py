import streamlit as st

from data.fake_data import get_dropdown_options


def render_startup_form() -> dict:

    options = get_dropdown_options()

    startup_col, funding_col = st.columns(
        2,
        gap="large",
    )

    # --------------------------------------------------
    # Startup Information
    # --------------------------------------------------

    with startup_col:

        with st.container(border=True):

            st.subheader("Startup Information")

            startup_name = st.text_input(
                "Startup Name (Optional)",
                placeholder="TechVenture",
            )

            country = st.selectbox(
                "Country",
                options["countries"],
            )

            region = st.selectbox(
                "Region",
                options["regions"],
            )

            industry = st.selectbox(
                "Industry",
                options["industries"],
            )

    # --------------------------------------------------
    # Funding Information
    # --------------------------------------------------

    with funding_col:

        with st.container(border=True):

            st.subheader("Funding Information")

            funding_stage = st.selectbox(
                "Funding Stage",
                options["funding_stages"],
            )

            last_round = st.selectbox(
                "Last Funding Round",
                options["funding_stages"],
            )

            total_funding = st.number_input(
                "Total Funding (USD)",
                min_value=0.0,
                step=100000.0,
                format="%.0f",
            )

            investors = st.number_input(
                "Number of Investors",
                min_value=0,
                step=1,
            )

    st.write("")

    predict = st.button(
        "Predict Startup Success",
        type="primary",
        use_container_width=True,
    )

    return {

        "predict": predict,

        "startup_name": startup_name,

        "country": country,

        "region": region,

        "industry": industry,

        "funding_stage": funding_stage,

        "last_funding_round": last_round,

        "total_funding": total_funding,

        "num_investors": investors,

    }