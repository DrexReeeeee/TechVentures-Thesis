import streamlit as st


def render_upload_zone():

    with st.container(border=True):

        st.subheader("Bulk CSV Upload")

        st.caption(
            "Upload a CSV file containing multiple startups for prediction."
        )

        uploaded_file = st.file_uploader(
            "CSV File",
            type=["csv"],
        )

        left, right = st.columns(
            2,
            gap="medium",
        )

        with left:

            st.button(
                "Download Template",
                use_container_width=True,
            )

        with right:

            predict = st.button(
                "Predict Dataset",
                type="primary",
                use_container_width=True,
            )

    return {

        "uploaded_file": uploaded_file,

        "predict": predict,

    }