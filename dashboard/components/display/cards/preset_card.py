import streamlit as st


def render_preset_card(
    *,
    title: str,
    subtitle: str,
    icon: str,
    key: str,
) -> bool:
    """
    Render one preset model card.
    """

    with st.container(border=True):

        left, right = st.columns(
            [1, 5],
            vertical_alignment="top",
        )

        with left:

            st.button(
                "",
                icon=f":material/{icon}:",
                disabled=True,
                key=f"{key}_icon",
                use_container_width=True,
            )

        with right:

            st.markdown(f"#### {title}")

            st.caption(subtitle)

        st.write("")

        return st.button(
            "Load Preset",
            key=key,
            use_container_width=True,
        )