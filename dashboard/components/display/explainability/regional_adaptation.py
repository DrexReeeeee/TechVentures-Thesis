import streamlit as st
import plotly.graph_objects as go

def render_regional_heatmap(data: dict):
    """
    Renders learned spatial affine transformation parameters (gamma scaling and beta shift)
    alongside a comparative cross-regional modulation chart.
    """
    with st.container(border=True):
        st.subheader("Regional Feature Transformation")
        st.caption("Active affine scaling (γ) and shift (β) conditioning parameters extracted from SFTNN.")
        st.divider()

        selected_region = data.get("selected_region", "Unknown")
        gamma = data.get("gamma", 1.0)
        beta = data.get("beta", 0.0)

        st.markdown(f"**Target Ecosystem:** `{selected_region}`")
        st.write("")

        # Display Metrics with human-readable interpretation
        col1, col2 = st.columns(2)

        with col1:
            gamma_pct = (gamma - 1.0) * 100
            gamma_delta = f"{gamma_pct:+.1f}% Velocity"
            st.metric(
                label="Regional Scale (γ)",
                value=f"{gamma:.2f}",
                delta=gamma_delta,
                help="Scaling factor applied to startup feature maps. Values > 1.0 amplify feature responsiveness in this region."
            )

        with col2:
            beta_pct = beta * 100
            beta_delta = f"{beta_pct:+.1f}% Base Shift"
            st.metric(
                label="Regional Shift (β)",
                value=f"{beta:.2f}",
                delta=beta_delta,
                delta_color="normal" if beta >= 0 else "inverse",
                help="Additive probability offset capturing local ecosystem baseline advantages or headwinds."
            )

        st.divider()
        st.caption("🌐 **Cross-Ecosystem Modulation Comparison**")

        heatmap_data = data.get("heatmap", [])
        if heatmap_data:
            regions = [item[0] for item in heatmap_data]
            scores = [item[1] for item in heatmap_data]

            # Highlight the currently selected region with Brand Green (#3ACC9F)
            colors = ["#3ACC9F" if reg == selected_region else "rgba(255,255,255,0.25)" for reg in regions]

            fig = go.Figure(go.Bar(
                x=scores,
                y=regions,
                orientation='h',
                marker=dict(color=colors),
                text=[f"{val:.2f}" for val in scores],
                textposition='outside',
                textfont=dict(color='white')
            ))

            fig.update_layout(
                margin=dict(l=10, r=20, t=10, b=10),
                height=200,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)',
                    range=[0, max(scores) * 1.25] if scores else [0, 2.0]
                ),
                yaxis=dict(
                    autorange="reversed",
                    tickfont=dict(color='white', size=12)
                )
            )

            st.plotly_chart(fig, use_container_width=True)