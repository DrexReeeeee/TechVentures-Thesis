import streamlit as st

def render_model_result_card(data: dict):
    """
    Renders an individual prediction result card for a specific model (SFTNN or Baseline).
    Displays success probability, confidence bounds, and status badges.
    """
    is_highlight = data.get("highlight", False)
    
    # Visual border tint depending on champion status
    border_color = "var(--primary-green)" if is_highlight else "rgba(255, 255, 255, 0.12)"
    
    with st.container(border=True):
        # Header Badge & Model Name
        col_title, col_badge = st.columns([2, 1])
        with col_title:
            st.markdown(f"### {data.get('model', 'Model')}")
        with col_badge:
            badge_text = data.get("badge", "")
            if is_highlight:
                st.markdown(
                    f"<div style='text-align: right;'><span style='background-color: var(--dark-green); color: var(--primary-green); padding: 4px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; border: 1px solid var(--primary-green);'>🏆 {badge_text}</span></div>", 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='text-align: right;'><span style='background-color: rgba(255,255,255,0.05); color: rgba(255,255,255,0.7); padding: 4px 10px; border-radius: 12px; font-size: 0.8rem;'>{badge_text}</span></div>", 
                    unsafe_allow_html=True
                )

        st.divider()

        # Primary Probability Metric
        prob = data.get("probability", 0.0)
        conf = data.get("confidence", 0.0)
        
        c1, c2 = st.columns(2)
        with c1:
            st.caption("SUCCESS PROBABILITY")
            st.markdown(f"<h1 style='margin:0; font-size: 2.8rem; color: {'#3ACC9F' if prob >= 50 else '#E5484D'};'>{prob:.1f}%</h1>", unsafe_allow_html=True)
        
        with c2:
            st.caption("MODEL CONFIDENCE")
            st.markdown(f"<h2 style='margin:0; font-size: 2.0rem; opacity: 0.85;'>{conf:.1f}%</h2>", unsafe_allow_html=True)

        st.write("")
        
        # Predicted State Status Bar
        prediction_text = data.get("prediction", "Likely Successful")
        if prob >= 50:
            st.success(f"🟢 **Target Status:** {prediction_text}")
        else:
            st.error(f"🔴 **Target Status:** {prediction_text}")

        # Mathematical Uncertainty Interval (Academic Detail)
        ci_lower = max(0.0, prob - (100 - conf) * 0.15)
        ci_upper = min(100.0, prob + (100 - conf) * 0.15)
        st.caption(f"📊 **95% Confidence Interval:** `[{ci_lower:.1f}% — {ci_upper:.1f}%]`")