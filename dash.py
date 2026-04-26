import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
st.set_page_config(page_title="Wavetable CVAE Explorer", layout="wide")

@st.cache_data
def load_data():
    try:
        df = pd.read_pickle('dataset_cvae.pkl')
        return df
    except FileNotFoundError:
        st.error("Dataset 'dataset_cvae.pkl' not found. Please export it from your Jupyter Notebook.")
        st.stop()

df = load_data()
label_cols = ['Brightness', 'Richness', 'Fullness', 'Symmetry', 'Undulation']

if 'selected_idx' not in st.session_state:
    st.session_state.selected_idx = 0

# ==========================================
# 2. SIDEBAR (FILTERS & AXES)
# ==========================================
st.sidebar.header("Space Configuration")
axis_x = st.sidebar.selectbox("X Axis (Model Input)", label_cols, index=0)
axis_y = st.sidebar.selectbox("Y Axis (Model Input)", label_cols, index=1)

st.sidebar.markdown("---")
st.sidebar.header("Data Filtering")
min_x, max_x = st.sidebar.slider(f"Range for {axis_x}", -4.0, 4.0, (-2.5, 2.5))
min_y, max_y = st.sidebar.slider(f"Range for {axis_y}", -4.0, 4.0, (-2.5, 2.5))

filtered_df = df[
    (df[axis_x] >= min_x) & (df[axis_x] <= max_x) &
    (df[axis_y] >= min_y) & (df[axis_y] <= max_y)
].reset_index(drop=True)

# ==========================================
# 3. MAIN LAYOUT
# ==========================================
st.title("Acoustic Feature Analysis")
st.write(f"Visualizing the relationship between sound features for CVAE training.")

if filtered_df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

col1, col2 = st.columns([2, 1])


if st.session_state.selected_idx >= len(filtered_df):
    st.session_state.selected_idx = 0
current_sample = filtered_df.iloc[st.session_state.selected_idx]


with col2:
    st.subheader("Sample Detail")
    if st.button("Shuffle Selection", use_container_width=True):
        st.session_state.selected_idx = np.random.randint(0, len(filtered_df))
        st.rerun()
        
    fig_wave, ax_wave = plt.subplots(figsize=(5, 3))
    ax_wave.plot(current_sample['waveform'], color='#FF4B4B', linewidth=2)
    ax_wave.set_axis_off()
    fig_wave.patch.set_facecolor('#0E1117')
    st.pyplot(fig_wave)
    
    st.write("**Sample Parameters:**")
    m_cols = st.columns(2)
    for i, col in enumerate(label_cols):
        m_cols[i % 2].metric(label=col, value=f"{current_sample[col]:.2f}")


with col1:
    st.subheader("Latent Space")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered_df[axis_x], y=filtered_df[axis_y], mode='markers',
        marker=dict(color='#4A90E2', opacity=0.5, size=8),
        name='All Samples'
    ))
    fig.add_trace(go.Scatter(
        x=[current_sample[axis_x]], y=[current_sample[axis_y]], mode='markers',
        marker=dict(color='#FF4B4B', size=16, line=dict(color='white', width=2)),
        name='Current Selection'
    ))
    fig.update_layout(
        plot_bgcolor='#0E1117', paper_bgcolor='#0E1117', font=dict(color='white'),
        xaxis=dict(title=axis_x, zerolinecolor='#666', gridcolor='#333'),
        yaxis=dict(title=axis_y, zerolinecolor='#666', gridcolor='#333'),
        height=500, margin=dict(l=0, r=0, t=0, b=0), showlegend=False
    )
    
    event = st.plotly_chart(fig, on_select="rerun", selection_mode="points")
    if event and len(event.selection.points) > 0:
        clicked_idx = event.selection.points[0].get('point_index')
        if clicked_idx is not None and clicked_idx != st.session_state.selected_idx:
            st.session_state.selected_idx = clicked_idx
            st.rerun()

# ==========================================
# 4. CORRELATION & RECOMMENDATION
# ==========================================
st.markdown("---")
st.subheader("Statistical Analysis & Recommendations")

corr_val = filtered_df[axis_x].corr(filtered_df[axis_y])

c_col1, c_col2 = st.columns([1, 3])

with c_col1:
    st.metric("Pearson Coefficient", f"{corr_val:.3f}")

with c_col2:
    if abs(corr_val) > 0.85:
        st.error("**Critical Correlation!**")
        st.write(f"The features **{axis_x}** and **{axis_y}** are nearly identical. If you use both as conditions for the CVAE, the model will struggle to disentangle their influence. It is recommended to use only one or combine them.")
    elif abs(corr_val) > 0.5:
        st.warning("**Strong Relationship**")
        st.write(f"These parameters share a strong trend. During sound generation, expect changes in brightness to significantly affect harmonic richness as well. This is realistic but provides less flexibility for independent control.")
    elif abs(corr_val) < 0.2:
        st.success("**Ideal Independence**")
        st.write(f"Excellent! The parameters **{axis_x}** and **{axis_y}** are independent. For a CVAE, these are perfect 'knobs' that allow you to control two distinct aspects of the sound without interference.")
    else:
        st.info("ℹ**Moderate Correlation**")
        st.write("The relationship between these parameters is in a safe zone. They offer a good compromise between realistic acoustic behavior and controllable independence.")
