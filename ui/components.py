import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def metric_card(title, value, delta=None, delta_color="normal", type="default"):
    """Displays a styled metric card."""
    card_class = "metric-card-positive" if type == "positive" else \
                 "metric-card-negative" if type == "negative" else \
                 "metric-card-warning" if type == "warning" else ""
    
    st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
    st.metric(label=title, value=value, delta=delta, delta_color=delta_color)
    st.markdown('</div>', unsafe_allow_html=True)

def agent_status_card(name, role, status, last_action):
    """Displays an agent status card."""
    status_class = "agent-status-active" if status.lower() == "active" else "agent-status-idle"
    st.markdown(f"""
    <div class="agent-card">
        <div class="agent-title">🤖 {name}</div>
        <div><strong>Role:</strong> {role}</div>
        <div><strong>Status:</strong> <span class="{status_class}">{status}</span></div>
        <div style="margin-top:10px; font-size:0.9em; color:#b0bec5;">
            <em>Last Action:</em> {last_action}
        </div>
    </div>
    """, unsafe_allow_html=True)

def risk_gauge(score, title="Risk Score"):
    """Displays a gauge chart for risk score."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        title = {'text': title},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "rgba(0,0,0,0)"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score}
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
    return fig

def confusion_matrix_plot(cm_data, labels):
    """Displays a confusion matrix heatmap."""
    fig = px.imshow(cm_data,
                    labels=dict(x="Predicted Label", y="True Label", color="Count"),
                    x=labels,
                    y=labels,
                    text_auto=True,
                    color_continuous_scale="Blues")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
    return fig

def roc_curve_plot(fpr, tpr, auc_score):
    """Displays an ROC curve."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {auc_score:.2f})', line=dict(color='#ff9800', width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), showlegend=False))
    fig.update_layout(title="Receiver Operating Characteristic (ROC)", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
    return fig

def quantization_comparison_chart(df):
    """Displays a radar chart comparing quantization metrics."""
    categories = ['Model Size (MB)', 'Inference Latency (ms)', 'Memory Usage (MB)', 'Accuracy (%)']
    
    fig = go.Figure()
    for index, row in df.iterrows():
        # Normalize data for radar chart visibility
        size_norm = (row['Model Size'] / df['Model Size'].max()) * 100
        latency_norm = (row['Latency'] / df['Latency'].max()) * 100
        memory_norm = (row['Memory'] / df['Memory'].max()) * 100
        acc_norm = row['Accuracy']
        
        fig.add_trace(go.Scatterpolar(
            r=[size_norm, latency_norm, memory_norm, acc_norm],
            theta=categories,
            fill='toself',
            name=row['Format']
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "white"}
    )
    return fig

def audit_timeline(entries):
    """Displays a timeline of audit entries."""
    for entry in entries:
        st.markdown(f"""
        <div style="border-left: 3px solid #1e88e5; padding-left: 15px; margin-bottom: 15px; background: rgba(30,30,30,0.5); padding: 10px; border-radius: 0 5px 5px 0;">
            <div style="color: #b0bec5; font-size: 0.85em;">{entry['timestamp']} | {entry['agent']}</div>
            <div style="font-weight: bold; margin: 5px 0;">{entry['action']}</div>
            <div style="font-size: 0.9em;">{entry['details']}</div>
        </div>
        """, unsafe_allow_html=True)
