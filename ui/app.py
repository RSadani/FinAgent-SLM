import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import json
import random

# Try importing local modules, but fallback to demo mode
try:
    from ui.styles import CUSTOM_CSS
    from ui.components import (
        metric_card, agent_status_card, risk_gauge, 
        confusion_matrix_plot, roc_curve_plot, 
        quantization_comparison_chart, audit_timeline
    )
except ImportError:
    st.error("Failed to import UI components. Make sure you are running from the project root.")
    st.stop()

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="FinAgent-SLM Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
def init_session_state():
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Dashboard Overview"
    if 'demo_data_generated' not in st.session_state:
        generate_demo_data()
        st.session_state.demo_data_generated = True
    if 'is_analyzing' not in st.session_state:
        st.session_state.is_analyzing = False

def generate_demo_data():
    """Generate realistic demo data for the dashboard."""
    # 1. Transaction Data
    np.random.seed(42)
    n_samples = 1000
    dates = [datetime.now() - timedelta(days=x, hours=random.randint(0,23)) for x in range(30)]
    
    tx_types = ['Transfer', 'Withdrawal', 'Deposit', 'Payment', 'Crypto']
    amounts = np.random.exponential(scale=5000, size=n_samples)
    
    # Introduce some "fraudulent" patterns
    is_fraud = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    risk_scores = np.where(is_fraud == 1, np.random.uniform(70, 100, n_samples), np.random.uniform(0, 40, n_samples))
    
    st.session_state.tx_data = pd.DataFrame({
        'Transaction_ID': [f"TXN-{random.randint(10000,99999)}" for _ in range(n_samples)],
        'Date': np.random.choice(dates, n_samples),
        'Type': np.random.choice(tx_types, n_samples, p=[0.3, 0.2, 0.2, 0.25, 0.05]),
        'Amount': amounts,
        'Risk_Score': risk_scores,
        'Is_Fraud_Pred': is_fraud,
        'Status': np.where(is_fraud == 1, 'Flagged', 'Cleared')
    }).sort_values('Date', ascending=False)
    
    # 2. Agent Logs
    st.session_state.agent_logs = [
        {"timestamp": (datetime.now() - timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S"), "agent": "Fraud Detector", "action": "Analyzed TXN-58291", "details": "Detected unusual cross-border transfer. Risk Score: 85."},
        {"timestamp": (datetime.now() - timedelta(minutes=8)).strftime("%Y-%m-%d %H:%M:%S"), "agent": "Tax Analyzer", "action": "Reviewing TXN-58291", "details": "Checking historical tax records for entity. Found mismatch in reported income."},
        {"timestamp": (datetime.now() - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"), "agent": "Audit Trail", "action": "Generated SAR", "details": "Suspicious Activity Report generated and filed for TXN-58291."},
        {"timestamp": (datetime.now() - timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M:%S"), "agent": "Fraud Detector", "action": "Monitoring Stream", "details": "Processing batch of 50 new transactions."}
    ]
    
    # 3. Model Metrics
    st.session_state.model_metrics = {
        'epochs': list(range(1, 21)),
        'loss': [max(0.1, 0.8 * (0.85 ** x) + random.uniform(-0.02, 0.02)) for x in range(20)],
        'accuracy': [min(0.98, 0.6 + 0.35 * (1 - np.exp(-x/4)) + random.uniform(-0.01, 0.01)) for x in range(20)],
        'cm': np.array([[850, 15], [8, 127]]),
        'fpr': [0, 0.05, 0.1, 0.2, 0.5, 1],
        'tpr': [0, 0.8, 0.9, 0.95, 0.98, 1],
        'auc': 0.94
    }
    
    # 4. Quantization Data
    st.session_state.quant_data = pd.DataFrame({
        'Format': ['FP32 (Base)', 'FP16', 'INT8', 'INT4'],
        'Model Size': [420.5, 210.2, 105.1, 52.6],
        'Latency': [45.2, 28.5, 15.3, 8.7],
        'Memory': [1200, 650, 340, 180],
        'Accuracy': [96.5, 96.3, 95.8, 93.2]
    })

init_session_state()

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/bank-building.png", width=60)
    st.markdown("## FinAgent-SLM")
    st.markdown("Enterprise Fraud Detection")
    st.divider()
    
    pages = [
        "Dashboard Overview",
        "SLM Generative Copilot",
        "Multi-Agent Monitor",
        "Model Performance",
        "Quantization Matrix",
        "Audit Trail"
    ]
    
    for page in pages:
        if st.button(page, use_container_width=True, type="primary" if st.session_state.current_page == page else "secondary"):
            st.session_state.current_page = page
            st.rerun()
            
    st.divider()
    st.markdown("### System Status")
    st.success("🟢 API Connected")
    st.success("🟢 Agents Active")
    st.success("🟢 Model Loaded (INT8)")

# ==========================================
# PAGE VIEWS
# ==========================================

def render_dashboard():
    st.title("Dashboard Overview")
    st.markdown("Real-time monitoring of tax fraud detection systems.")
    
    df = st.session_state.tx_data
    total_tx = len(df)
    total_fraud = len(df[df['Is_Fraud_Pred'] == 1])
    det_rate = (total_fraud / total_tx) * 100
    avg_risk = df['Risk_Score'].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Analyzed Transactions", f"{total_tx:,}", "+124 this hour", "normal", "default")
    with col2:
        metric_card("Fraud Detected", f"{total_fraud}", "+3 today", "inverse", "warning")
    with col3:
        metric_card("Detection Rate", f"{det_rate:.1f}%", "+0.2%", "normal", "positive")
    with col4:
        metric_card("Average Risk Score", f"{avg_risk:.1f}", "-2.1", "inverse", "default")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_l, col_r = st.columns([2, 1])
    
    with col_l:
        st.subheader("Real-Time Fraud Detection Volume")
        daily_vol = df.groupby(df['Date'].dt.date).size().reset_index(name='Count')
        daily_fraud = df[df['Is_Fraud_Pred'] == 1].groupby(df['Date'].dt.date).size().reset_index(name='Fraud_Count')
        merged = pd.merge(daily_vol, daily_fraud, on='Date', how='left').fillna(0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=merged['Date'], y=merged['Count'], mode='lines', name='Total Transactions', line=dict(color='#1e88e5')))
        fig.add_trace(go.Scatter(x=merged['Date'], y=merged['Fraud_Count'], mode='lines', name='Fraud Flags', line=dict(color='#f44336')))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
    with col_r:
        st.subheader("Transaction Types")
        type_dist = df['Type'].value_counts()
        fig = px.pie(values=type_dist.values, names=type_dist.index, hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
    st.subheader("Recent High-Risk Alerts")
    high_risk = df[df['Risk_Score'] > 75].head(5)
    
    # Styled dataframe
    st.dataframe(
        high_risk[['Transaction_ID', 'Date', 'Type', 'Amount', 'Risk_Score', 'Status']].style.applymap(
            lambda x: 'color: #f44336; font-weight: bold;' if isinstance(x, (int, float)) and x > 80 else '',
            subset=['Risk_Score']
        ).format({'Amount': '${:,.2f}', 'Risk_Score': '{:.1f}'}),
        use_container_width=True,
        hide_index=True
    )

def render_slm_copilot():
    st.title("🤖 SLM Generative Copilot")
    st.markdown("Interact directly with the FinAgent Small Language Model to query the database, generate natural language explanations for fraud, and investigate specific entities.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Automated SAR Generation")
        st.write("Select a flagged transaction to see the SLM generate a natural language reasoning trace.")
        
        flagged_tx = st.session_state.tx_data[st.session_state.tx_data['Status'] == 'Flagged']
        if not flagged_tx.empty:
            selected_tx = st.selectbox("Select Suspicious Transaction", flagged_tx['Transaction_ID'])
            tx_details = flagged_tx[flagged_tx['Transaction_ID'] == selected_tx].iloc[0]
            
            st.markdown(f"**Data:** `Amount: ${tx_details['Amount']:,.2f} | Type: {tx_details['Type']} | Risk: {tx_details['Risk_Score']:.1f}`")
            
            if st.button("🧠 Generate SLM Explanation", type="primary"):
                st.markdown("#### SLM Reasoning Stream:")
                
                def generate_slm_text():
                    explanation = f"Based on my analysis of the transaction graph and historical baselines, transaction {selected_tx} exhibits severe anomalies typical of layering in money laundering. \n\n1. **Velocity Anomaly**: The {tx_details['Type']} of ${tx_details['Amount']:,.2f} represents a 400% deviation from this entity's standard 30-day moving average.\n2. **Balance Exhaustion**: The transaction perfectly drains the origin account down to a zero balance immediately after a large cash influx, which strongly correlates with synthetic identity drain-out patterns.\n3. **Tax Discrepancy**: Cross-referencing the entity's tax profile, this single transfer exceeds their total reported annual income by a significant margin.\n\n**SLM Conclusion**: I recommend an immediate freeze on the destination account and the automated generation of a Suspicious Activity Report (SAR) for the Ministry of Finance."
                    
                    for word in explanation.split(" "):
                        yield word + " "
                        time.sleep(0.04)
                
                with st.chat_message("assistant", avatar="🤖"):
                    st.write_stream(generate_slm_text)
                    st.caption("Generated by FinAgent-SLM (4.8M parameters) • 52 tokens/sec")
        else:
            st.success("No flagged transactions currently.")
            
    with col2:
        st.subheader("Chat with FinAgent-SLM")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I am the FinAgent Small Language Model. I have currently analyzed 100,000 transactions. How can I assist your investigation today?"}
            ]

        # Display chat messages from history on app rerun
        chat_container = st.container(height=400)
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("Ask the SLM (e.g., 'Summarize fraud patterns')"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant", avatar="🤖"):
                    message_placeholder = st.empty()
                    
                    if "summarize" in prompt.lower() or "pattern" in prompt.lower():
                        slm_reply = "I have detected three primary fraud patterns in today's batch: \n\n1. **Ghost Refunds**: 14 instances of tax returns claiming standard deductions but routing refunds to newly created offshore accounts.\n2. **Rapid Drain**: 42 transfers matching the 'rapid successive cashout' signature.\n3. **Income Underreporting**: 8 entities show transaction velocities indicating cash flows >300% higher than their reported tax income.\n\nWould you like me to draft SARs for these entities?"
                    else:
                        slm_reply = f"I am analyzing the context of your request: '{prompt}'. Based on my embedded knowledge of Ministry of Finance compliance rules and current graph states, I can confirm the flagged accounts have been isolated. I am ready to process further natural language queries."
                    
                    full_response = ""
                    for chunk in slm_reply.split(" "):
                        full_response += chunk + " "
                        time.sleep(0.04)
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    st.caption("Generation speed: 64 tokens/sec | Model: FinAgent-SLM-Instruct")
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})

def render_multi_agent():
    st.title("Multi-Agent Monitor")
    st.markdown("Monitor the ReAct (Reasoning and Acting) loop of the multi-agent system.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        agent_status_card("Fraud Detector", "Primary Anomaly Detection", "Active", "Processing batch of 50 new transactions.")
    with col2:
        agent_status_card("Tax Analyzer", "Regulatory Compliance Check", "Active", "Checking historical tax records for entity.")
    with col3:
        agent_status_card("Audit Trail", "Documentation & SAR Gen", "Idle", "Waiting for flagged events.")
        
    st.divider()
    
    st.subheader("Live Agent Communication Log")
    
    log_container = st.container(height=400)
    with log_container:
        for log in st.session_state.agent_logs:
            st.markdown(f"""
            <div style="background-color: #1e1e1e; padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 3px solid #ff9800;">
                <span style="color: #b0bec5; font-size: 0.8em;">{log['timestamp']}</span> | 
                <strong style="color: #1e88e5;">{log['agent']}</strong><br>
                <strong>Action:</strong> {log['action']}<br>
                <em>{log['details']}</em>
            </div>
            """, unsafe_allow_html=True)
            
    if st.button("Simulate New Agent Activity"):
        new_log = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "agent": random.choice(["Fraud Detector", "Tax Analyzer", "Audit Trail"]),
            "action": f"Action_{random.randint(100,999)}",
            "details": "Simulated log entry representing internal reasoning."
        }
        st.session_state.agent_logs.insert(0, new_log)
        st.rerun()

def render_model_performance():
    st.title("Model Performance")
    st.markdown("Evaluation metrics for the core SLM (Small Language Model).")
    
    metrics = st.session_state.model_metrics
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Training History")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=metrics['epochs'], y=metrics['loss'], name='Loss', line=dict(color='#f44336')))
        fig.add_trace(go.Scatter(x=metrics['epochs'], y=metrics['accuracy'], name='Accuracy', yaxis='y2', line=dict(color='#4caf50')))
        fig.update_layout(
            yaxis2=dict(title='Accuracy', overlaying='y', side='right', range=[0, 1]),
            yaxis=dict(title='Loss', range=[0, 1]),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"},
            height=300, margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("ROC Curve")
        st.plotly_chart(roc_curve_plot(metrics['fpr'], metrics['tpr'], metrics['auc']), use_container_width=True)
        
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Confusion Matrix")
        st.plotly_chart(confusion_matrix_plot(metrics['cm'], ["Legitimate", "Fraud"]), use_container_width=True)
        
    with col4:
        st.subheader("Classification Report")
        report_data = {
            'Class': ['Legitimate', 'Fraud', 'Accuracy', 'Macro Avg'],
            'Precision': ['0.99', '0.89', '', '0.94'],
            'Recall': ['0.98', '0.94', '', '0.96'],
            'F1-Score': ['0.99', '0.92', '0.98', '0.95'],
            'Support': ['865', '135', '1000', '1000']
        }
        st.table(pd.DataFrame(report_data).set_index('Class'))

def render_quantization():
    st.title("Quantization Matrix")
    st.markdown("Comparison of model optimization techniques for edge deployment.")
    
    df = st.session_state.quant_data
    
    st.dataframe(
        df.style.highlight_min(subset=['Model Size', 'Latency', 'Memory'], color='rgba(76, 175, 80, 0.3)')
                .highlight_max(subset=['Accuracy'], color='rgba(76, 175, 80, 0.3)')
                .format({'Model Size': '{:.1f} MB', 'Latency': '{:.1f} ms', 'Memory': '{:.0f} MB', 'Accuracy': '{:.1f}%'}),
        use_container_width=True, hide_index=True
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Latency vs Accuracy Trade-off")
        fig = px.scatter(df, x='Latency', y='Accuracy', text='Format', size='Model Size', color='Format',
                         title="Bubble size represents Model Size")
        fig.update_traces(textposition='top center')
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Multi-metric Comparison")
        st.plotly_chart(quantization_comparison_chart(df), use_container_width=True)
        
    st.info("💡 **Recommendation:** The INT8 quantized model offers the best balance for current hardware, reducing latency by 66% while maintaining 99.2% of the original FP32 accuracy.")

def render_audit_trail():
    st.title("Audit Trail & Compliance")
    st.markdown("Immutable record of system actions and generated reports.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("System Timeline")
        audit_timeline(st.session_state.agent_logs)
        
    with col2:
        st.subheader("Export & Reports")
        st.markdown("Generate compliant documentation for regulators.")
        
        if st.button("📄 Generate SAR (Suspicious Activity Report)", use_container_width=True):
            with st.spinner("Compiling multi-agent reasoning into SAR format..."):
                time.sleep(1.5)
                st.success("SAR generated successfully!")
                
        if st.button("💾 Export Audit Log (JSON)", use_container_width=True):
            log_json = json.dumps(st.session_state.agent_logs, indent=2)
            st.download_button(
                label="Click to Download",
                data=log_json,
                file_name="audit_log.json",
                mime="application/json",
                use_container_width=True
            )
            
        st.divider()
        st.markdown("### Compliance Checklist")
        st.checkbox("Data Anonymization Applied", value=True, disabled=True)
        st.checkbox("Reasoning Trace Recorded", value=True, disabled=True)
        st.checkbox("Human-in-the-loop Notification", value=True, disabled=True)
        st.checkbox("Regulatory Format (FINCEN) Validated", value=True, disabled=True)

# ==========================================
# ROUTING
# ==========================================
if st.session_state.current_page == "Dashboard Overview":
    render_dashboard()
elif st.session_state.current_page == "SLM Generative Copilot":
    render_slm_copilot()
elif st.session_state.current_page == "Multi-Agent Monitor":
    render_multi_agent()
elif st.session_state.current_page == "Model Performance":
    render_model_performance()
elif st.session_state.current_page == "Quantization Matrix":
    render_quantization()
elif st.session_state.current_page == "Audit Trail":
    render_audit_trail()

