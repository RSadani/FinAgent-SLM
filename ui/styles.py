CUSTOM_CSS = """
<style>
    /* Dark enterprise theme with blue/orange accents */
    :root {
        --primary-blue: #1e88e5;
        --secondary-blue: #1565c0;
        --accent-orange: #ff9800;
        --dark-bg: #121212;
        --card-bg: #1e1e1e;
        --text-main: #ffffff;
        --text-muted: #b0bec5;
        --success: #4caf50;
        --warning: #ffb300;
        --danger: #f44336;
    }

    .stApp {
        background-color: var(--dark-bg);
        color: var(--text-main);
    }
    
    .stMetric {
        background-color: var(--card-bg);
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border-left: 4px solid var(--primary-blue);
    }

    .metric-card-positive {
        border-left: 4px solid var(--success);
    }
    
    .metric-card-negative {
        border-left: 4px solid var(--danger);
    }

    .metric-card-warning {
        border-left: 4px solid var(--warning);
    }

    h1, h2, h3 {
        color: var(--text-main) !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    .sidebar .sidebar-content {
        background-color: #1a1a24;
    }

    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    .agent-card {
        background-color: var(--card-bg);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
        border-top: 3px solid var(--accent-orange);
    }

    .agent-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: var(--primary-blue);
        margin-bottom: 10px;
    }

    .agent-status-active {
        color: var(--success);
        font-weight: bold;
    }

    .agent-status-idle {
        color: var(--text-muted);
        font-weight: bold;
    }
    
    .stButton>button {
        background-color: var(--primary-blue);
        color: white;
        border-radius: 5px;
        border: none;
        transition: all 0.3s;
    }

    .stButton>button:hover {
        background-color: var(--secondary-blue);
        box-shadow: 0 4px 8px rgba(30,136,229,0.4);
    }
</style>
"""
