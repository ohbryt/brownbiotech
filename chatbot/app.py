"""
Brown Biotech Research Agent Chatbot
Streamlit Application
"""

import streamlit as st
import os
import sys
import time
import json
from datetime import datetime
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import Settings
from agents.router import RouterAgent
from agents.literature import LiteratureAgent
from agents.pipeline import PipelineAgent
from agents.market import MarketAgent
from agents.dataset import DatasetAgent
from agents.synthesizer import SynthesizerAgent
from utils.storage import ChatStorage
from utils.citations import CitationFormatter

# Page config
st.set_page_config(
    page_title="Brown Biotech Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    :root {
        --primary: #1E3A5F;
        --secondary: #2D5A87;
        --accent: #4ECDC4;
        --bg: #0D1B2A;
        --surface: #1B2838;
        --text: #E8F1F8;
        --warning: #F4A261;
        --success: #2ECC71;
    }
    
    .stApp {
        background-color: var(--bg);
    }
    
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--accent);
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1rem;
        color: #8899A6;
        margin-bottom: 1.5rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 0.75rem;
        border-left: 4px solid;
    }
    
    .user-message {
        background-color: var(--surface);
        border-color: var(--accent);
    }
    
    .agent-message {
        background-color: #15202B;
        border-color: var(--secondary);
    }
    
    .source-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        background-color: var(--primary);
        border-radius: 4px;
        font-size: 0.75rem;
        color: var(--text);
        margin-right: 0.25rem;
    }
    
    .metric-card {
        background-color: var(--surface);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--accent);
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #8899A6;
    }
    
    .tab-content {
        padding: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem;
        background-color: var(--surface);
        border-radius: 8px 8px 0 0;
    }
    
    .stTextInput > div > div > input {
        background-color: var(--surface);
        color: var(--text);
        border: 1px solid var(--secondary);
    }
    
    .stButton > button {
        background-color: var(--accent);
        color: var(--bg);
        font-weight: 600;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 8px;
    }
    
    .stButton > button:hover {
        background-color: #3DBDB5;
    }
    
    .sources-box {
        background-color: var(--surface);
        padding: 0.75rem;
        border-radius: 8px;
        font-size: 0.85rem;
        color: #8899A6;
        margin-bottom: 1rem;
    }
    
    .reference-item {
        font-size: 0.8rem;
        color: #8899A6;
        padding: 0.25rem 0;
        border-bottom: 1px solid #2D3E50;
    }
    
    .success-box {
        background-color: rgba(46, 204, 113, 0.1);
        border: 1px solid var(--success);
        padding: 1rem;
        border-radius: 8px;
        color: var(--success);
    }
    
    .error-box {
        background-color: rgba(244, 162, 97, 0.1);
        border: 1px solid var(--warning);
        padding: 1rem;
        border-radius: 8px;
        color: var(--warning);
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "settings" not in st.session_state:
        st.session_state.settings = Settings()
    if "storage" not in st.session_state:
        st.session_state.storage = ChatStorage()
    if "current_tab" not in st.session_state:
        st.session_state.current_tab = "💬 Chat"
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = {}


def get_router():
    """Get router agent instance."""
    return RouterAgent(st.session_state.settings)


def get_agent(agent_type: str):
    """Get specific agent instance."""
    agents = {
        "literature": LiteratureAgent,
        "pipeline": PipelineAgent,
        "market": MarketAgent,
        "dataset": DatasetAgent,
    }
    return agents.get(agent_type, LiteratureAgent)(st.session_state.settings)


def render_header():
    """Render application header."""
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown('<p class="main-header">🔬 Brown Biotech Research Agent</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">AI-powered pharmaceutical & biotech research analysis</p>', unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(st.session_state.messages) // 2}</div>
            <div class="metric-label">Queries Today</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.settings.model}</div>
            <div class="metric-label">Active Model</div>
        </div>
        """, unsafe_allow_html=True)
    st.divider()


def render_chat_tab():
    """Render chat interface tab."""
    # Display chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You:</strong><br>
                {msg["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            # Render agent response
            st.markdown(f"""
            <div class="chat-message agent-message">
                <strong>🔬 Brown Biotech Agent:</strong><br><br>
                {msg["content"]}
            </div>
            """, unsafe_allow_html=True)
            if msg.get("sources"):
                st.markdown(f'<div class="sources-box">📚 Sources: {msg["sources"]}</div>', unsafe_allow_html=True)
            if msg.get("references"):
                with st.expander("📖 References"):
                    for ref in msg["references"]:
                        st.markdown(f'<div class="reference-item">{ref}</div>', unsafe_allow_html=True)
            if msg.get("metadata"):
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"⏱️ {msg['metadata'].get('time', 'N/A')}")
                with col2:
                    st.caption(f"💰 Cost: {msg['metadata'].get('cost', 'N/A')}")

    # Query input
    st.divider()
    query = st.text_input(
        "Ask a research question...",
        placeholder="e.g., 'Analyze the GLP-1 agonist pipeline' or 'What are the latest findings on TGF-β in skin aging?'",
        key="query_input"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_button:
        st.session_state.messages = []
        st.rerun()
    
    if analyze_button and query:
        with st.spinner("🔬 Analyzing... This may take 10-30 seconds."):
            start_time = time.time()
            
            # Route the query
            router = get_router()
            intent = router.classify(query)
            
            # Get appropriate agent
            agent = get_agent(intent.agent_type)
            
            # Execute analysis
            result = agent.analyze(query, intent)
            
            # Synthesize response
            synthesizer = SynthesizerAgent(st.session_state.settings)
            response = synthesizer.synthesize(result, intent)
            
            elapsed = time.time() - start_time
            cost_estimate = result.get("cost", 0)
            
            # Add to messages
            st.session_state.messages.append({
                "role": "user",
                "content": query
            })
            st.session_state.messages.append({
                "role": "agent",
                "content": response.content,
                "sources": response.sources,
                "references": response.references,
                "metadata": {
                    "time": f"{elapsed:.1f}s",
                    "cost": f"${cost_estimate:.2f}",
                    "agent": intent.agent_type
                }
            })
            
            # Store for history
            st.session_state.storage.save_message(
                query=query,
                response=response.content,
                intent=intent.agent_type,
                sources=response.sources
            )
        
        st.rerun()


def render_analysis_tab():
    """Render structured analysis tab."""
    st.subheader("📊 Quick Analysis Templates")
    
    col1, col2, col3 = st.columns(3)
    
    templates = [
        ("💊 Drug Pipeline", "pipeline", "GLP-1, FXR, PPAR, GLP-1, SGLT2", "Analyzes drug pipeline by mechanism"),
        ("📚 Literature Review", "literature", "TGF-beta skin aging 2024", "Searches and summarizes recent papers"),
        ("📈 Market Analysis", "market", "Anti-aging cosmetics market size Korea", "Market research and competitive analysis"),
        ("🧬 Dataset Analysis", "dataset", "skin_aging_targets_2026", "Analyzes MERFISH or custom datasets"),
        ("🏢 Competitor Intel", "pipeline", "Novo Nordisk competitors", "Competitive landscape analysis"),
        ("🧪 Clinical Trials", "literature", "Phase 3 NASH clinical trials", "Clinical trial status tracking"),
    ]
    
    for i, (name, agent_type, example, desc) in enumerate(templates):
        with [col1, col2, col3][i % 3]:
            with st.container():
                st.markdown(f"**{name}**")
                st.caption(desc)
                if st.button(f"Try: {example[:30]}...", key=f"template_{i}", use_container_width=True):
                    # Auto-fill query
                    st.session_state.messages.append({
                        "role": "user",
                        "content": example
                    })
                    st.rerun()
    
    st.divider()
    st.subheader("📋 Recent Analyses")
    
    history = st.session_state.storage.get_history(limit=10)
    if history:
        for item in history:
            with st.expander(f"🔍 {item['query'][:50]}... | {item['timestamp']}"):
                st.markdown(f"**Query:** {item['query']}")
                st.markdown(f"**Type:** {item['intent']}")
                st.markdown(f"**Sources:** {item.get('sources', 'N/A')}")
    else:
        st.info("No analysis history yet. Start by asking a question in the Chat tab.")


def render_reports_tab():
    """Render generated reports tab."""
    st.subheader("📄 Generated Reports")
    
    # Check for saved reports
    reports_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    if os.path.exists(reports_dir):
        reports = [f for f in os.listdir(reports_dir) if f.endswith(('.md', '.pdf', '.docx'))]
        if reports:
            for report in reports:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"📄 **{report}**")
                with col2:
                    with open(os.path.join(reports_dir, report), 'r') as f:
                        content = f.read()
                    st.download_button(
                        "Download",
                        content,
                        file_name=report,
                        mime="text/markdown",
                        use_container_width=True
                    )
        else:
            st.info("No reports generated yet. Complete an analysis to create reports.")
    else:
        st.info("Reports directory not found.")
    
    st.divider()
    st.subheader("📊 Analysis Statistics")
    
    stats = st.session_state.storage.get_stats()
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Queries", stats.get("total", 0))
        with col2:
            st.metric("Today", stats.get("today", 0))
        with col3:
            st.metric("This Week", stats.get("week", 0))
        with col4:
            st.metric("Avg Cost", f"${stats.get('avg_cost', 0):.2f}")


def render_settings_tab():
    """Render settings tab."""
    st.subheader("⚙️ Configuration")
    
    # API Keys section
    with st.expander("🔑 API Keys", expanded=True):
        openrouter_key = st.text_input(
            "OpenRouter API Key",
            value=st.session_state.settings.OPENROUTER_API_KEY[:20] + "..." if st.session_state.settings.OPENROUTER_API_KEY else "",
            type="password",
            help="Get your key from openrouter.ai"
        )
        
        tinyfish_key = st.text_input(
            "TinyFish API Key",
            value=st.session_state.settings.TINYFISH_API_KEY[:20] + "..." if st.session_state.settings.TINYFISH_API_KEY else "",
            type="password",
            help="Get your key from tinyfish.ai"
        )
        
        google_key = st.text_input(
            "Google API Key (Optional)",
            value=st.session_state.settings.GOOGLE_API_KEY[:20] + "..." if st.session_state.settings.GOOGLE_API_KEY else "",
            type="password",
            help="For Gemini models"
        )
        
        if st.button("💾 Save Keys"):
            st.session_state.settings.OPENROUTER_API_KEY = openrouter_key
            st.session_state.settings.TINYFISH_API_KEY = tinyfish_key
            st.session_state.settings.GOOGLE_API_KEY = google_key
            st.session_state.settings.save()
            st.success("API keys saved successfully!")
    
    # Model selection
    with st.expander("🤖 Model Settings"):
        model_options = [
            "minimax/minimax-m2.7",
            "google/gemini-2.0-flash-exp",
            "anthropic/claude-3.5-haiku",
            "openai/gpt-4o-mini",
        ]
        selected_model = st.selectbox("Default Model", model_options, index=0)
        if st.button("💾 Save Model"):
            st.session_state.settings.model = selected_model
            st.session_state.settings.save()
            st.success("Model saved!")
    
    # Data sources
    with st.expander("📂 Data Sources"):
        st.text_input("MERFISH Dataset Path", value="/Users/ocm/.openclaw/workspace/skin_atlas_analysis/output/merfish.h5ad")
        st.text_input("ARP Pipeline Path", value="/Users/ocm/.openclaw/workspace/arp-v3/")
    
    # About
    with st.expander("ℹ️ About"):
        st.markdown("""
        **Brown Biotech Research Agent** v1.0
        
        Built with:
        - Streamlit (UI)
        - OpenRouter (AI Models)
        - TinyFish (Web Research)
        - ARP Pipeline (Analysis)
        
        **Contact:** Brown Biotech Co., Ltd.
        """)


def main():
    """Main application entry point."""
    init_session_state()
    render_header()
    
    # Tab navigation
    tabs = ["💬 Chat", "📊 Analysis", "📄 Reports", "⚙️ Settings"]
    tab = st.tabs(tabs)
    
    with tab[0]:
        render_chat_tab()
    with tab[1]:
        render_analysis_tab()
    with tab[2]:
        render_reports_tab()
    with tab[3]:
        render_settings_tab()


if __name__ == "__main__":
    main()
