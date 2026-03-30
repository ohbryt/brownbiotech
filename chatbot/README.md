# Brown Biotech Research Agent Chatbot 🔬

AI-powered pharmaceutical & biotech research analysis interface for Brown Biotech Co., Ltd.

## Quick Start

```bash
cd /Users/ocm/.openclaw/workspace/brown-biotech-chatbot

# Set API keys
export OPENROUTER_API_KEY="your-key-here"

# Run the app
streamlit run app.py --server.port 8501
```

## Features

- **💬 Chat Interface** — Natural language queries for research
- **📊 Drug Pipeline Analysis** — GLP-1, FXR, PPAR, SGLT2, THR-β
- **📚 Literature Search** — PubMed + web research
- **📈 Market Analysis** — Anti-aging, cosmeceutical, biotech markets
- **🧬 Dataset Analysis** — MERFISH skin atlas + custom datasets

## Architecture

```
User Query → Router Agent → Intent Classification
                              ↓
              ┌──────────────┼──────────────┐
              ↓              ↓              ↓
         Literature      Pipeline        Market
              └──────────────┼──────────────┘
                              ↓
                    Synthesizer Agent
                              ↓
                         Response
```

## Project Structure

```
brown-biotech-chatbot/
├── app.py              # Streamlit main app
├── agents/
│   ├── router.py       # Intent classification
│   ├── literature.py   # Paper search
│   ├── pipeline.py     # Drug pipeline
│   ├── market.py       # Market analysis
│   ├── dataset.py      # Dataset analysis
│   └── synthesizer.py  # Response compilation
├── utils/
│   ├── storage.py      # SQLite chat history
│   └── citations.py    # Citation formatting
├── config/
│   └── settings.py     # API keys & config
└── requirements.txt
```

## API Keys Required

| Service | Key | Get from |
|---------|-----|----------|
| OpenRouter | `OPENROUTER_API_KEY` | openrouter.ai |
| TinyFish | `TINYFISH_API_KEY` | tinyfish.ai |

## Tab Overview

### 💬 Chat
Main conversational interface. Ask research questions in natural language.

### 📊 Analysis
Quick templates for common queries:
- Drug pipeline templates
- Literature review templates
- Market analysis templates
- Competitor intelligence

### 📄 Reports
View and download generated reports. See usage statistics.

### ⚙️ Settings
Configure API keys, select default AI model, set data paths.

## Example Queries

```
"Analyze the GLP-1 agonist pipeline"
"What are the latest findings on TGF-β in skin aging?"
"Market size of anti-aging cosmetics in Korea"
"Compare Novo Nordisk vs Eli Lilly obesity drugs"
"Analyze skin aging genes from MERFISH data"
```

## License

Proprietary — Brown Biotech Co., Ltd.
