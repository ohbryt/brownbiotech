# Brown Biotech Research Agent Chatbot

AI-powered research assistant for pharmaceutical and biotech analysis.

## Quick Start

### Option 1: Streamlit (Development)
```bash
cd chatbot
pip install -r requirements.txt
streamlit run app.py --server.port 8501
```

### Option 2: Static HTML (Production - Vercel Deploy)
The `static/index.html` is a self-contained HTML/JS chatbot that works without a backend.

**Deploy to Vercel:**
```bash
cd chatbot/static
vercel
```

## Features

- 💊 **Drug Pipeline Analysis** — GLP-1, FXR, PPAR, SGLT2, THR-β
- 📚 **Literature Search** — PubMed + web research
- 📈 **Market Analysis** — Anti-aging, cosmeceutical markets
- 🧬 **Dataset Analysis** — MERFISH skin atlas + custom datasets

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Web Interface                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │ Static HTML │  │  Streamlit  │  │ FastAPI +   │ │
│  │ (Vercel)   │  │   (Local)   │  │  React      │ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│                  API Backend                        │
│  Router Agent → Literature/Pipeline/Market/Dataset │
│  ↓                                                    │
│  Synthesizer Agent → Response + Citations           │
└─────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│              AI Models (OpenRouter)                 │
│  MiniMax | Gemini | Nemotron | Stepfun             │
└─────────────────────────────────────────────────────┘
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/query` | POST | Submit research query |
| `/examples` | GET | Get example queries |

## Example Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze the GLP-1 agonist pipeline"}'
```

## Files

```
chatbot/
├── agents/           # AI agent modules
│   ├── router.py     # Query intent classification
│   ├── literature.py  # PubMed search
│   ├── pipeline.py   # Drug pipeline
│   ├── market.py     # Market analysis
│   └── dataset.py    # Dataset analysis
├── static/           # Static web UI (Vercel deployable)
│   ├── index.html
│   └── style.css
├── api/              # FastAPI backend
│   └── main.py
├── app.py           # Streamlit app
└── config/          # Settings
```

## Configuration

Set API keys as environment variables:
- `OPENROUTER_API_KEY` — AI models (openrouter.ai)
- `TINYFISH_API_KEY` — Web research (optional)

## License

Proprietary — Brown Biotech Co., Ltd.
