# Brown Biotech Research Agent Chatbot — SPEC

## 1. Concept & Vision

**브라운바이오텍 연구 챗봇** — 제약/바이오텍 산업 전문가를 위한 AI 기반 연구 분석 인터페이스. 자연어로 질문하면 ARP 파이프라인이 文献검색 + 데이터 분석 + 비즈니스 인사이트를 제공. Dr. OCM (KAIST, 3,818 인용)의 학술적 권위를 기반으로 한 과학적 근거 제공.

**Vibe:** Bloomberg Terminal meets Claude — 전문적이고 신뢰감 있는Research Cockpit

## 2. Design Language

### Aesthetic
- **방향:** Fintech/Research terminal — 어두운 배경에 데이터viz, 깔끔한 typography
- **키워드:** Professional, Data-rich, Scientific, Trustworthy

### Color Palette
```
Primary:     #1E3A5F (Deep Navy)
Secondary:   #2D5A87 (Steel Blue)  
Accent:      #4ECDC4 (Teal/Cyan)
Background:  #0D1B2A (Dark Navy)
Surface:     #1B2838 (Card Background)
Text:        #E8F1F8 (Light Blue-White)
Warning:     #F4A261 (Amber)
Success:     #2ECC71 (Green)
```

### Typography
- **Headings:** Inter (Bold) — modern, professional
- **Body:** Inter (Regular)
- **Code/Data:** JetBrains Mono — scientific feel

### Motion
- Subtle loading states with typing indicators
- Smooth transitions between sections
- Progress indicators for long-running analyses

## 3. Layout & Structure

### 메인 레이아웃

```
┌─────────────────────────────────────────────────────────────┐
│  🔬 Brown Biotech Research Agent          [Dr. OCM Model]  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ [Tab: 💬 Chat] [📊 Analysis] [📄 Reports] [⚙️ Settings] │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                     │   │
│  │              Chat History / Results                 │   │
│  │                                                     │   │
│  │  [User Query 1]                                     │   │
│  │  [Agent Response with citations]                    │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ [Query Input]                            [Analyze ▶] │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐         │
│  │ 📚 Literature│ │ 🧬 Pipeline  │ │ 📈 Market    │         │
│  └──────────────┘ └──────────────┘ └──────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### Tabs
1. **💬 Chat** — 대화형 인터페이스
2. **📊 Analysis** — 구조화된 분석 결과 (파이프라인, 경쟁사 등)
3. **📄 Reports** — 생성된 보고서 다운로드
4. **⚙️ Settings** — API keys, 모델 선택

## 4. Features & Interactions

### 4.1 Query Types

| Type | Description | Example |
|------|-------------|---------|
| **Literature** | 논문 검색 + 요약 | "TGF-β skin aging 최근 연구" |
| **Pipeline** | 약물 파이프라인 분석 | "GLP-1 agonist 임상단계 현황" |
| **Market** | 시장 분석 | "안티에이징 코스메슈티컬 시장 규모" |
| **Dataset** | MERFISH 데이터 분석 | "이 단일세포 데이터 분석해줘" |
| **Competitor** | 경쟁사 분석 | "Issius pharma 경쟁사 분석" |
| **Custom** | 복합 쿼리 | "NASH 치료제 FDA 승인 전망" |

### 4.2 Response Format

```
┌─────────────────────────────────────────────────────────────┐
│ 📚 Sources: PubMed (12), ClinicalTrials (5), Web (8)       │
│ ⏱️ Generated in 23s | Cost: $0.42                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ ## Executive Summary                                        │
│                                                             │
│ GLP-1 작용제 시장의 2024년 FDA 승인 3개 이후..."             │
│                                                             │
│ ## Key Findings                                             │
│                                                             │
│ 1. **Semaglutide** (Novo Nordisk)                          │
│    - Phase: Marketed                                        │
│    - Mechanism: GLP-1R agonist                             │
│    - Efficacy: HbA1c ↓1.8%, Weight ↓15%                     │
│                                                             │
│ 2. **Tirzepatide** (Lilly)                                  │
│    - Phase: Marketed                                        │
│    - Mechanism: Dual GIP/GLP-1 agonist                     │
│    - Efficacy: HbA1c ↓2.4%, Weight ↓20%                     │
│                                                             │
│ ## Pipeline Status                                          │
│ ...                                                         │
│                                                             │
│ ## Business Implications                                    │
│ ...                                                         │
│                                                             │
│ ──────────────────────────────────────────────────────────  │
│ 📖 References (5)                                           │
│ [1] Naugler et al. (2024) NEJM — PMID: 12345678            │
│ [2] ...                                                     │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 Error Handling

| Error | Response |
|-------|----------|
| API Key missing | "API 키를 설정해주세요. [Settings로 이동]" |
| No results | "검색 결과가 없습니다. 다른 키워드로 시도해주세요." |
| Rate limit | "요청이 많습니다. 30초 후 재시도해주세요." |
| Analysis timeout | "분석 시간이 초과했습니다. 더狭い範囲로 시도해주세요." |

## 5. Technical Architecture

### Stack
```
Frontend:  Streamlit (Python)
Backend:   FastAPI (optional, for production)
AI:       OpenRouter (MiniMax, Gemini, Nemotron)
Search:    TinyFish API (web scraping)
Data:     MERFISH h5ad (local)
Storage:   SQLite (chat history)
```

### Data Flow
```
User Query
    ↓
Router Agent (classify intent)
    ↓
┌─────────────────────────────────────────┐
│ Literature → TinyFish + PubMed search   │
│ Pipeline   → ClinicalTrials + DB        │
│ Market     → Web scraping + reports     │
│ Dataset    → MERFISH scanpy analysis   │
│ Competitor → Web search + analysis     │
└─────────────────────────────────────────┘
    ↓
Synthesis Agent (compile results)
    ↓
Formatted Response + Citations
```

### File Structure
```
brown-biotech-chatbot/
├── app.py                 # Streamlit main
├── agents/
│   ├── __init__.py
│   ├── router.py          # Intent classification
│   ├── literature.py     # PubMed/TinyFish search
│   ├── pipeline.py       # Drug pipeline analysis
│   ├── market.py         # Market research
│   ├── dataset.py        # MERFISH analysis
│   └── synthesizer.py    # Response compilation
├── utils/
│   ├── __init__.py
│   ├── cache.py          # Response caching
│   ├── citations.py      # Citation formatter
│   └── storage.py        # SQLite chat history
├── config/
│   └── settings.py        # API keys, config
├── requirements.txt
└── README.md
```

## 6. MVP Features (v1)

### Must Have
- [x] Chat interface with query input
- [x] Literature search (TinyFish + PubMed)
- [x] Pipeline analysis (pre-built templates)
- [x] Response with citations
- [x] Chat history (session-based)

### Should Have
- [ ] Report download (PDF/Word)
- [ ] MERFISH data upload + analysis
- [ ] Market analysis module

### Future
- [ ] Multi-user auth + cloud storage
- [ ] Real-time pipeline tracking
- [ ] API endpoint for external integration

## 7. API Keys Required

| Service | Key | Purpose |
|---------|-----|---------|
| OpenRouter | `OPENROUTER_API_KEY` | AI models |
| TinyFish | `TINYFISH_API_KEY` | Web research |
| Google | `GOOGLE_API_KEY` | Gemini (optional) |

## 8. Pricing Model (for reference)

| Tier | Price | Requests/mo | Features |
|------|-------|--------------|----------|
| Free | $0 | 10 | Basic queries |
| Pro | ₩99,000/mo | 100 | Full features |
| Enterprise | Custom | Unlimited | API access |

## 9. Success Metrics

- Average response time < 30s (simple queries)
- Citation accuracy > 95%
- User satisfaction > 4/5
- Report download rate > 20%
