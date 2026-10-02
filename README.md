# Defense-in-Depth Hallucination Mitigation Framework

A pure-Python, asynchronous, multi-layer framework for detecting and mitigating
Large Language Model (LLM) hallucinations. Developed as part of an M.Tech thesis.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Defense-in-Depth Pipeline                           │
│                                                                       │
│  User Query ──► Ollama (Draft Generation)                             │
│                     │                                                 │
│                     ▼                                                 │
│  ┌──────────────────────────────────┐                                 │
│  │ LAYER 1: Semantic Entropy Probe  │                                 │
│  │ h_t → σ(W·h + b) → H_sem        │                                 │
│  └──────────┬───────────────────────┘                                 │
│             │                                                         │
│     H_sem < T_safe?                                                   │
│    ╱              ╲                                                   │
│  YES              NO                                                  │
│   │          ╱         ╲                                              │
│   │    Knowledge    Reasoning                                         │
│   │    Uncertain    Uncertain                                         │
│   │        │             │                                            │
│   │        ▼             ▼                                            │
│   │  ┌──────────┐  ┌──────────────┐                                   │
│   │  │ LAYER 2  │  │   LAYER 3    │                                   │
│   │  │ NLI-RAG  │  │  HalluClean  │                                   │
│   │  │ DeBERTa  │  │  Plan→Exec   │                                   │
│   │  │  +Search │  │  →Judge→Ref  │                                   │
│   │  └────┬─────┘  └──────┬───────┘                                   │
│   │       │               │                                           │
│   ▼       ▼               ▼                                           │
│  ┌─────────────────────────────────┐                                  │
│  │     Verified Response           │                                  │
│  └─────────────────────────────────┘                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+ (tested on 3.14.2)
- [Ollama](https://ollama.ai/) running locally with `llama3.1:8b`

### Installation

```bash
cd defense_in_depth_framework

# Create virtual environment (optional but recommended)
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Server

```bash
# Start the FastAPI server
python -m app.main

# Or using uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/health` | GET | Health check with config & model status |
| `/api/v1/query` | POST | Full defense-in-depth pipeline |
| `/api/v1/query/layer1` | POST | Layer 1 only (SEP triage) |
| `/api/v1/query/layer2` | POST | Layer 2 only (NLI-RAG) |
| `/api/v1/query/layer3` | POST | Layer 3 only (HalluClean) |

### Example Query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?", "include_audit": true}'
```

### Run Benchmarks

```bash
# Start the server first, then in another terminal:
python -m evaluation.benchmark_runner --max-samples 50
```

Results are saved to `evaluation/results/` as CSV, JSON, and LaTeX tables.

## Configuration

All settings can be configured via environment variables or a `.env` file:

| Variable | Default | Description |
|---|---|---|
| `DID_T_SAFE` | `0.55` | Safety threshold for semantic entropy |
| `DID_NLI_CONFIDENCE_THRESHOLD` | `0.8` | Min confidence for NLI entailment |
| `DID_OLLAMA_MODEL` | `llama3.1:8b` | Default Ollama model |
| `DID_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `DID_PROBE_MODEL_NAME` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Probe model |
| `DID_NLI_MODEL_NAME` | `cross-encoder/nli-deberta-v3-base` | NLI model |
| `DID_SEARCH_BACKEND` | `duckduckgo` | Search backend |
| `DID_HOST` | `0.0.0.0` | Server host |
| `DID_PORT` | `8000` | Server port |

## Project Structure

```
defense_in_depth_framework/
├── app/
│   ├── main.py                     # FastAPI application entrypoint
│   ├── config.py                   # Pydantic BaseSettings configuration
│   ├── router.py                   # Dynamic triage state machine (orchestrator)
│   ├── models/
│   │   └── schemas.py              # All Pydantic v2 data models
│   ├── layer1_triage/
│   │   ├── activation_hook.py      # PyTorch hidden-state extraction
│   │   └── probe_model.py          # Semantic Entropy Probe (SEP)
│   ├── layer2_nli_rag/
│   │   ├── proposition_parser.py   # Atomic Proposition extractor
│   │   ├── search_retriever.py     # Async web + vector search
│   │   └── nli_verifier.py         # DeBERTa-v3 NLI classifier
│   ├── layer3_halluclean/
│   │   ├── planner.py              # Symbolic task planner (Stage 1)
│   │   ├── executor.py             # Plan-guided reasoner (Stage 2)
│   │   └── judge.py                # Judge + refiner (Stages 3-4)
│   └── llm_clients/
│       └── ollama_client.py        # Async Ollama SDK wrapper
├── evaluation/
│   ├── benchmark_runner.py         # Batch testing harness
│   ├── metrics.py                  # Factual accuracy, AUROC, latency
│   └── plot_generator.py           # Publication-ready Matplotlib charts
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Key Technologies

- **FastAPI** + **AsyncIO** — High-performance async web framework
- **PyTorch** + **HuggingFace Transformers** — Neural network inference
- **Ollama** — Local LLM serving
- **DeBERTa-v3** — NLI-based claim verification
- **DuckDuckGo Search** — Free web search for evidence retrieval
- **ChromaDB** — Local vector database for cached documents
- **Pydantic v2** — Data validation and settings management
- **Matplotlib** — Publication-quality visualization

## License

MIT
