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

- Python 3.11+ (tested on 3.12, Apple M4 with MPS)
- [Ollama](https://ollama.ai/) running locally with these models:

```bash
ollama pull llama2:7b      # generator (baseline)
ollama pull llama3.1:8b    # verifier for Layer 2 / Layer 3
ollama pull qwen2.5:7b     # independent TruthfulQA judge (evaluation only)
```

Layer 1 sends 4 parallel samples. Start Ollama with parallel slots for best speed:

```bash
OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_CONTEXT_LENGTH=4096 \
OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve
```

### Installation

```bash
cd defense_in_depth_framework

# Create virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate   # macOS / Linux  (.venv\Scripts\activate on Windows)

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
python -m evaluation.benchmark_runner --n 100 \
    --configs baseline full_framework reference_verifier \
    --out-dir evaluation/results/phase1_llama2_all_configs

# Score / report existing CSVs again (TruthfulQA judge + statistics)
python -m evaluation.benchmark_runner --judge-only --out-dir evaluation/results/phase1_llama2_all_configs
```

- `baseline` is the exact greedy draft the framework received, so every comparison is paired.
- GSM8K uses an exact final-answer match. TruthfulQA uses an independent judge
  (`qwen2.5:7b`) with the correct and incorrect reference answers, plus ROUGE-L accuracy.
- The report gives 95% bootstrap CIs, exact McNemar p-values, latency and tokens over all LLM calls.
- The runner appends rows as it goes and resumes after an interruption.

### Figures and Presentation

```bash
python -m evaluation.presentation_plots --results evaluation/results/phase1_llama2_all_configs
python -m evaluation.build_presentation --results evaluation/results/phase1_llama2_all_configs \
    --source Hallucination_Mitigation_Presentation_Beautified.pptx \
    --output Hallucination_Mitigation_Phase1_Review.pptx
```

## Configuration

All settings can be configured via environment variables or a `.env` file:

| Variable | Default | Description |
|---|---|---|
| `DID_TRIAGE_MODE` | `semantic_entropy` | Layer 1 signal: sampled semantic entropy or `surrogate_probe` |
| `DID_SE_NUM_SAMPLES` | `4` | Extra generator samples for semantic entropy |
| `DID_T_SAFE` | `0.35` | Safety threshold for factual (knowledge) queries |
| `DID_T_SAFE_REASONING` | `0.0` | Safety threshold for quantitative queries (0 = always verify) |
| `DID_HALLUCLEAN_EXECUTOR_MODE` | `single_pass` | Layer 3 plan execution: `single_pass` or `per_step` |
| `DID_OLLAMA_JUDGE_MODEL` | `llama3.1:8b` | Verifier model for Layer 2 / Layer 3 |
| `DID_NLI_CONFIDENCE_THRESHOLD` | `0.8` | Min confidence for NLI entailment |
| `DID_OLLAMA_MODEL` | `llama2:7b` | Default Ollama model |
| `DID_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `DID_PROBE_MODEL_NAME` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Probe model |
| `DID_NLI_MODEL_NAME` | `cross-encoder/nli-deberta-v3-base` | NLI model |
| `DID_SEARCH_BACKEND` | `duckduckgo` | Search backend (uses the `ddgs` package) |
| `DID_HOST` | `0.0.0.0` | Server host |
| `DID_PORT` | `8000` | Server port |

## Project Structure

```
defense_in_depth_framework/
├── app/
│   ├── main.py                     # FastAPI application entrypoint
│   ├── config.py                   # Pydantic BaseSettings configuration
│   ├── router.py                   # Dynamic triage state machine (orchestrator)
│   ├── answer_utils.py             # Final-answer parsing and task-type detection
│   ├── models/
│   │   └── schemas.py              # All Pydantic v2 data models
│   ├── layer1_triage/
│   │   ├── activation_hook.py      # PyTorch hidden-state extraction
│   │   ├── probe_model.py          # Semantic Entropy Probe (SEP, needs trained weights)
│   │   └── semantic_entropy.py     # Sampled semantic entropy triage (default)
│   ├── layer2_nli_rag/
│   │   ├── proposition_parser.py   # Atomic Proposition extractor
│   │   ├── search_retriever.py     # Async web + vector search
│   │   ├── nli_verifier.py         # DeBERTa-v3 NLI classifier
│   │   └── reviser.py              # Evidence-grounded rewrite + Chain-of-Verification
│   ├── layer3_halluclean/
│   │   ├── planner.py              # Symbolic task planner (Stage 1)
│   │   ├── executor.py             # Plan-guided reasoner (Stage 2)
│   │   └── judge.py                # Judge + refiner (Stages 3-4)
│   └── llm_clients/
│       └── ollama_client.py        # Async Ollama SDK wrapper
├── evaluation/
│   ├── benchmark_runner.py         # Batch testing harness
│   ├── metrics.py                  # Factual accuracy, AUROC, latency
│   ├── truthfulqa_judge.py         # Independent LLM judge + ROUGE-L accuracy
│   ├── plot_generator.py           # Publication-ready Matplotlib charts
│   ├── presentation_plots.py       # Figures for the Phase 1 review deck
│   └── build_presentation.py       # Builds the Phase 1 review deck from results
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Key Technologies

- **FastAPI** + **AsyncIO** — High-performance async web framework
- **PyTorch** + **HuggingFace Transformers** — Neural network inference
- **Ollama** — Local LLM serving
- **DeBERTa-v3** — NLI-based claim verification
- **DuckDuckGo Search (`ddgs`)** — Free web search for evidence retrieval
- **ChromaDB** — Local vector database for cached documents
- **Pydantic v2** — Data validation and settings management
- **Matplotlib** — Publication-quality visualization

## License

MIT
