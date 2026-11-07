# app/models/schemas.py — data shapes passed between layers

Pydantic models for the API request/response and for everything the layers hand to each
other. Together they form the **audit trail** (`PipelineResult`) returned with
`include_audit: true`, which the evaluation scripts save to `*_audit.jsonl`.

| Lines | Model | What it holds |
|---|---|---|
| 28–33 | `TriageDecision` | Layer 1 verdict: `SAFE` (fast path, return the draft), `KNOWLEDGE_UNCERTAIN` (→ Layer 2), `REASONING_UNCERTAIN` (→ Layer 3). |
| 36–41 | `NLILabel` | entailment / contradiction / neutral. |
| 44–49 | `ConsensusVerdict` | accept / reject / neutral for a claim after weighing all evidence passages. |
| 52–58 | `PipelineLayer` | Which path ran: `layer1_only`, `layer1_layer2`, `layer1_layer3`. |
| 66–83 | **`QueryRequest`** | API input: `prompt`; `model` (defaults to the pipeline model from settings); `force_layer` 1–3 to skip triage; `include_audit` (default true). |
| 86–93 | **`QueryResponse`** | API output: `answer` plus the optional `pipeline_result`. |
| 101–110 | `ActivationData` | Metadata of probe activations (surrogate-probe mode only). |
| 113–139 | `ErrorClassification` | Output of the dual-head probe: predicted entropy, knowledge vs reasoning logits, routing confidence (surrogate-probe mode only). |
| 142–164 | **`TriageResult`** | Layer 1 output: decision, semantic entropy `H_sem`, hallucination probability, threshold used, the draft, latency. |
| 172–180 | `AtomicProposition` | One checkable claim extracted from the draft, with the sentence it came from. |
| 183–194 | `SearchResult` | One evidence passage: title, URL, snippet, relevance weight (web 0.7, Wikipedia 0.9). |
| 197–218 | `NLIPassageScore` | NLI probabilities of one passage against one claim, and its weighted contribution `w·(P(entail) − P(contradict))`. |
| 221–250 | `ConflictAwareNLIResult` | Consensus over all passages: `S = Σ w·(P(entail) − P(contradict))`, verdict by the accept/reject thresholds, per-passage breakdown. |
| 253–276 | `NLIResult` | Final label for one claim, its confidence, the evidence used, and the consensus details. |
| 279–294 | **`Layer2Result`** | Claims, their NLI results, counts entailed/contradicted/neutral, the revised answer, searches used, latency. |
| 302–312 | `PlanNode` | One step of a Layer 3 plan (status, output). |
| 315–320 | `ExecutionGraph` | The plan: the question plus its steps. |
| 323–337 | `JudgmentResult` | Layer 3 verdict: did the answer hold up (`is_consistent`), what changed (`violations`), confidence (share of votes). |
| 340–352 | **`Layer3Result`** | Plan, reasoning traces (e.g. `[Samples] draft=4 answers=[…]`), judgment, final answer, latency. |
| 360–389 | **`PipelineResult`** | The full audit trail: path taken, triage result, Layer 2/3 results, final answer, latency, **tokens across all LLM calls**, draft tokens and latency, whether a layer changed the answer, **trace ID**, **task type** (math / factual / time_sensitive), **degradations**, timestamp. |
| 397–436 | `BenchmarkRecord`, `MetricsReport` | Older evaluation record types (Phase 1); current scripts write CSVs instead. |
