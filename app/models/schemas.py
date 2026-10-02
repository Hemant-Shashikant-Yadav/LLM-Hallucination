"""
Pydantic v2 data models for the Defense-in-Depth Hallucination Mitigation Framework.

All shared request/response schemas, internal pipeline data structures,
and evaluation record types are defined here.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# Enumerations
# =============================================================================


class TriageDecision(str, Enum):
    """Triage outcome from Layer 1 Semantic Entropy Probe."""

    SAFE = "safe"
    KNOWLEDGE_UNCERTAIN = "knowledge_uncertain"
    REASONING_UNCERTAIN = "reasoning_uncertain"


class NLILabel(str, Enum):
    """Natural Language Inference classification labels."""

    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"


class PipelineLayer(str, Enum):
    """Which pipeline layer(s) processed a query."""

    LAYER1_ONLY = "layer1_only"
    LAYER1_LAYER2 = "layer1_layer2"
    LAYER1_LAYER3 = "layer1_layer3"
    FULL_PIPELINE = "full_pipeline"


# =============================================================================
# API Request / Response
# =============================================================================


class QueryRequest(BaseModel):
    """Incoming user query to the framework."""

    prompt: str = Field(..., min_length=1, description="User prompt / question")
    model: str = Field(default="llama3.1:8b", description="Ollama model identifier")
    force_layer: int | None = Field(
        default=None,
        ge=1,
        le=3,
        description="Force a specific layer (1-3) for testing; None = auto-triage",
    )
    include_audit: bool = Field(
        default=True,
        description="Include detailed audit trail in response",
    )


class QueryResponse(BaseModel):
    """Final response from the framework."""

    answer: str = Field(..., description="The final verified / mitigated answer")
    pipeline_result: PipelineResult | None = Field(
        default=None,
        description="Detailed pipeline audit trail (if requested)",
    )


# =============================================================================
# Layer 1: Semantic Entropy Probe
# =============================================================================


class ActivationData(BaseModel):
    """Hidden-state activation data extracted from a transformer forward pass."""

    model_name: str
    layer_indices: list[int] = Field(
        description="Which transformer layers the activations came from"
    )
    hidden_dim: int = Field(description="Dimensionality d of each hidden-state vector")
    token_count: int = Field(description="Number of tokens in the generated sequence")
    # Actual tensors are passed in-memory; this model tracks metadata only.


class TriageResult(BaseModel):
    """Result of the Layer 1 triage evaluation."""

    decision: TriageDecision
    semantic_entropy: float = Field(
        ge=0.0,
        description="Computed semantic entropy H_sem",
    )
    hallucination_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="P(Hallucination | h_t) from the probe",
    )
    threshold_used: float = Field(description="T_safe threshold value used")
    draft_response: str = Field(description="The initial draft LLM response")
    latency_ms: float = Field(ge=0.0, description="Layer 1 processing time")


# =============================================================================
# Layer 2: Hypothesis-Testing RAG + NLI
# =============================================================================


class AtomicProposition(BaseModel):
    """A single atomic, independently verifiable claim extracted from text."""

    id: int = Field(description="Proposition index within the parent response")
    text: str = Field(description="The atomic claim text")
    source_span: str = Field(
        default="",
        description="Original text span this proposition was extracted from",
    )


class SearchResult(BaseModel):
    """A single search result from web or vector retrieval."""

    title: str = Field(default="")
    url: str = Field(default="")
    snippet: str = Field(description="Relevant text excerpt from the source")
    relevance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Retrieval relevance score",
    )


class NLIResult(BaseModel):
    """NLI verification result for a single atomic proposition."""

    proposition: AtomicProposition
    label: NLILabel
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Softmax probability of the predicted label",
    )
    premise_used: str = Field(
        description="The source evidence text used as NLI premise"
    )
    all_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Raw softmax scores for all three NLI classes",
    )


class Layer2Result(BaseModel):
    """Aggregated result from Layer 2 processing."""

    propositions: list[AtomicProposition]
    nli_results: list[NLIResult]
    entailed_count: int = Field(default=0)
    contradicted_count: int = Field(default=0)
    neutral_count: int = Field(default=0)
    refined_response: str = Field(
        description="Response after removing contradicted claims"
    )
    search_results_used: int = Field(
        default=0,
        description="Number of search results retrieved",
    )
    latency_ms: float = Field(ge=0.0, description="Layer 2 processing time")


# =============================================================================
# Layer 3: HalluClean Structured Reasoning
# =============================================================================


class PlanNode(BaseModel):
    """A single step in the HalluClean symbolic execution graph."""

    step_id: int
    description: str = Field(description="Natural language description of this step")
    dependencies: list[int] = Field(
        default_factory=list,
        description="Step IDs that must complete before this one",
    )
    status: str = Field(default="pending", description="pending | running | completed | failed")
    output: str = Field(default="", description="Step execution output")


class ExecutionGraph(BaseModel):
    """The full HalluClean symbolic execution plan."""

    query: str = Field(description="Original user query")
    steps: list[PlanNode] = Field(default_factory=list)
    total_steps: int = Field(default=0)


class JudgmentResult(BaseModel):
    """Result from the HalluClean judge stage."""

    is_consistent: bool = Field(
        description="Whether the synthesized answer is consistent with constraints"
    )
    violations: list[str] = Field(
        default_factory=list,
        description="List of constraint violations found",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Judge confidence in the final answer",
    )


class Layer3Result(BaseModel):
    """Aggregated result from Layer 3 HalluClean processing."""

    execution_graph: ExecutionGraph
    reasoning_traces: list[str] = Field(
        default_factory=list,
        description="Step-by-step reasoning outputs",
    )
    judgment: JudgmentResult
    refined_response: str = Field(
        description="Final polished response after structured reasoning"
    )
    latency_ms: float = Field(ge=0.0, description="Layer 3 processing time")


# =============================================================================
# Pipeline Audit Trail
# =============================================================================


class PipelineResult(BaseModel):
    """Complete audit trail of a query through the defense-in-depth pipeline."""

    query: str
    layers_invoked: PipelineLayer
    triage_result: TriageResult
    layer2_result: Layer2Result | None = None
    layer3_result: Layer3Result | None = None
    final_answer: str
    total_latency_ms: float = Field(ge=0.0)
    tokens_generated: int = Field(default=0, ge=0)
    timestamp: float = Field(default_factory=time.time)


# =============================================================================
# Evaluation & Benchmarking
# =============================================================================


class BenchmarkRecord(BaseModel):
    """A single benchmark evaluation record."""

    dataset: str = Field(description="Dataset name (e.g., 'truthfulqa', 'halueval')")
    question_id: str
    question: str
    reference_answer: str = Field(default="")
    baseline_answer: str = Field(description="Zero-shot LLM answer without framework")
    framework_answer: str = Field(description="Answer after framework processing")
    is_hallucinated_baseline: bool = Field(default=False)
    is_hallucinated_framework: bool = Field(default=False)
    layers_invoked: PipelineLayer = Field(default=PipelineLayer.LAYER1_ONLY)
    baseline_latency_ms: float = Field(default=0.0)
    framework_latency_ms: float = Field(default=0.0)
    baseline_tokens: int = Field(default=0)
    framework_tokens: int = Field(default=0)


class MetricsReport(BaseModel):
    """Aggregated metrics from a benchmark run."""

    dataset: str
    total_samples: int
    factual_accuracy_baseline: float = Field(ge=0.0, le=1.0)
    factual_accuracy_framework: float = Field(ge=0.0, le=1.0)
    accuracy_improvement: float = Field(description="Absolute improvement in accuracy")
    auroc: float = Field(ge=0.0, le=1.0, description="AUROC for probe classification")
    mean_latency_baseline_ms: float
    mean_latency_framework_ms: float
    latency_delta_pct: float = Field(description="Percentage change in latency")
    total_tokens_baseline: int
    total_tokens_framework: int
    token_reduction_pct: float = Field(
        description="Percentage reduction in token expenditure"
    )
    layer1_bypass_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of queries resolved at Layer 1 (fast path)",
    )
