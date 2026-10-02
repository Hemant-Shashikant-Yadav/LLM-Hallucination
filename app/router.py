"""
Dynamic Triage Router — Core Orchestrator.

Implements the defense-in-depth pipeline state machine:
1. Receives user query → generates draft via Ollama
2. Extracts hidden states → runs SEP probe → computes H_sem
3. If H_sem < T_safe → return draft immediately (fast path)
4. If knowledge uncertainty → route to Layer 2 (NLI-RAG)
5. If reasoning uncertainty → route to Layer 3 (HalluClean)
6. Aggregates results → returns final verified response with audit trail
"""

from __future__ import annotations

import time

from loguru import logger

from app.config import settings
from app.layer1_triage.activation_hook import ActivationExtractor
from app.layer1_triage.probe_model import TriageEngine
from app.layer2_nli_rag.nli_verifier import NLIVerifier
from app.layer2_nli_rag.proposition_parser import PropositionParser
from app.layer2_nli_rag.search_retriever import SearchRetriever
from app.layer3_halluclean.executor import HalluCleanExecutor
from app.layer3_halluclean.judge import HalluCleanJudge
from app.layer3_halluclean.planner import HalluCleanPlanner
from app.llm_clients.ollama_client import OllamaClient
from app.models.schemas import (
    PipelineLayer,
    PipelineResult,
    QueryRequest,
    QueryResponse,
    TriageDecision,
)


class DefenseInDepthRouter:
    """
    Central orchestrator for the defense-in-depth pipeline.

    Routes queries through the appropriate layers based on
    real-time semantic entropy triage from Layer 1.
    """

    def __init__(self) -> None:
        # Initialize all components
        self._ollama = OllamaClient()

        # Layer 1
        self._activation_extractor = ActivationExtractor()
        self._triage_engine = TriageEngine()

        # Layer 2
        self._proposition_parser = PropositionParser(self._ollama)
        self._search_retriever = SearchRetriever()
        self._nli_verifier = NLIVerifier()

        # Layer 3
        self._planner = HalluCleanPlanner(self._ollama)
        self._executor = HalluCleanExecutor(self._ollama)
        self._judge = HalluCleanJudge(self._ollama)

        self._initialized = False
        logger.info("DefenseInDepthRouter created")

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    async def initialize(self) -> None:
        """
        Initialize all models and resources.

        Called during FastAPI lifespan startup.
        """
        logger.info("Initializing Defense-in-Depth pipeline...")

        # Validate Ollama connection and models
        await self._ollama.validate_models()
        await self._ollama.warmup()

        # Load probe model (Layer 1)
        try:
            await self._activation_extractor.load_model()
            # Initialize the probe with correct dimensions
            # We need to do a dummy extraction to get the hidden dim
            dummy_bundle = await self._activation_extractor.extract_activations(
                "test", max_length=32
            )
            input_dim = dummy_bundle.flattened().shape[0]
            self._triage_engine.initialize_probe(input_dim)
            logger.info("Layer 1 (SEP) initialized | probe_input_dim={}", input_dim)
        except Exception as exc:
            logger.warning(
                "Layer 1 probe model could not be loaded (will use logprob proxy): {}",
                exc,
            )

        # Load NLI model (Layer 2)
        try:
            await self._nli_verifier.load_model()
            logger.info("Layer 2 (NLI) initialized")
        except Exception as exc:
            logger.warning("Layer 2 NLI model could not be loaded: {}", exc)

        # Initialize ChromaDB (optional)
        try:
            await self._search_retriever.init_chromadb()
        except Exception:
            pass

        self._initialized = True
        logger.info("Defense-in-Depth pipeline fully initialized")

    async def shutdown(self) -> None:
        """Clean up resources during shutdown."""
        logger.info("Shutting down Defense-in-Depth pipeline...")
        self._activation_extractor.unload()
        self._nli_verifier.unload()
        self._initialized = False

    # -------------------------------------------------------------------------
    # Main Pipeline
    # -------------------------------------------------------------------------

    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """
        Process a user query through the defense-in-depth pipeline.

        Args:
            request: The incoming query request.

        Returns:
            QueryResponse with the verified answer and optional audit trail.
        """
        start = time.perf_counter()

        logger.info("Processing query: {:.80}...", request.prompt)

        # Step 1: Generate draft response via Ollama
        draft_result = await self._ollama.generate(
            prompt=request.prompt,
            model=request.model,
            temperature=0.0,
        )
        draft_response = draft_result.text

        logger.info(
            "Draft generated | tokens={} | latency={:.1f}ms",
            draft_result.total_tokens,
            draft_result.latency_ms,
        )

        # Check if a specific layer is forced (for testing)
        if request.force_layer is not None:
            return await self._process_forced_layer(
                request, draft_response, draft_result, start
            )

        # Step 2: Layer 1 Triage
        triage_result = await self._run_layer1(draft_response)

        # Step 3: Route based on triage decision
        if triage_result.decision == TriageDecision.SAFE:
            # Fast path — bypass external verification
            total_ms = (time.perf_counter() - start) * 1000

            pipeline_result = PipelineResult(
                query=request.prompt,
                layers_invoked=PipelineLayer.LAYER1_ONLY,
                triage_result=triage_result,
                final_answer=draft_response,
                total_latency_ms=total_ms,
                tokens_generated=draft_result.total_tokens,
            )

            logger.info(
                "FAST PATH | H_sem={:.4f} < T_safe={} | {:.1f}ms total",
                triage_result.semantic_entropy,
                triage_result.threshold_used,
                total_ms,
            )

            return QueryResponse(
                answer=draft_response,
                pipeline_result=pipeline_result if request.include_audit else None,
            )

        elif triage_result.decision == TriageDecision.KNOWLEDGE_UNCERTAIN:
            # Route to Layer 2: NLI-RAG Verification
            return await self._process_layer2(
                request, draft_response, draft_result, triage_result, start
            )

        else:
            # Route to Layer 3: HalluClean Structured Reasoning
            return await self._process_layer3(
                request, draft_response, draft_result, triage_result, start
            )

    # -------------------------------------------------------------------------
    # Layer 1: Semantic Entropy Triage
    # -------------------------------------------------------------------------

    async def _run_layer1(self, draft_response: str):
        """Run Layer 1 triage on the draft response."""
        from app.models.schemas import TriageResult as TR

        try:
            # Extract activations from probe model
            activation_bundle = await self._activation_extractor.extract_activations(
                draft_response
            )

            # Run triage
            return await self._triage_engine.triage(
                activation_bundle=activation_bundle,
                draft_response=draft_response,
            )

        except Exception as exc:
            logger.warning("Layer 1 activation extraction failed: {}", exc)
            # Fallback: use logprob-based estimation
            return await self._triage_engine.triage(
                activation_bundle=None,
                draft_response=draft_response,
                generation_logprobs=None,
            ) if hasattr(self._triage_engine, '_triage_sync') else TR(
                decision=TriageDecision.KNOWLEDGE_UNCERTAIN,
                semantic_entropy=0.7,
                hallucination_probability=0.5,
                threshold_used=settings.t_safe,
                draft_response=draft_response,
                latency_ms=0.0,
            )

    # -------------------------------------------------------------------------
    # Layer 2: NLI-RAG Processing
    # -------------------------------------------------------------------------

    async def _process_layer2(
        self, request, draft_response, draft_result, triage_result, start
    ) -> QueryResponse:
        """Process query through Layer 2: NLI-RAG verification."""
        layer2_start = time.perf_counter()

        logger.info("Routing to Layer 2 (NLI-RAG) | H_sem={:.4f}", triage_result.semantic_entropy)

        # Extract atomic propositions
        propositions = await self._proposition_parser.extract_propositions(
            draft_response
        )

        if not propositions:
            # No propositions to verify, return draft
            total_ms = (time.perf_counter() - start) * 1000
            pipeline_result = PipelineResult(
                query=request.prompt,
                layers_invoked=PipelineLayer.LAYER1_ONLY,
                triage_result=triage_result,
                final_answer=draft_response,
                total_latency_ms=total_ms,
                tokens_generated=draft_result.total_tokens,
            )
            return QueryResponse(
                answer=draft_response,
                pipeline_result=pipeline_result if request.include_audit else None,
            )

        # Search for evidence
        evidence_map = await self._search_retriever.search_for_propositions(
            propositions
        )

        # Verify propositions via NLI
        nli_results = await self._nli_verifier.verify_propositions(
            propositions, evidence_map
        )

        # Aggregate results
        layer2_result = NLIVerifier.aggregate_results(
            propositions, nli_results, draft_response
        )
        layer2_result.latency_ms = (time.perf_counter() - layer2_start) * 1000

        total_ms = (time.perf_counter() - start) * 1000

        pipeline_result = PipelineResult(
            query=request.prompt,
            layers_invoked=PipelineLayer.LAYER1_LAYER2,
            triage_result=triage_result,
            layer2_result=layer2_result,
            final_answer=layer2_result.refined_response,
            total_latency_ms=total_ms,
            tokens_generated=draft_result.total_tokens,
        )

        logger.info(
            "Layer 2 complete | entailed={} | contradicted={} | neutral={} | {:.1f}ms",
            layer2_result.entailed_count,
            layer2_result.contradicted_count,
            layer2_result.neutral_count,
            total_ms,
        )

        return QueryResponse(
            answer=layer2_result.refined_response,
            pipeline_result=pipeline_result if request.include_audit else None,
        )

    # -------------------------------------------------------------------------
    # Layer 3: HalluClean Processing
    # -------------------------------------------------------------------------

    async def _process_layer3(
        self, request, draft_response, draft_result, triage_result, start
    ) -> QueryResponse:
        """Process query through Layer 3: HalluClean structured reasoning."""
        layer3_start = time.perf_counter()

        logger.info(
            "Routing to Layer 3 (HalluClean) | H_sem={:.4f}",
            triage_result.semantic_entropy,
        )

        # Stage 1: Plan
        graph = await self._planner.generate_plan(request.prompt)

        # Stage 2: Execute
        graph, reasoning_traces = await self._executor.execute_plan(graph)

        # Stages 3 & 4: Judge and Refine
        layer3_result = await self._judge.judge_and_refine(graph, reasoning_traces)
        layer3_result.latency_ms = (time.perf_counter() - layer3_start) * 1000

        total_ms = (time.perf_counter() - start) * 1000

        pipeline_result = PipelineResult(
            query=request.prompt,
            layers_invoked=PipelineLayer.LAYER1_LAYER3,
            triage_result=triage_result,
            layer3_result=layer3_result,
            final_answer=layer3_result.refined_response,
            total_latency_ms=total_ms,
            tokens_generated=draft_result.total_tokens,
        )

        logger.info(
            "Layer 3 complete | steps={} | consistent={} | {:.1f}ms",
            len(graph.steps),
            layer3_result.judgment.is_consistent,
            total_ms,
        )

        return QueryResponse(
            answer=layer3_result.refined_response,
            pipeline_result=pipeline_result if request.include_audit else None,
        )

    # -------------------------------------------------------------------------
    # Forced Layer Processing (for benchmarking)
    # -------------------------------------------------------------------------

    async def _process_forced_layer(
        self, request, draft_response, draft_result, start
    ) -> QueryResponse:
        """Process with a forced layer (bypasses triage for testing)."""
        # Create a synthetic triage result
        from app.models.schemas import TriageResult

        triage = TriageResult(
            decision=(
                TriageDecision.SAFE if request.force_layer == 1
                else TriageDecision.KNOWLEDGE_UNCERTAIN if request.force_layer == 2
                else TriageDecision.REASONING_UNCERTAIN
            ),
            semantic_entropy=0.6,
            hallucination_probability=0.5,
            threshold_used=settings.t_safe,
            draft_response=draft_response,
            latency_ms=0.0,
        )

        if request.force_layer == 1:
            total_ms = (time.perf_counter() - start) * 1000
            pipeline_result = PipelineResult(
                query=request.prompt,
                layers_invoked=PipelineLayer.LAYER1_ONLY,
                triage_result=triage,
                final_answer=draft_response,
                total_latency_ms=total_ms,
                tokens_generated=draft_result.total_tokens,
            )
            return QueryResponse(
                answer=draft_response,
                pipeline_result=pipeline_result if request.include_audit else None,
            )
        elif request.force_layer == 2:
            return await self._process_layer2(
                request, draft_response, draft_result, triage, start
            )
        else:
            return await self._process_layer3(
                request, draft_response, draft_result, triage, start
            )
