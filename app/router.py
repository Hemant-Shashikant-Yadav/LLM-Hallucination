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

import asyncio
import contextvars
import time

from loguru import logger

from app.answer_utils import (
    Calculation,
    calculate,
    extract_final_answer,
    is_numeric_reasoning_query,
    numbers_match,
)
from app.config import settings
from app.layer1_triage.activation_hook import ActivationExtractor
from app.layer1_triage.probe_model import SurrogateExtractor, TriageEngine
from app.layer1_triage.task_router import TaskRouter
from app.layer1_triage.semantic_entropy import SemanticEntropyTriage
from app.layer2_nli_rag.nli_verifier import NLIVerifier
from app.layer2_nli_rag.proposition_parser import PropositionParser
from app.layer2_nli_rag.reviser import EvidenceReviser
from app.layer2_nli_rag.search_retriever import SearchRetriever
from app.layer3_halluclean.executor import HalluCleanExecutor
from app.layer3_halluclean.judge import HalluCleanJudge
from app.layer3_halluclean.planner import HalluCleanPlanner
from app.layer3_halluclean.program_verifier import ProgramVerifier
from app.llm_clients.ollama_client import OllamaClient, _token_meter, start_token_meter
from app.telemetry import (
    current_trace_id,
    degradations,
    record_degradation,
    request_trace,
    stage,
)
from app.models.schemas import (
    ExecutionGraph,
    Layer3Result,
    PlanNode,
    PipelineLayer,
    PipelineResult,
    QueryRequest,
    QueryResponse,
    TriageDecision,
)


# Task type of the query being processed, for the audit trail
_task_type: contextvars.ContextVar[str] = contextvars.ContextVar("task_type", default="")


class DefenseInDepthRouter:
    """
    Central orchestrator for the defense-in-depth pipeline.

    Routes queries through the appropriate layers based on
    real-time semantic entropy triage from Layer 1.
    """

    def __init__(self) -> None:
        # Initialize all components
        self._ollama = OllamaClient()
        
        # Asymmetric Verification: Use a stronger model (llama3.1:8b) for the Judge/Refiner
        # This allows the framework to perfectly correct weak baseline models.
        self._judge_ollama = OllamaClient(model=settings.ollama_judge_model)

        # Layer 1: Surrogate Probing is the primary thesis architecture
        self._surrogate_extractor = SurrogateExtractor(
            device=settings.device,
            use_fp16=settings.use_fp16,
        )
        self._activation_extractor: ActivationExtractor | None = None
        self._triage_engine = TriageEngine(
            device=settings.device,
        )

        # Layer 2
        self._proposition_parser = PropositionParser(self._judge_ollama)
        self._search_retriever = SearchRetriever()
        self._nli_verifier = NLIVerifier(
            accept_threshold=settings.nli_accept_threshold,
            reject_threshold=settings.nli_reject_threshold,
            device=settings.device,
        )
        self._reviser = EvidenceReviser(self._judge_ollama)

        # Layer 1 (primary): sampled semantic entropy over the generator
        self._se_triage = SemanticEntropyTriage(self._ollama, self._nli_verifier)
        self._task_router = TaskRouter(self._ollama)

        # Layer 3
        self._planner = HalluCleanPlanner(self._judge_ollama)
        self._executor = HalluCleanExecutor(self._judge_ollama)
        self._judge = HalluCleanJudge(self._judge_ollama)
        self._program_verifier = ProgramVerifier(self._judge_ollama)

        self._initialized = False
        logger.info("DefenseInDepthRouter created | device={}", settings.device)

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    async def initialize(self) -> None:
        """
        Initialize all models and resources.

        Called during FastAPI lifespan startup.
        """
        logger.info("Initializing Defense-in-Depth pipeline...")

        # Load NLI model first (Layer 2, also Layer 1 semantic clustering) so that
        # an Ollama outage at startup does not also leave NLI unloaded.
        if not self._nli_verifier.is_loaded:
            try:
                await self._nli_verifier.load_model()
                logger.info("NLI model initialized | device={}", settings.device)
            except Exception as exc:
                logger.error(
                    "NLI model could not be loaded — Layer 1 will escalate every "
                    "open-domain query and Layer 2 NLI will mark every claim neutral: {}",
                    exc,
                )

        # Generator is mandatory: without it no query can run (raises).
        await self._ollama.validate_models()
        await self._ollama.warmup()

        # Judge is optional, but if it is unusable we must actually fall back,
        # otherwise every Layer 2/3 call fails and is swallowed downstream.
        try:
            await self._judge_ollama.validate_models()
            await self._judge_ollama.warmup()
        except Exception as exc:
            logger.error(
                "Judge model {} unusable ({}: {}) — Layer 2/3 will use generator {} instead",
                self._judge_ollama.default_model, type(exc).__name__, exc, settings.ollama_model,
            )
            self._judge_ollama.use_model(settings.ollama_model)

        if settings.triage_mode == "semantic_entropy":
            logger.info(
                "Layer 1 mode: sampled semantic entropy (K={}) — surrogate probe not loaded",
                settings.se_num_samples,
            )
            await self._init_retriever()
            self._initialized = True
            logger.info("Defense-in-Depth pipeline fully initialized")
            return

        # Load surrogate extractor (Layer 1 — legacy probe architecture)
        surrogate_loaded = False
        try:
            await self._surrogate_extractor.load_model()
            self._triage_engine.set_surrogate(self._surrogate_extractor)

            # Initialize dual-head classifier with surrogate hidden dim
            surrogate_dim = self._surrogate_extractor.hidden_dim
            self._triage_engine.initialize_probe(surrogate_dim)
            surrogate_loaded = True
            logger.info(
                "Layer 1 surrogate probe attached | hidden_dim={} | device={}",
                surrogate_dim,
                settings.device,
            )
        except Exception as exc:
            logger.warning(
                "Surrogate extractor could not be loaded: {}", exc
            )

        # Fallback: Load legacy activation extractor ONLY if surrogate failed
        if not surrogate_loaded:
            try:
                self._activation_extractor = ActivationExtractor(
                    device=settings.device,
                    use_fp16=settings.use_fp16,
                )
                await self._activation_extractor.load_model()
                dummy_bundle = await self._activation_extractor.extract_activations(
                    "test", max_length=32
                )
                input_dim = dummy_bundle.flattened().shape[0]
                self._triage_engine.initialize_probe(input_dim)
                logger.info("Layer 1 legacy probe initialized | probe_input_dim={}", input_dim)
            except Exception as exc:
                logger.warning(
                    "Layer 1 legacy probe could not be loaded (will use logprob proxy): {}",
                    exc,
                )

        await self._init_retriever()
        self._initialized = True
        logger.info("Defense-in-Depth pipeline fully initialized")

    async def _init_retriever(self) -> None:
        """Initialize ChromaDB (optional)."""
        try:
            await self._search_retriever.init_chromadb()
        except Exception as exc:
            logger.warning("ChromaDB unavailable, using web search only: {}", exc)

    async def shutdown(self) -> None:
        """Clean up resources during shutdown."""
        logger.info("Shutting down Defense-in-Depth pipeline...")
        if self._activation_extractor:
            self._activation_extractor.unload()
        self._surrogate_extractor.unload()
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
        with request_trace():
            async with stage(
                "pipeline", model=request.model, force_layer=request.force_layer
            ) as out:
                response = await self._process_query(request)
                pr = response.pipeline_result
                out.update(
                    layers=pr.layers_invoked.value if pr else "?",
                    degradations=len(degradations()),
                )
                return response

    async def _process_query(self, request: QueryRequest) -> QueryResponse:
        start = time.perf_counter()
        start_token_meter()
        _task_type.set("")

        logger.info("Processing query: {:.80}...", request.prompt)

        # Step 1: Generate draft response via Ollama
        async with stage("draft", model=request.model) as out:
            draft_result = await self._ollama.generate(
                prompt=request.prompt,
                model=request.model,
                temperature=0.0,
            )
            draft_response = draft_result.text
            out.update(tokens=draft_result.total_tokens, done_reason=draft_result.done_reason)
        if not draft_response.strip():
            record_degradation("draft", "generator returned an empty draft")

        # Check if a specific layer is forced (for testing)
        if request.force_layer is not None:
            return await self._process_forced_layer(
                request, draft_response, draft_result, start
            )

        # Step 2a: What kind of question is this? (math / factual / time_sensitive)
        async with stage("layer1.task_route") as out:
            task_type = await self._task_router.classify(request.prompt)
            exact = calculate(request.prompt) if task_type == "math" else None
            out.update(task_type=task_type, exact=exact.display if exact else None)
        _task_type.set(task_type)

        # Step 2b: Layer 1 Triage
        se_estimate = None
        async with stage("layer1.triage", mode=settings.triage_mode) as out:
            if settings.triage_mode == "semantic_entropy":
                triage_result, se_estimate = await self._se_triage.triage(
                    request.prompt, draft_response, model=request.model, task_type=task_type
                )
            else:
                triage_result = await self._run_layer1(
                    request.prompt, draft_response
                )
            # An exactly computable query whose draft disagrees with the exact value
            # is wrong regardless of how consistent the samples were.
            if exact is not None and not numbers_match(
                extract_final_answer(draft_response), exact.value
            ):
                triage_result.decision = TriageDecision.REASONING_UNCERTAIN
            out.update(
                decision=triage_result.decision.value,
                H_sem=round(triage_result.semantic_entropy, 4),
            )

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
                **self._cost_fields(draft_result),
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
                request, draft_response, draft_result, triage_result, start,
                task_type=task_type,
            )

        else:
            # Route to Layer 3: HalluClean Structured Reasoning
            return await self._process_layer3(
                request, draft_response, draft_result, triage_result, start,
                se_estimate=se_estimate, task_type=task_type, exact=exact,
            )

    @staticmethod
    def _cost_fields(draft_result) -> dict:
        """Token / latency accounting plus the request's trace ID and degradations."""
        meter = _token_meter.get()
        total = sum(meter) if meter else draft_result.total_tokens
        return {
            "tokens_generated": total,
            "draft_tokens": draft_result.total_tokens,
            "draft_latency_ms": draft_result.latency_ms,
            "trace_id": current_trace_id(),
            "degradations": degradations(),
            "task_type": _task_type.get(),
        }

    # -------------------------------------------------------------------------
    # Layer 1: Semantic Entropy Triage
    # -------------------------------------------------------------------------

    async def _run_layer1(
        self,
        prompt: str,
        draft_response: str,
    ):
        """Run Layer 1 triage on the draft response."""
        from app.models.schemas import TriageResult as TR

        # Primary: Surrogate Extractor is loaded and active
        if self._surrogate_extractor.is_loaded:
            try:
                return await self._triage_engine.triage(
                    prompt=prompt,
                    draft_response=draft_response,
                    activation_bundle=None,
                )
            except Exception as exc:
                logger.warning("Layer 1 surrogate triage failed: {}", exc)

        # Fallback: legacy activation extractor if loaded
        if self._activation_extractor is not None:
            try:
                activation_bundle = await self._activation_extractor.extract_activations(
                    draft_response
                )
                return await self._triage_engine.triage(
                    prompt=prompt,
                    draft_response=draft_response,
                    activation_bundle=activation_bundle,
                )
            except Exception as exc:
                logger.warning("Layer 1 legacy activation extraction failed: {}", exc)

        # Fallback: pure proxy mode
        try:
            return await self._triage_engine.triage(
                prompt=prompt,
                draft_response=draft_response,
                activation_bundle=None,
                generation_logprobs=None,
            )
        except Exception:
            return TR(
                decision=TriageDecision.KNOWLEDGE_UNCERTAIN,
                semantic_entropy=0.7,
                hallucination_probability=0.5,
                error_classification=None,
                threshold_used=settings.t_safe,
                draft_response=draft_response,
                latency_ms=0.0,
            )

    # -------------------------------------------------------------------------
    # Layer 2: NLI-RAG Processing
    # -------------------------------------------------------------------------

    async def _process_layer2(
        self, request, draft_response, draft_result, triage_result, start,
        task_type: str = "factual",
    ) -> QueryResponse:
        """Process query through Layer 2: NLI-RAG verification."""
        layer2_start = time.perf_counter()

        logger.info("Routing to Layer 2 (NLI-RAG) | H_sem={:.4f}", triage_result.semantic_entropy)

        # Extract atomic propositions; search evidence for the question itself in parallel
        async with stage("layer2.extract_and_search") as out:
            propositions, question_evidence = await asyncio.gather(
                self._proposition_parser.extract_propositions(
                    draft_response, max_propositions=6
                ),
                self._search_retriever.search(request.prompt),
            )
            out.update(propositions=len(propositions), question_evidence=len(question_evidence))

        if not propositions and not question_evidence:
            # Nothing to verify against, return draft — but say so
            record_degradation(
                "layer2", "no propositions and no evidence; returning unverified draft"
            )
            total_ms = (time.perf_counter() - start) * 1000
            pipeline_result = PipelineResult(
                query=request.prompt,
                layers_invoked=PipelineLayer.LAYER1_LAYER2,
                triage_result=triage_result,
                final_answer=draft_response,
                total_latency_ms=total_ms,
                **self._cost_fields(draft_result),
            )
            return QueryResponse(
                answer=draft_response,
                pipeline_result=pipeline_result if request.include_audit else None,
            )

        # Search for evidence
        async with stage("layer2.search_propositions") as out:
            evidence_map = await self._search_retriever.search_for_propositions(
                propositions
            )
            out.update(with_evidence=sum(1 for v in evidence_map.values() if v))
            # Per-claim searches for long, specific sentences often come back empty
            # (39% of claims on the dev slice). Optionally check every claim against the
            # question's own results too. Off by default: on dev it raised coverage to
            # 89% but also produced spurious contradictions and more harmful edits.
            if question_evidence and settings.layer2_question_evidence_for_claims:
                for prop in propositions:
                    own = evidence_map.get(prop.id, [])
                    seen = {e.url for e in own}
                    evidence_map[prop.id] = own + [e for e in question_evidence if e.url not in seen]
        if propositions and not any(evidence_map.values()) and not question_evidence:
            record_degradation("layer2.search", "web search returned no evidence for any claim")

        # Verify propositions via NLI
        async with stage("layer2.nli") as out:
            nli_results = await self._nli_verifier.verify_propositions(
                propositions, evidence_map
            )
            out.update(labels=[r.label.value for r in nli_results])

        # Aggregate results
        layer2_result = NLIVerifier.aggregate_results(
            propositions, nli_results, draft_response
        )

        # Evidence-grounded revision of the draft
        async with stage("layer2.revise"):
            layer2_result.refined_response = await self._reviser.revise(
                question=request.prompt,
                draft=draft_response,
                nli_results=nli_results,
                question_evidence=question_evidence,
                task_type=task_type,
            )
        layer2_result.search_results_used += len(question_evidence)
        layer2_result.latency_ms = (time.perf_counter() - layer2_start) * 1000

        total_ms = (time.perf_counter() - start) * 1000

        pipeline_result = PipelineResult(
            query=request.prompt,
            layers_invoked=PipelineLayer.LAYER1_LAYER2,
            triage_result=triage_result,
            layer2_result=layer2_result,
            final_answer=layer2_result.refined_response,
            total_latency_ms=total_ms,
            layer_answer_changed=layer2_result.refined_response != draft_response,
            **self._cost_fields(draft_result),
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
        self, request, draft_response, draft_result, triage_result, start,
        se_estimate=None, task_type: str | None = None, exact: Calculation | None = None,
    ) -> QueryResponse:
        """Process query through Layer 3: HalluClean structured reasoning."""
        layer3_start = time.perf_counter()
        is_math = (
            task_type == "math" if task_type is not None
            else is_numeric_reasoning_query(request.prompt)
        )

        logger.info(
            "Routing to Layer 3 (HalluClean) | H_sem={:.4f}",
            triage_result.semantic_entropy,
        )

        if exact is not None:
            # Plain arithmetic: an exact calculation replaces planning and voting
            graph = ExecutionGraph(
                query=request.prompt,
                steps=[PlanNode(step_id=1, description=f"Evaluate {exact.expression}",
                                dependencies=[], status="completed", output=exact.display)],
                total_steps=1,
            )
            refined, reasoning_traces, judgment = await self._program_verifier.verify(
                request.prompt, "", [draft_response], draft_response, exact=exact
            )
            layer3_result = Layer3Result(
                execution_graph=graph, reasoning_traces=reasoning_traces,
                judgment=judgment, refined_response=refined, latency_ms=0.0,
            )
            return self._finish_layer3(
                request, draft_response, draft_result, triage_result, start,
                layer3_start, graph, layer3_result,
            )

        # Stage 1: Plan
        async with stage("layer3.plan") as out:
            graph = await self._planner.generate_plan(request.prompt)
            out.update(steps=len(graph.steps))

        if settings.halluclean_reasoning_mode == "program_vote" and is_math:
            # Stages 2-4: program execution as external feedback + verified vote.
            # The LLM executor/judge below are intentionally NOT used on this path.
            plan_text = "\n".join(f"{i}. {s.description}" for i, s in enumerate(graph.steps, start=1))
            candidates = se_estimate.responses if se_estimate is not None else [draft_response]
            if settings.l3_extra_samples:
                async with stage("layer3.extra_samples", n=settings.l3_extra_samples) as out:
                    extra = await self._se_triage.sample(
                        request.prompt, request.model,
                        n=settings.l3_extra_samples, seed_base=2000,
                    )
                    candidates = [*candidates, *extra]
                    out.update(ok=len(extra))
            async with stage("layer3.program_vote", candidates=len(candidates)) as out:
                refined, reasoning_traces, judgment = await self._program_verifier.verify(
                    request.prompt, plan_text, candidates, draft_response
                )
                out.update(consistent=judgment.is_consistent, confidence=judgment.confidence)
            layer3_result = Layer3Result(
                execution_graph=graph,
                reasoning_traces=reasoning_traces,
                judgment=judgment,
                refined_response=refined,
                latency_ms=0.0,
            )
        else:
            # Stage 2: Execute
            async with stage("layer3.execute", mode=settings.halluclean_executor_mode) as out:
                graph, reasoning_traces = await self._executor.execute_plan(graph)
                out.update(failed_steps=sum(1 for s in graph.steps if s.status == "failed"))

            # Stages 3 & 4: Judge and Refine
            async with stage("layer3.judge_refine") as out:
                layer3_result = await self._judge.judge_and_refine(graph, reasoning_traces, draft_response)
                out.update(consistent=layer3_result.judgment.is_consistent)
        return self._finish_layer3(
            request, draft_response, draft_result, triage_result, start,
            layer3_start, graph, layer3_result,
        )

    def _finish_layer3(
        self, request, draft_response, draft_result, triage_result, start,
        layer3_start, graph, layer3_result,
    ) -> QueryResponse:
        layer3_result.latency_ms = (time.perf_counter() - layer3_start) * 1000
        total_ms = (time.perf_counter() - start) * 1000

        pipeline_result = PipelineResult(
            query=request.prompt,
            layers_invoked=PipelineLayer.LAYER1_LAYER3,
            triage_result=triage_result,
            layer3_result=layer3_result,
            final_answer=layer3_result.refined_response,
            total_latency_ms=total_ms,
            layer_answer_changed=layer3_result.refined_response != draft_response,
            **self._cost_fields(draft_result),
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
            error_classification=None,
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
                **self._cost_fields(draft_result),
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
