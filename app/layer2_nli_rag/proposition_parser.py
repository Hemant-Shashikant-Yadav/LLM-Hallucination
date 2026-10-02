"""
Layer 2: Atomic Proposition Parser.

Extracts independent, verifiable Atomic Propositions (APs) from a draft
LLM response. Each AP is a minimal factual claim that can be independently
verified against external evidence via NLI.
"""

from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger

from app.llm_clients.ollama_client import OllamaClient
from app.models.schemas import AtomicProposition


# System prompt for AP extraction
_AP_EXTRACTION_SYSTEM = """You are a precise fact extraction assistant. Your task is to decompose 
a given text into Atomic Propositions (APs) — minimal, self-contained factual claims.

Rules:
1. Each AP must be a single, independently verifiable statement.
2. Resolve all pronouns and coreferences to their full entity names.
3. Each AP must be understandable without context from the original text.
4. Do NOT include opinions, subjective assessments, or meta-statements.
5. Preserve numerical values, dates, and proper nouns exactly.

Respond with a JSON array of strings, each being one Atomic Proposition.
Example output: ["The Eiffel Tower is located in Paris.", "The Eiffel Tower was completed in 1889."]
"""

_AP_EXTRACTION_PROMPT = """Extract all Atomic Propositions from the following text.

TEXT:
{text}

Return ONLY a JSON array of strings. No explanation or preamble."""


class PropositionParser:
    """
    Extracts Atomic Propositions from LLM-generated text.

    Uses an Ollama model to decompose complex responses into
    individually verifiable factual claims.
    """

    def __init__(self, ollama_client: OllamaClient) -> None:
        self._client = ollama_client
        logger.info("PropositionParser initialized")

    async def extract_propositions(
        self,
        text: str,
        max_propositions: int = 20,
    ) -> list[AtomicProposition]:
        """
        Extract Atomic Propositions from the given text.

        Args:
            text: The draft LLM response to decompose.
            max_propositions: Maximum number of APs to extract.

        Returns:
            List of AtomicProposition objects.
        """
        if not text.strip():
            return []

        try:
            result = await self._client.generate_json(
                prompt=_AP_EXTRACTION_PROMPT.format(text=text),
                system=_AP_EXTRACTION_SYSTEM,
                temperature=0.0,
                max_tokens=2048,
            )

            propositions = self._parse_json_response(result.text, text)

            # Limit to max_propositions
            propositions = propositions[:max_propositions]

            logger.info(
                "Extracted {} atomic propositions from {} chars of text",
                len(propositions),
                len(text),
            )

            return propositions

        except Exception as exc:
            logger.error("AP extraction failed: {}", exc)
            # Fallback: treat each sentence as a proposition
            return self._fallback_sentence_split(text, max_propositions)

    def _parse_json_response(
        self,
        response_text: str,
        source_text: str,
    ) -> list[AtomicProposition]:
        """Parse the JSON array response from the LLM."""
        # Try to extract JSON array from the response
        try:
            # Direct parse
            parsed = json.loads(response_text.strip())
        except json.JSONDecodeError:
            # Try to find JSON array in the response
            match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.warning("Could not parse JSON from LLM response")
                    return self._fallback_sentence_split(source_text)
            else:
                return self._fallback_sentence_split(source_text)

        if not isinstance(parsed, list):
            return self._fallback_sentence_split(source_text)

        propositions = []
        for i, item in enumerate(parsed):
            if isinstance(item, str) and item.strip():
                propositions.append(
                    AtomicProposition(
                        id=i,
                        text=item.strip(),
                        source_span=source_text,
                    )
                )
            elif isinstance(item, dict) and "text" in item:
                propositions.append(
                    AtomicProposition(
                        id=i,
                        text=str(item["text"]).strip(),
                        source_span=source_text,
                    )
                )

        return propositions

    @staticmethod
    def _fallback_sentence_split(
        text: str,
        max_propositions: int = 20,
    ) -> list[AtomicProposition]:
        """
        Fallback: split text into sentences as crude propositions.

        Used when the LLM-based extraction fails.
        """
        # Simple sentence splitting on period, exclamation, question mark
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        propositions = []

        for i, sentence in enumerate(sentences[:max_propositions]):
            sentence = sentence.strip()
            if len(sentence) > 10:  # Skip very short fragments
                propositions.append(
                    AtomicProposition(
                        id=i,
                        text=sentence,
                        source_span=text,
                    )
                )

        logger.warning(
            "Used fallback sentence splitting: {} propositions", len(propositions)
        )
        return propositions
