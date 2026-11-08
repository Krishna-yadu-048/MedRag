from __future__ import annotations

from typing import Any

from src.core.base import ProjectConfig
from src.core.retriever import build_query_engine
from src.core.schemas import QueryArtifacts, RAGResponse
from src.core.settings import AppSettings



def answer_question(index: Any, question: str, config: ProjectConfig, settings: AppSettings) -> QueryArtifacts:
    query_engine = build_query_engine(index=index, config=config, settings=settings)
    response = query_engine.query(question)
    source_nodes = list(getattr(response, "source_nodes", []) or [])
    sources = _dedupe_sources(source_nodes)
    retrieval_context = _context_snippets(source_nodes)
    evidence = _build_evidence_summary(source_nodes)
    confidence = _infer_confidence(source_nodes)

    return QueryArtifacts(
        response=RAGResponse(
            answer=str(response),
            evidence=evidence,
            sources=sources,
            confidence=confidence,
            disclaimer=config.disclaimer,
        ),
        retrieval_context=retrieval_context,
    )


def _dedupe_sources(source_nodes: list[Any]) -> list[str]:
    seen: list[str] = []
    for source_node in source_nodes:
        metadata = getattr(getattr(source_node, "node", None), "metadata", {}) or {}
        source_org = metadata.get("source_org") or metadata.get("source") or "Unknown source"
        label = f"{source_org}: {_document_name(metadata)}"
        page = metadata.get("page")
        if page is not None:
            label = f"{label} (page {page})"
        if label not in seen:
            seen.append(label)
    return seen


# Placeholder file names shared by many documents; a per-document title is more useful.
_GENERIC_SOURCE_FILES = {"bootstrap_seed"}


def _document_name(metadata: dict[str, Any]) -> str:
    source_file = metadata.get("source_file")
    # PubmedReader stores the paper title under this key, not under "title".
    title = metadata.get("title") or metadata.get("Title of this paper")
    if source_file and source_file not in _GENERIC_SOURCE_FILES:
        return str(source_file)
    return str(title or source_file or "Unknown document")


def _context_snippets(source_nodes: list[Any]) -> list[str]:
    snippets: list[str] = []
    for source_node in source_nodes[:4]:
        text = getattr(getattr(source_node, "node", None), "text", "") or ""
        snippet = " ".join(text.split())
        if snippet:
            snippets.append(snippet[:500])
    return snippets


def _build_evidence_summary(source_nodes: list[Any]) -> str:
    if not source_nodes:
        return "No supporting evidence was retrieved."
    snippets = _context_snippets(source_nodes[:2])
    return " | ".join(snippets) if snippets else "Supporting evidence was retrieved."


def _infer_confidence(source_nodes: list[Any]) -> str:
    count = len(source_nodes)
    if count >= 4:
        return "high"
    if count >= 2:
        return "moderate"
    return "low"
