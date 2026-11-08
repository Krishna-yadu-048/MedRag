from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

from src.core.base import ProjectConfig
from src.core.settings import AppSettings


def _build_llm(config: ProjectConfig, settings: AppSettings):
    from llama_index.llms.openai import OpenAI

    try:
        return OpenAI(
            model=settings.openai_model,
            temperature=0.1,
            system_prompt=config.system_prompt,
        )
    except TypeError:
        return OpenAI(model=settings.openai_model, temperature=0.1)


def filter_by_relative_score(nodes: list[Any], ratio: float) -> list[Any]:
    """Keep nodes scoring at least `ratio` x the best node's score.

    A fixed top_k always returns k chunks, so on a small corpus most of them are off-topic
    (diabetes chunks for a hypertension question). A cutoff relative to the best match adapts
    per question without hand-tuning an absolute similarity threshold for the embedding model.
    The best node is always kept.
    """
    if not nodes or ratio <= 0:
        return nodes
    scores = [node.score for node in nodes if getattr(node, "score", None) is not None]
    if not scores:
        return nodes
    best = max(scores)
    if best <= 0:
        return nodes
    floor = best * ratio
    return [node for node in nodes if node.score is None or node.score >= floor]


def _relative_score_postprocessor(ratio: float):
    from llama_index.core.postprocessor.types import BaseNodePostprocessor

    class RelativeScoreCutoff(BaseNodePostprocessor):
        @classmethod
        def class_name(cls) -> str:
            return "RelativeScoreCutoff"

        def _postprocess_nodes(self, nodes, query_bundle=None):
            return filter_by_relative_score(nodes, ratio)

    return RelativeScoreCutoff()


def rerank_nodes(nodes: list[Any], query: str, scorer, top_n: int) -> list[Any]:
    """Re-score nodes against the query with a cross-encoder and keep the best `top_n`.

    Embedding similarity finds the right passage but ranks neighbouring passages almost as
    high, so the LLM and evals get several mostly off-topic chunks. A cross-encoder reads the
    query and chunk together and separates them far more sharply.
    """
    if not nodes or top_n <= 0:
        return nodes
    texts = [getattr(getattr(node, "node", None), "text", "") or "" for node in nodes]
    scores = list(scorer(query, texts))
    for node, score in zip(nodes, scores):
        # MS MARCO cross-encoders emit logits; the sigmoid gives a 0-1 relevance score that
        # downstream code (confidence labels) can interpret.
        node.score = 1.0 / (1.0 + math.exp(-float(score)))
    ranked = sorted(nodes, key=lambda node: node.score, reverse=True)
    return ranked[:top_n]


@lru_cache(maxsize=2)
def _cross_encoder(model_name: str):
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=model_name)


def warm_reranker(settings: AppSettings) -> None:
    """Load the reranker once at startup so the first query does not pay for the download."""
    if settings.rerank_model and settings.rerank_top_n > 0:
        list(_cross_encoder(settings.rerank_model).rerank("warmup", ["warmup"]))


def _rerank_postprocessor(model_name: str, top_n: int):
    from llama_index.core.postprocessor.types import BaseNodePostprocessor

    def scorer(query: str, texts: list[str]):
        return _cross_encoder(model_name).rerank(query, texts)

    class CrossEncoderRerank(BaseNodePostprocessor):
        @classmethod
        def class_name(cls) -> str:
            return "CrossEncoderRerank"

        def _postprocess_nodes(self, nodes, query_bundle=None):
            query = getattr(query_bundle, "query_str", "") or ""
            return rerank_nodes(nodes, query, scorer, top_n)

    return CrossEncoderRerank()


def build_query_engine(index, config: ProjectConfig, settings: AppSettings):
    llm = _build_llm(config, settings)
    query_kwargs: dict[str, Any] = {
        "llm": llm,
        "similarity_top_k": settings.similarity_top_k,
        "response_mode": "compact",
    }
    postprocessors = []
    if settings.rerank_model and settings.rerank_top_n > 0:
        postprocessors.append(_rerank_postprocessor(settings.rerank_model, settings.rerank_top_n))
    elif settings.relative_score_cutoff > 0:
        # Cross-encoder scores are unbounded logits, so the relative cutoff only applies
        # to plain similarity scores, when the reranker is turned off.
        postprocessors.append(_relative_score_postprocessor(settings.relative_score_cutoff))
    if postprocessors:
        query_kwargs["node_postprocessors"] = postprocessors
    if settings.query_mode == "hybrid":
        query_kwargs.update(
            {
                "vector_store_query_mode": "hybrid",
                "sparse_top_k": settings.sparse_top_k,
                "alpha": settings.hybrid_alpha,
            }
        )
    return index.as_query_engine(**query_kwargs)
