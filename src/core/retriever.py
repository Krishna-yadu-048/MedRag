from __future__ import annotations

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


def build_query_engine(index, config: ProjectConfig, settings: AppSettings):
    llm = _build_llm(config, settings)
    query_kwargs: dict[str, Any] = {
        "llm": llm,
        "similarity_top_k": settings.similarity_top_k,
        "response_mode": "compact",
    }
    if settings.relative_score_cutoff > 0:
        query_kwargs["node_postprocessors"] = [
            _relative_score_postprocessor(settings.relative_score_cutoff)
        ]
    if settings.query_mode == "hybrid":
        query_kwargs.update(
            {
                "vector_store_query_mode": "hybrid",
                "sparse_top_k": settings.sparse_top_k,
                "alpha": settings.hybrid_alpha,
            }
        )
    return index.as_query_engine(**query_kwargs)
