from __future__ import annotations

from types import SimpleNamespace

from src.core.generator import _dedupe_sources


def _node(**metadata):
    return SimpleNamespace(node=SimpleNamespace(metadata=metadata, text=""))


def test_pubmed_sources_use_paper_title():
    sources = _dedupe_sources(
        [_node(source_org="PubMed", source="pubmed", **{"Title of this paper": "Metformin trial"})]
    )

    assert sources == ["PubMed: Metformin trial"]


def test_bootstrap_sources_are_listed_per_document():
    sources = _dedupe_sources(
        [
            _node(source_org="Bootstrap", source_file="bootstrap_seed", title="Diabetes overview"),
            _node(source_org="Bootstrap", source_file="bootstrap_seed", title="Hypertension overview"),
        ]
    )

    assert sources == ["Bootstrap: Diabetes overview", "Bootstrap: Hypertension overview"]


def test_pdf_sources_keep_file_name_and_page():
    sources = _dedupe_sources([_node(source_org="WHO", source_file="WHO_BP.pdf", page=3)])

    assert sources == ["WHO: WHO_BP.pdf (page 3)"]


def _scored(score, text="chunk"):
    return SimpleNamespace(score=score, node=SimpleNamespace(metadata={}, text=text))


def test_relative_cutoff_drops_weak_matches():
    from src.core.retriever import filter_by_relative_score

    nodes = [_scored(0.88), _scored(0.85), _scored(0.70), _scored(0.62)]

    kept = filter_by_relative_score(nodes, 0.9)

    assert [n.score for n in kept] == [0.88, 0.85]


def test_relative_cutoff_always_keeps_best_and_can_be_disabled():
    from src.core.retriever import filter_by_relative_score

    nodes = [_scored(0.5), _scored(0.1)]

    assert [n.score for n in filter_by_relative_score(nodes, 0.99)] == [0.5]
    assert filter_by_relative_score(nodes, 0) == nodes


def test_confidence_uses_best_score_not_chunk_count():
    from src.core.generator import _infer_confidence

    assert _infer_confidence([_scored(0.86)]) == "high"
    assert _infer_confidence([_scored(0.5)] * 8) == "low"
    assert _infer_confidence([]) == "low"


def test_retrieval_context_is_full_text_of_every_chunk():
    from src.core.generator import _retrieval_context

    long_text = "x" * 2000
    context = _retrieval_context([_scored(0.9, long_text)] * 6)

    assert len(context) == 6
    assert context[0] == long_text


def test_rerank_keeps_best_n_by_cross_encoder_and_normalizes_scores():
    from src.core.retriever import rerank_nodes

    nodes = [_scored(0.9, "target ranges"), _scored(0.8, "first-line drug classes"), _scored(0.7, "lab tests")]
    logits = {"target ranges": -2.0, "first-line drug classes": 6.0, "lab tests": 1.0}

    kept = rerank_nodes(nodes, "first-line drugs?", lambda q, texts: [logits[t] for t in texts], 2)

    assert [n.node.text for n in kept] == ["first-line drug classes", "lab tests"]
    assert all(0.0 < n.score < 1.0 for n in kept)
    assert kept[0].score > 0.99
