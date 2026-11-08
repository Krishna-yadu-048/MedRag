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
