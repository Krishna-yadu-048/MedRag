from pathlib import Path
from types import SimpleNamespace

from src.projects.medrag.ingestor import MedRAGIngestor


class DummyDocument:
    def __init__(self, metadata):
        self.metadata = metadata


def test_enrich_metadata_maps_pubmed_documents():
    ingestor = MedRAGIngestor(config=None)  # type: ignore[arg-type]
    doc = DummyDocument({"source": "pubmed", "query": "hypertension management guideline"})

    enriched = ingestor.enrich_metadata([doc])[0]

    assert enriched.metadata["source_org"] == "PubMed"
    assert enriched.metadata["evidence_type"] == "research_abstract"
    assert enriched.metadata["specialty"] == "cardiology"


def test_enrich_metadata_maps_fda_labels():
    ingestor = MedRAGIngestor(config=None)  # type: ignore[arg-type]
    doc = DummyDocument({"source_file": "FDA_metformin_label.pdf"})

    enriched = ingestor.enrich_metadata([doc])[0]

    assert enriched.metadata["source_org"] == "FDA"
    assert enriched.metadata["evidence_type"] == "drug_label"
    assert enriched.metadata["specialty"] == "endocrinology"


def test_load_guideline_pdfs_respects_zero_limit(monkeypatch, tmp_path: Path):
    guideline_dir = tmp_path / "guidelines"
    guideline_dir.mkdir()
    (guideline_dir / "WHO_BP.pdf").write_bytes(b"fake pdf")
    monkeypatch.setenv("MAX_GUIDELINE_FILES", "0")

    ingestor = MedRAGIngestor(config=SimpleNamespace(data_dir=tmp_path))

    docs = ingestor._load_guideline_pdfs()

    assert docs == []


def test_load_and_parse_includes_bootstrap_documents(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PUBMED_ENABLED", "false")
    monkeypatch.setenv("MEDRAG_INCLUDE_BOOTSTRAP", "true")

    ingestor = MedRAGIngestor(config=SimpleNamespace(data_dir=tmp_path))

    docs = ingestor.load_and_parse()

    titles = {doc.metadata["title"] for doc in docs}
    assert "Type 2 diabetes first-line therapy overview" in titles
    assert "Why medical guidance should include a disclaimer" in titles


def _page(number, markdown=None, error=None):
    if error:
        return SimpleNamespace(page_number=number, success=False, error=error)
    return SimpleNamespace(page_number=number, success=True, markdown=markdown)


class FakeLlamaCloud:
    pages = []
    calls = []

    def __init__(self, api_key):
        self.parsing = self

    def parse(self, **kwargs):
        FakeLlamaCloud.calls.append(kwargs)
        return SimpleNamespace(markdown=SimpleNamespace(pages=FakeLlamaCloud.pages))


def _pdf_setup(monkeypatch, tmp_path: Path, pages):
    import llama_cloud

    guideline_dir = tmp_path / "guidelines"
    guideline_dir.mkdir()
    (guideline_dir / "WHO_BP.pdf").write_bytes(b"%PDF fake")
    monkeypatch.setenv("LLAMA_CLOUD_API_KEY", "llx-test")
    monkeypatch.setenv("MAX_GUIDELINE_FILES", "3")
    FakeLlamaCloud.pages = pages
    FakeLlamaCloud.calls = []
    monkeypatch.setattr(llama_cloud, "LlamaCloud", FakeLlamaCloud)
    return MedRAGIngestor(config=SimpleNamespace(data_dir=tmp_path))


def test_load_guideline_pdfs_uses_v2_api_and_keeps_page_numbers(monkeypatch, tmp_path: Path):
    ingestor = _pdf_setup(
        monkeypatch,
        tmp_path,
        [_page(1, "# Hypertension"), _page(2, error="timeout"), _page(3, "Targets <140/90")],
    )

    docs = ingestor._load_guideline_pdfs()

    assert [(d.metadata["page"], d.text) for d in docs] == [
        (1, "# Hypertension"),
        (3, "Targets <140/90"),
    ]
    assert docs[0].metadata["source_file"] == "WHO_BP.pdf"
    call = FakeLlamaCloud.calls[0]
    assert call["expand"] == ["markdown"]
    assert call["upload_file"][0] == "WHO_BP.pdf"


def test_load_guideline_pdfs_fails_loudly_when_nothing_parses(monkeypatch, tmp_path: Path):
    import pytest

    ingestor = _pdf_setup(monkeypatch, tmp_path, [_page(1, error="unreadable")])

    with pytest.raises(RuntimeError, match="no text for 'WHO_BP.pdf'"):
        ingestor._load_guideline_pdfs()
