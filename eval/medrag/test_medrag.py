from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.shared.metrics import build_rag_metrics
from src.core.service import RAGService

deepeval = pytest.importorskip("deepeval")
deepeval_test_case = pytest.importorskip("deepeval.test_case")
LLMTestCase = deepeval_test_case.LLMTestCase

GOLDEN_DATASET = json.loads(
    (Path(__file__).resolve().parent / "golden_dataset.json").read_text()
)


@pytest.fixture(scope="module")
def medrag_service():
    service = RAGService.from_project_name("medrag")
    if not service.collection_ready():
        pytest.skip("MedRAG collection is not indexed yet.")
    return service


# One test per golden case, so CI reports every case instead of stopping at the first failure.
@pytest.mark.integration
@pytest.mark.parametrize("item", GOLDEN_DATASET, ids=[item["id"] for item in GOLDEN_DATASET])
def test_medrag_golden_case(medrag_service, item):
    result = medrag_service.query(item["query"])
    test_case = LLMTestCase(
        input=item["query"],
        actual_output=result.response.answer,
        expected_output=item["expected_answer"],
        retrieval_context=result.retrieval_context,
    )

    metrics = build_rag_metrics()
    for metric in metrics:
        metric.measure(test_case, _show_indicator=False)

    # Printed with `pytest -s` so a failing CI run shows why, not just the scores.
    print(f"\n=== {item['id']}: {item['query']}")
    print(f"answer: {result.response.answer}")
    print(f"sources ({len(result.retrieval_context)} chunks): {result.response.sources}")
    for metric in metrics:
        print(f"- {metric.name}: {metric.score:.2f} (threshold {metric.threshold}) {metric.reason}")

    failed = [
        f"{metric.name} {metric.score:.2f} < {metric.threshold}"
        for metric in metrics
        if not metric.success
    ]
    assert not failed, f"{item['id']} failed: {', '.join(failed)}"
