from __future__ import annotations

import logging
import os
from pathlib import Path

from llama_index.core.schema import Document

from src.core.base import DocumentIngestor

logger = logging.getLogger(__name__)


class MedRAGIngestor(DocumentIngestor):
    bootstrap_documents = (
        {
            "title": "Type 2 diabetes first-line therapy overview",
            "text": (
                "Type 2 diabetes management usually begins with lifestyle modification, "
                "glycemic monitoring, and metformin when there are no contraindications. "
                "Guidelines commonly describe individualized escalation to GLP-1 receptor "
                "agonists, SGLT2 inhibitors, or insulin based on comorbidities, kidney "
                "function, cardiovascular risk, and glycemic control."
            ),
            "specialty": "endocrinology",
        },
        {
            "title": "Type 1 diabetes treatment overview",
            "text": (
                "Type 1 diabetes requires insulin replacement therapy rather than oral "
                "first-line agents. Guideline-based care typically includes basal-bolus "
                "or pump-based insulin delivery, glucose monitoring, hypoglycemia "
                "education, and individualized nutrition planning."
            ),
            "specialty": "endocrinology",
        },
        {
            "title": "Hypertension guideline overview",
            "text": (
                "Hypertension guidelines generally cover blood pressure thresholds, "
                "target ranges, home blood pressure monitoring, lifestyle changes, and "
                "medication classes such as ACE inhibitors, ARBs, calcium-channel "
                "blockers, and thiazide-type diuretics. Follow-up cadence depends on "
                "severity, symptoms, and treatment response."
            ),
            "specialty": "cardiology",
        },
        {
            "title": "Why medical guidance should include a disclaimer",
            "text": (
                "Medical guidance in retrieval-augmented systems should include a disclaimer "
                "that the information is for educational purposes only and is not a substitute "
                "for clinician judgment, diagnosis, or personalized treatment. The disclaimer "
                "helps users understand the limits of the retrieved material and encourages "
                "consultation with qualified healthcare professionals."
            ),
            "specialty": "general_medicine",
        },
    )
    pubmed_queries = (
        "type 2 diabetes treatment guideline",
        "hypertension management guideline",
    )
    pubmed_max_results = 5

    def load_and_parse(self):
        documents = []
        documents.extend(self._load_guideline_pdfs())
        documents.extend(self._load_pubmed_abstracts())
        if os.getenv("MEDRAG_INCLUDE_BOOTSTRAP", "true").lower() in {"1", "true", "yes"}:
            documents.extend(self._load_bootstrap_documents())
        return documents

    def enrich_metadata(self, docs):
        for doc in docs:
            metadata = dict(getattr(doc, "metadata", {}) or {})
            source_file = str(metadata.get("source_file", "")).lower()
            text_hint = " ".join(
                [
                    source_file,
                    str(metadata.get("title", "")),
                    str(metadata.get("query", "")),
                ]
            ).lower()

            if metadata.get("source") == "pubmed":
                metadata["source_org"] = "PubMed"
                metadata["evidence_type"] = "research_abstract"
            elif any(token in source_file for token in ("fda", "dailymed", "label")):
                metadata["source_org"] = "FDA"
                metadata["evidence_type"] = "drug_label"
            else:
                metadata["source_org"] = metadata.get("source_org", "WHO")
                metadata["evidence_type"] = metadata.get("evidence_type", "guideline")

            metadata["specialty"] = self._infer_specialty(text_hint)
            doc.metadata = metadata
        return docs

    def _load_guideline_pdfs(self):
        guideline_dir = Path(self.config.data_dir) / "guidelines"
        pdf_paths = sorted(guideline_dir.glob("*.pdf"))
        max_guideline_files = int(os.getenv("MAX_GUIDELINE_FILES", "3"))
        if max_guideline_files >= 0:
            pdf_paths = pdf_paths[:max_guideline_files]
        if not pdf_paths:
            return []

        api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LLAMA_CLOUD_API_KEY is required to parse guideline PDFs with LlamaParse."
            )

        from llama_cloud import LlamaCloud

        client = LlamaCloud(api_key=api_key)
        tier = os.getenv("LLAMA_PARSE_TIER", "cost_effective")

        documents = []
        for pdf_path in pdf_paths:
            pages = self._parse_pdf_pages(client, pdf_path, tier)
            if not pages:
                raise RuntimeError(f"LlamaParse returned no text for '{pdf_path.name}'.")
            for page_number, markdown in pages:
                documents.append(
                    Document(
                        text=markdown,
                        metadata={
                            "source_file": pdf_path.name,
                            "page": page_number,
                            "parser": "llamaparse_v2",
                            "source": "guideline_pdf",
                        },
                    )
                )
        return documents

    @staticmethod
    def _parse_pdf_pages(client, pdf_path: Path, tier: str) -> list[tuple[int, str]]:
        """Parse one PDF with the LlamaParse v2 API and return (page_number, markdown) pairs.

        The v1 API used by the old `llama-parse` package is unavailable to newer accounts,
        and that package swallowed the error, so PDFs silently dropped out of the index.
        The v2 client raises on a failed job instead.
        """
        options = {}
        if tier.startswith("agentic"):
            options["agentic_options"] = {
                "custom_prompt": (
                    "Clinical guideline or label with evidence tables, dose recommendations, "
                    "warnings, and section hierarchy. Preserve tables and headings."
                )
            }
        with pdf_path.open("rb") as handle:
            result = client.parsing.parse(
                upload_file=(pdf_path.name, handle, "application/pdf"),
                tier=tier,
                version="latest",
                expand=["markdown"],
                **options,
            )

        pages: list[tuple[int, str]] = []
        failed: list[int] = []
        for page in getattr(result.markdown, "pages", None) or []:
            if not getattr(page, "success", False):
                failed.append(page.page_number)
                continue
            if page.markdown.strip():
                pages.append((page.page_number, page.markdown))
        if failed:
            logger.warning("LlamaParse failed on pages %s of '%s'.", failed, pdf_path.name)
        return pages


    def _load_pubmed_abstracts(self):
        if os.getenv("PUBMED_ENABLED", "true").lower() not in {"1", "true", "yes"}:
            return []

        from llama_index.readers.papers import PubmedReader

        reader = PubmedReader()
        documents = []
        max_queries = int(os.getenv("PUBMED_QUERY_LIMIT", "1"))
        queries = self.pubmed_queries[:max_queries] if max_queries > 0 else ()
        max_results = int(os.getenv("PUBMED_MAX_RESULTS", str(self.pubmed_max_results)))

        for query in queries:
            abstracts = reader.load_data(search_query=query, max_results=max_results)
            for doc in abstracts:
                metadata = dict(getattr(doc, "metadata", {}) or {})
                metadata.update({"source": "pubmed", "query": query, "parser": "pubmed_reader"})
                doc.metadata = metadata
                documents.append(doc)
        return documents

    

    def _load_bootstrap_documents(self):
        documents = []
        for item in self.bootstrap_documents:
            documents.append(
                Document(
                    text=item["text"],
                    metadata={
                        "source": "bootstrap",
                        "source_file": "bootstrap_seed",
                        "title": item["title"],
                        "parser": "bootstrap_seed",
                        "source_org": "Bootstrap",
                        "evidence_type": "guideline_summary",
                        "specialty": item["specialty"],
                    },
                )
            )
        return documents

    @staticmethod
    def _infer_specialty(text_hint: str) -> str:
        mapping = {
            "diabetes": "endocrinology",
            "glucose": "endocrinology",
            "metformin": "endocrinology",
            "insulin": "endocrinology",
            "hypertension": "cardiology",
            "blood pressure": "cardiology",
            "warfarin": "cardiology",
            "lisinopril": "cardiology",
            "asthma": "pulmonology",
            "copd": "pulmonology",
            "kidney": "nephrology",
        }
        for keyword, specialty in mapping.items():
            if keyword in text_hint:
                return specialty
        return "general_medicine"