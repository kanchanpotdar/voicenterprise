from retrieval.canonical import deduplicate_chunks, evidence_hash
from retrieval.indexer import RetrievedChunk
from retrieval.indexer import EnterpriseRetriever


def chunk(doc, text, score=1):
    return RetrievedChunk(doc, doc, doc, score, "jira", "Title", "Section", text,
                          None, "none", [], [])


def test_duplicates_retain_aliases_and_ranking_without_mutation():
    source = [chunk("b", "Limit 10 MB", 9), chunk("a", "Limit 10 MB", 8)]
    result = deduplicate_chunks(source)
    assert len(result) == 1
    assert result[0].doc_id == "b" and result[0].score == 9
    assert result[0].alias_doc_ids == ["a", "b"]
    assert source[0].content_hash is None


def test_conflicts_case_indentation_and_empty_evidence_are_preserved():
    texts = ["Limit 10 MB", "Limit 20 MB", "limit 10 MB", "  Limit 10 MB", "", ""]
    assert len(deduplicate_chunks([chunk(str(i), t) for i, t in enumerate(texts)])) == 6


def test_hash_fields_are_unambiguous_and_unicode_normalized():
    assert evidence_hash("a|b", "c", "d") != evidence_hash("a", "b|c", "d")
    assert evidence_hash("é", "", "x\r\ny") == evidence_hash("e\u0301", "", "x\ny")


def test_production_retrieval_switch_changes_candidates_and_preserves_aliases():
    retriever = EnterpriseRetriever.__new__(EnterpriseRetriever)
    source = [chunk("a", "same", 9), chunk("b", "same", 8), chunk("c", "distinct", 7)]
    retriever.retrieve_chunks = lambda **kwargs: source
    raw_ids, raw = retriever.retrieve_documents("query", top_k_docs=2)
    filtered_ids, filtered = retriever.retrieve_documents("query", top_k_docs=2, canonical_filter=True)
    assert raw_ids == ["a", "b"]
    assert filtered_ids == ["a", "c"]
    assert len(raw) == 3 and len(filtered) == 2
    assert filtered[0].alias_doc_ids == ["a", "b"]
