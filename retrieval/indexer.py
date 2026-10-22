"""Reproducible Qdrant Hybrid Indexer & Retriever (P1.4).

Configures two named vector spaces in Qdrant per chunk:
1. `"dense"`: `BAAI/bge-small-en-v1.5` (384-dim, Cosine similarity) accelerated via
   Apple Silicon Metal (`mps` FP16) with length-sorted dynamic batching (falls back to
   CPU automatically if MPS is unavailable).
2. `"bm25"`: `Qdrant/bm25` sparse BM25 term-frequency + length-normalization vectors
   paired with `models.SparseVectorParams(modifier=models.Modifier.IDF)` in Qdrant so
   that true Okapi BM25 (TF * IDF) scoring is computed across the collection.

Supports three distinct, independently testable retrieval modes on the same index:
- `"bm25"`: Queries only the `"bm25"` sparse vector space (pure lexical BM25).
- `"dense"`: Queries only the `"dense"` vector space (pure semantic BGE-small-en-v1.5).
- `"hybrid"`: Queries both `"dense"` and `"bm25"` via Qdrant `Prefetch` + Reciprocal
  Rank Fusion (`models.Fusion.RRF`).
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from transformers import AutoModel, AutoTokenizer

DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DENSE_DIM = 384
BM25_MODEL_NAME = "Qdrant/bm25"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

RetrievalMode = Literal["bm25", "dense", "hybrid"]


def deterministic_point_uuid(chunk_id: str) -> str:
    """Generate a deterministic UUIDv5 from chunk_id so re-indexing is idempotent."""
    return str(uuid.uuid5(uuid.NAMESPACE_OID, f"enterpriserag_chunk:{chunk_id}"))


def _jaccard_similarity(text1: str, text2: str) -> float:
    """Fast word-level Jaccard similarity for near-duplicate chunk detection."""
    words1 = set(re.findall(r"\b\w{3,}\b", text1.lower()))
    words2 = set(re.findall(r"\b\w{3,}\b", text2.lower()))
    if not words1 or not words2:
        return 0.0
    return len(words1.intersection(words2)) / len(words1.union(words2))


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with provenance and score."""

    chunk_id: str
    record_id: str
    doc_id: str
    score: float
    source_type: str
    normalized_title: str
    section_title: str
    chunk_text: str
    latest_date: Optional[str]
    date_confidence: str
    slack_channels: List[str]
    ticket_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BGEDenseEmbedder:
    """Hardware-accelerated BAAI/bge-small-en-v1.5 embedder using MPS FP16 on Apple Silicon."""

    def __init__(self, model_name: str = DENSE_MODEL_NAME) -> None:
        self.model_name = model_name
        if torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.float16
        elif torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16
        else:
            self.device = "cpu"
            self.dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModel.from_pretrained(model_name, dtype=self.dtype)
            .to(self.device)
            .eval()
        )

    def embed_passages(self, texts: Sequence[str], batch_size: int = 128) -> np.ndarray:
        """Embed passage texts using length-sorted dynamic batching to minimize padding."""
        n = len(texts)
        if n == 0:
            return np.zeros((0, DENSE_DIM), dtype=np.float32)

        order = sorted(range(n), key=lambda i: len(texts[i]))
        sorted_texts = [texts[i] for i in order]
        sorted_vecs = np.zeros((n, DENSE_DIM), dtype=np.float32)

        with torch.no_grad():
            for start in range(0, n, batch_size):
                batch_texts = sorted_texts[start : start + batch_size]
                enc = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                cls_out = self.model(**enc)[0][:, 0]
                normed = torch.nn.functional.normalize(cls_out, p=2, dim=1)
                sorted_vecs[start : start + len(batch_texts)] = (
                    normed.to(torch.float32).cpu().numpy()
                )

        # Restore original order
        out = np.zeros((n, DENSE_DIM), dtype=np.float32)
        for sorted_idx, orig_idx in enumerate(order):
            out[orig_idx] = sorted_vecs[sorted_idx]
        return out

    def embed_query(self, query: str) -> List[float]:
        """Embed a search query using the standard BGE v1.5 query instruction prefix."""
        prefixed = f"{BGE_QUERY_INSTRUCTION}{query}"
        with torch.no_grad():
            enc = self.tokenizer(
                [prefixed],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            cls_out = self.model(**enc)[0][:, 0]
            normed = torch.nn.functional.normalize(cls_out, p=2, dim=1)
            return normed[0].to(torch.float32).cpu().tolist()


class EnterpriseRetriever:
    """Manages reproducible indexing and BM25 / Dense / Hybrid search in Qdrant."""

    def __init__(
        self,
        collection_name: str = "enterprise_rag_dev_10k",
        qdrant_path: Optional[Path] = None,
        dense_model_name: str = DENSE_MODEL_NAME,
        bm25_model_name: str = BM25_MODEL_NAME,
    ) -> None:
        self.collection_name = collection_name
        self.qdrant_path = qdrant_path
        self.dense_model_name = dense_model_name
        self.bm25_model_name = bm25_model_name

        if qdrant_path is not None:
            qdrant_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(qdrant_path))
        else:
            self.client = QdrantClient(":memory:")

        self.dense_embedder = BGEDenseEmbedder(model_name=self.dense_model_name)
        self.bm25_embedder = SparseTextEmbedding(model_name=self.bm25_model_name)

    def create_or_reset_collection(self) -> None:
        """Create (or recreate) the Qdrant collection with explicit Dense + BM25 IDF config."""
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=DENSE_DIM,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )

    def index_chunks_df(
        self,
        chunks_df: pd.DataFrame,
        batch_size: int = 1024,
        reset_collection: bool = True,
    ) -> Dict[str, Any]:
        """Deterministically embed and index a chunks DataFrame into Qdrant."""
        t0 = time.perf_counter()
        if reset_collection or not self.client.collection_exists(self.collection_name):
            self.create_or_reset_collection()

        # Sort by char_length before batching so MPS GPU batches have minimal padding waste
        sorted_df = chunks_df.sort_values("char_length").reset_index(drop=True)
        records = sorted_df.to_dict(orient="records")
        total = len(records)

        for start in range(0, total, batch_size):
            batch = records[start : start + batch_size]
            texts = [str(r["contextualized_text"]) for r in batch]

            dense_vecs = self.dense_embedder.embed_passages(texts, batch_size=128)
            bm25_vecs = list(self.bm25_embedder.embed(texts, batch_size=256))

            points: List[models.PointStruct] = []
            for row, d_vec, s_vec in zip(batch, dense_vecs, bm25_vecs):
                point_id = deterministic_point_uuid(str(row["chunk_id"]))
                payload = {
                    "chunk_id": str(row["chunk_id"]),
                    "record_id": str(row["record_id"]),
                    "doc_id": str(row["doc_id"]),
                    "chunk_index": int(row["chunk_index"]),
                    "total_chunks_in_doc": int(row["total_chunks_in_doc"]),
                    "source_type": str(row["source_type"]),
                    "normalized_title": str(row["normalized_title"]),
                    "section_title": str(row["section_title"]),
                    "chunk_text": str(row["chunk_text"]),
                    "latest_date": (
                        None
                        if pd.isna(row.get("latest_date"))
                        else str(row.get("latest_date"))
                    ),
                    "date_confidence": str(row.get("date_confidence") or "none"),
                    "slack_channels": (
                        []
                        if row.get("slack_channels") is None
                        else [str(x) for x in row["slack_channels"]]
                    ),
                    "ticket_ids": (
                        []
                        if row.get("ticket_ids") is None
                        else [str(x) for x in row["ticket_ids"]]
                    ),
                    "has_duplicate_doc_id_conflict": bool(
                        row.get("has_duplicate_doc_id_conflict", False)
                    ),
                    "inclusion_reason": str(row.get("inclusion_reason") or "full_corpus"),
                }
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector={
                            "dense": d_vec.tolist(),
                            "bm25": models.SparseVector(
                                indices=s_vec.indices.tolist(),
                                values=s_vec.values.tolist(),
                            ),
                        },
                        payload=payload,
                    )
                )

            self.client.upsert(collection_name=self.collection_name, points=points)
            done = start + len(batch)
            elapsed_so_far = time.perf_counter() - t0
            rate = done / max(0.001, elapsed_so_far)
            print(
                f"  Indexed {done}/{total} chunks ({100.0*done/total:.1f}%) "
                f"[{rate:.1f} chunks/sec, elapsed {elapsed_so_far:.1f}s]",
                flush=True,
            )

        elapsed = time.perf_counter() - t0
        info = self.client.get_collection(self.collection_name)
        return {
            "collection_name": self.collection_name,
            "dense_model": self.dense_model_name,
            "dense_device": self.dense_embedder.device,
            "dense_dtype": str(self.dense_embedder.dtype),
            "dense_dim": DENSE_DIM,
            "bm25_model": self.bm25_model_name,
            "bm25_modifier": "IDF",
            "chunks_indexed": total,
            "qdrant_points_count": info.points_count,
            "unique_parent_docs_indexed": int(chunks_df["doc_id"].nunique()) if total > 0 else 0,
            "indexing_seconds": round(elapsed, 2),
        }

    def _build_filter(
        self,
        source_types: Optional[Sequence[str]] = None,
    ) -> Optional[models.Filter]:
        if not source_types:
            return None
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="source_type",
                    match=models.MatchAny(any=[s.lower() for s in source_types]),
                )
            ]
        )

    def retrieve_chunks(
        self,
        query: str,
        mode: RetrievalMode = "hybrid",
        top_k_chunks: int = 40,
        source_types: Optional[Sequence[str]] = None,
    ) -> List[RetrievedChunk]:
        """Retrieve top_k_chunks using 'bm25', 'dense', or 'hybrid' (RRF) mode."""
        q_filter = self._build_filter(source_types=source_types)

        if mode == "dense":
            q_dense = self.dense_embedder.embed_query(query)
            resp = self.client.query_points(
                collection_name=self.collection_name,
                query=q_dense,
                using="dense",
                query_filter=q_filter,
                limit=top_k_chunks,
                with_payload=True,
            )
        elif mode == "bm25":
            q_bm25 = next(self.bm25_embedder.query_embed(query))
            sparse_q = models.SparseVector(
                indices=q_bm25.indices.tolist(),
                values=q_bm25.values.tolist(),
            )
            resp = self.client.query_points(
                collection_name=self.collection_name,
                query=sparse_q,
                using="bm25",
                query_filter=q_filter,
                limit=top_k_chunks,
                with_payload=True,
            )
        elif mode == "hybrid":
            q_dense = self.dense_embedder.embed_query(query)
            q_bm25 = next(self.bm25_embedder.query_embed(query))
            sparse_q = models.SparseVector(
                indices=q_bm25.indices.tolist(),
                values=q_bm25.values.tolist(),
            )
            prefetch_limit = max(top_k_chunks * 2, 50)
            resp = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=q_dense,
                        using="dense",
                        filter=q_filter,
                        limit=prefetch_limit,
                    ),
                    models.Prefetch(
                        query=sparse_q,
                        using="bm25",
                        filter=q_filter,
                        limit=prefetch_limit,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k_chunks,
                with_payload=True,
            )
        else:
            raise ValueError(f"Unsupported retrieval mode: {mode}")

        out: List[RetrievedChunk] = []
        for pt in resp.points:
            pl = pt.payload or {}
            out.append(
                RetrievedChunk(
                    chunk_id=str(pl.get("chunk_id", "")),
                    record_id=str(pl.get("record_id", "")),
                    doc_id=str(pl.get("doc_id", "")),
                    score=float(pt.score or 0.0),
                    source_type=str(pl.get("source_type", "")),
                    normalized_title=str(pl.get("normalized_title", "")),
                    section_title=str(pl.get("section_title", "")),
                    chunk_text=str(pl.get("chunk_text", "")),
                    latest_date=pl.get("latest_date"),
                    date_confidence=str(pl.get("date_confidence", "none")),
                    slack_channels=list(pl.get("slack_channels") or []),
                    ticket_ids=list(pl.get("ticket_ids") or []),
                )
            )
        return out

    def retrieve_documents(
        self,
        query: str,
        mode: RetrievalMode = "hybrid",
        top_k_docs: int = 20,
        chunk_oversample_factor: int = 4,
        source_types: Optional[Sequence[str]] = None,
        diversity_filter: bool = False,
        max_chunks_per_doc: int = 3,
        near_duplicate_threshold: float = 0.70,
    ) -> Tuple[List[str], List[RetrievedChunk]]:
        """Retrieve unique parent doc_ids ordered by best chunk score, plus top chunks.

        When diversity_filter=True (Experiment B), caps sibling chunks per document in
        the candidate stream and suppresses near-duplicate chunks (Jaccard >= 0.70)
        to prevent redundant passages from crowding out distinct parent documents.
        """
        chunks = self.retrieve_chunks(
            query=query,
            mode=mode,
            top_k_chunks=top_k_docs * chunk_oversample_factor,
            source_types=source_types,
        )

        if diversity_filter:
            filtered_chunks: List[RetrievedChunk] = []
            doc_counts: Dict[str, int] = {}
            for ch in chunks:
                cnt = doc_counts.get(ch.doc_id, 0)
                if cnt >= max_chunks_per_doc:
                    continue
                is_dup = False
                for ex in [c for c in filtered_chunks if c.doc_id == ch.doc_id]:
                    if _jaccard_similarity(ch.chunk_text, ex.chunk_text) >= near_duplicate_threshold:
                        is_dup = True
                        break
                if not is_dup:
                    filtered_chunks.append(ch)
                    doc_counts[ch.doc_id] = cnt + 1
            chunks = filtered_chunks

        seen_docs = set()
        ordered_doc_ids: List[str] = []
        for ch in chunks:
            if ch.doc_id not in seen_docs:
                seen_docs.add(ch.doc_id)
                ordered_doc_ids.append(ch.doc_id)
                if len(ordered_doc_ids) >= top_k_docs:
                    break
        return ordered_doc_ids, chunks
