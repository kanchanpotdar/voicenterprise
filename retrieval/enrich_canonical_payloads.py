"""Add canonical_v1 payloads in-place; acknowledged writes precede checkpoints."""
import argparse
import json
import time
from pathlib import Path
from qdrant_client import QdrantClient, models
from retrieval.canonical import canonical_metadata


def run(url, collection, checkpoint, batch_size=512):
    client = QdrantClient(url=url, prefer_grpc=True, timeout=300)
    identity = {"url": url, "collection": collection, "schema_version": 1}
    state = json.loads(checkpoint.read_text()) if checkpoint.exists() else {**identity, "offset": None, "processed": 0}
    if any(state.get(k) != v for k, v in identity.items()):
        raise ValueError("Checkpoint belongs to another endpoint/collection/schema")
    if state.get("verified_points") is not None:
        print(json.dumps(state)); return
    initial_count = client.count(collection, exact=True).count
    while not state.get("complete"):
        points, offset = client.scroll(collection, limit=batch_size, offset=state["offset"],
                                       with_payload=True, with_vectors=False)
        operations = [models.SetPayloadOperation(set_payload=models.SetPayload(
            payload={"canonical_v1": canonical_metadata(p.payload)}, points=[p.id])) for p in points]
        if operations:
            for attempt in range(6):
                try:
                    results = client.batch_update_points(collection, operations, wait=True)
                    if any(str(r.status) != "completed" for r in results):
                        raise RuntimeError("Payload write not completed")
                    break
                except Exception:
                    if attempt == 5: raise
                    time.sleep(min(30, 2 ** attempt))
        state.update(offset=offset, processed=state["processed"] + len(points), complete=offset is None)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(checkpoint)
        if state["processed"] % (batch_size * 100) == 0 or state["complete"]:
            print(json.dumps(state), flush=True)
        if state["complete"]: break
    enriched = client.count(collection, count_filter=models.Filter(must=[models.FieldCondition(
        key="canonical_v1.schema_version", match=models.MatchValue(value=1))]), exact=True).count
    final_count = client.count(collection, exact=True).count
    if final_count != initial_count or enriched != final_count:
        raise RuntimeError(f"Verification failed: initial={initial_count}, final={final_count}, enriched={enriched}")
    state.update(verified_points=enriched)
    checkpoint.write_text(json.dumps(state, indent=2) + "\n")
    client.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:6333")
    p.add_argument("--collection", default="enterprise_rag_full")
    p.add_argument("--checkpoint", type=Path, default=Path("data/canonical/payload_enrichment_v1.json"))
    a = p.parse_args()
    run(a.url, a.collection, a.checkpoint)
