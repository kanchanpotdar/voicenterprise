"""Add canonical_v1 payloads in-place; acknowledged writes precede checkpoints."""
import argparse
import json
import time
import subprocess
from pathlib import Path
from qdrant_client import QdrantClient, models
from retrieval.canonical import canonical_metadata


def memory_percent(container):
    """Read Docker's cgroup memory usage without a shell; fail closed on errors."""
    result = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.MemPerc}}", container],
                            capture_output=True, text=True, check=True, timeout=15)
    return float(result.stdout.strip().rstrip("%"))


def run(url, collection, checkpoint, batch_size=64, pause_seconds=1.0, max_batches=None,
        docker_container=None, max_memory_percent=78.0):
    if batch_size < 1 or pause_seconds < 0 or (max_batches is not None and max_batches < 1):
        raise ValueError("Invalid batch size, pause, or batch limit")
    client = QdrantClient(url=url, prefer_grpc=True, timeout=300)
    identity = {"url": url, "collection": collection, "schema_version": 1}
    state = json.loads(checkpoint.read_text()) if checkpoint.exists() else {**identity, "offset": None, "processed": 0}
    if any(state.get(k) != v for k, v in identity.items()):
        raise ValueError("Checkpoint belongs to another endpoint/collection/schema")
    if state.get("verified_points") is not None:
        print(json.dumps(state)); return
    initial_count = client.count(collection, exact=True).count
    batches = 0
    while not state.get("complete"):
        if docker_container and memory_percent(docker_container) >= max_memory_percent:
            print("Paused safely: Docker memory threshold reached; checkpoint preserved", flush=True)
            client.close()
            return
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
        batches += 1
        if max_batches is not None and batches >= max_batches:
            client.close()
            return
        time.sleep(pause_seconds)
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
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--pause-seconds", type=float, default=1.0)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--docker-container", default=None)
    p.add_argument("--max-memory-percent", type=float, default=78.0)
    a = p.parse_args()
    run(a.url, a.collection, a.checkpoint, a.batch_size, a.pause_seconds, a.max_batches,
        a.docker_container, a.max_memory_percent)
