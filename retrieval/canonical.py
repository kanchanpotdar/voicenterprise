"""Conservative canonical grouping of retrieved evidence, without vector changes."""
from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import replace


def evidence_hash(title: str, section: str, text: str) -> str:
    """Hash unambiguous fields; preserve case, punctuation and paragraph structure.

    Only canonical Unicode and line endings are normalized. Whitespace inside
    code blocks can be meaningful, so arbitrary whitespace is not collapsed.
    """
    fields = [unicodedata.normalize("NFC", value.replace("\r\n", "\n"))
              for value in (title, section, text)]
    return hashlib.sha256(json.dumps(fields, ensure_ascii=False).encode()).hexdigest()


def deduplicate_chunks(chunks):
    """Keep highest-ranked exact evidence once and attach every observed alias.

    Scores/order come from retrieval. Similar text and conflicting versions are
    never merged. Empty evidence is never collapsed. Inputs are not mutated.
    """
    output, positions = [], {}
    for chunk in chunks:
        digest = evidence_hash(chunk.normalized_title, chunk.section_title, chunk.chunk_text)
        aliases = sorted(set([chunk.doc_id] + list(chunk.alias_doc_ids)))
        if not chunk.chunk_text.strip() or digest not in positions:
            if chunk.chunk_text.strip():
                positions[digest] = len(output)
            output.append(replace(chunk, content_hash=digest, alias_doc_ids=aliases))
        else:
            index = positions[digest]
            previous = output[index]
            output[index] = replace(previous, alias_doc_ids=sorted(set(previous.alias_doc_ids + aliases)))
    return output
