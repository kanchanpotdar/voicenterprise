"""Deterministic metadata extraction with provenance and confidence tracking (P1.2).

Extracts explicit structured signals embedded inside `raw_title` and `raw_content`
without modifying or discarding the original text, and never guesses missing values:
- `extracted_dates`: Sorted unique ISO dates (`YYYY-MM-DD`) in [2020, 2030]
- `latest_date`: Most recent valid ISO date (`None` if no date present)
- `date_provenance`: Where `latest_date` was found (`"title"`, `"header_or_revision"`,
  `"body"`, or `"none"`)
- `date_confidence`: `"high"` (title or explicit date/revision header), `"medium"`
  (body occurrence), or `"none"`
- `slack_channels`: Explicit `#channel-name` references
- `ticket_ids`: Explicit Jira/Linear style keys (e.g. `ENG-142`, `PLAT-89`) and `PR #123`
- `email_addresses`: Explicit `@redwood...` or external email addresses
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

# Matches YYYY-MM-DD where year is 2020..2029, month 01..12, day 01..31
ISO_DATE_RE = re.compile(
    r"\b(202[0-9]-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01]))\b"
)

# Explicit header or revision lines that carry high-confidence document timestamps
HIGH_CONF_DATE_LINE_RE = re.compile(
    r"(?:date|created|updated|revision\s+history|change\s*log|sent|timestamp)\s*[:\-]?"
    r".*?\b(202[0-9]-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01]))\b",
    re.IGNORECASE,
)

REVISION_SECTION_HEADER_RE = re.compile(
    r"^\s*(?:#{1,4}\s*)?(?:revision\s+history|change\s*log|version\s+history)\s*:?\s*$",
    re.IGNORECASE,
)

# Matches #channel-name (lowercase start, contains hyphen or known channel pattern, not markdown header)
SLACK_CHANNEL_RE = re.compile(r"(?:^|[\s(])#([a-z][a-z0-9_\-]{2,45})\b")

# Matches Jira/Linear ticket keys like ENG-1234, PLAT-402, SUP-99
TICKET_KEY_RE = re.compile(r"\b([A-Z]{2,8}-\d{2,6})\b")
TICKET_FALSE_POSITIVES = {
    "UTF", "SHA", "ISO", "SOC", "TLS", "AES", "RSA", "RFC", "CVE", "CPU", "GPU",
    "RAM", "VPC", "NAT", "DNS", "IAM", "OIDC", "SAML", "SCIM", "KMS", "HSM", "API",
    "SDK", "CLI", "SLA", "SLO", "SLI", "QPS", "RPS", "TTFS", "TTFP", "SEV", "USD",
}

PR_REF_RE = re.compile(r"\b(?:PR\s*#(\d{1,6})|pull/(\d{1,6}))\b", re.IGNORECASE)

EMAIL_RE = re.compile(r"\b([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)\b")


@dataclass
class ExtractedMetadata:
    """Extracted metadata with explicit provenance and confidence."""

    record_id: str
    doc_id: str
    extracted_dates: List[str]
    latest_date: Optional[str]
    earliest_date: Optional[str]
    date_provenance: str
    date_confidence: str
    slack_channels: List[str]
    ticket_ids: List[str]
    email_addresses: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def extract_metadata_for_record(
    record_id: str,
    doc_id: str,
    raw_title: str,
    raw_content: str,
) -> ExtractedMetadata:
    """Deterministically extract metadata from a single record's title and content."""
    title_dates = ISO_DATE_RE.findall(raw_title or "")
    header_dates: List[str] = []
    in_revision_block = False
    for line in (raw_content or "").splitlines():
        stripped = line.strip()
        if REVISION_SECTION_HEADER_RE.match(stripped):
            in_revision_block = True
            continue
        if in_revision_block:
            if stripped.startswith(("-", "*")) or re.match(r"^202[0-9]-", stripped):
                header_dates.extend(ISO_DATE_RE.findall(stripped))
                continue
            elif stripped == "":
                continue
            else:
                in_revision_block = False

        m = HIGH_CONF_DATE_LINE_RE.search(line)
        if m:
            header_dates.extend(ISO_DATE_RE.findall(line))

    body_dates = ISO_DATE_RE.findall(raw_content or "")
    all_dates = sorted(set(title_dates + header_dates + body_dates))

    if title_dates:
        latest_date = max(title_dates)
        date_provenance = "title"
        date_confidence = "high"
    elif header_dates:
        latest_date = max(header_dates)
        date_provenance = "header_or_revision"
        date_confidence = "high"
    elif all_dates:
        latest_date = max(all_dates)
        date_provenance = "body"
        date_confidence = "medium"
    else:
        latest_date = None
        date_provenance = "none"
        date_confidence = "none"

    earliest_date = min(all_dates) if all_dates else None

    combined_text = f"{raw_title or ''}\n{raw_content or ''}"

    channels = sorted(
        {
            f"#{ch}"
            for ch in SLACK_CHANNEL_RE.findall(combined_text)
            if not ch.isdigit()
        }
    )

    tickets_set = set()
    for key in TICKET_KEY_RE.findall(combined_text):
        prefix = key.split("-", 1)[0]
        if prefix not in TICKET_FALSE_POSITIVES:
            tickets_set.add(key)

    for pr_a, pr_b in PR_REF_RE.findall(combined_text):
        pr_num = pr_a or pr_b
        if pr_num:
            tickets_set.add(f"PR-{pr_num}")

    emails = sorted(set(EMAIL_RE.findall(combined_text)))

    return ExtractedMetadata(
        record_id=record_id,
        doc_id=doc_id,
        extracted_dates=all_dates,
        latest_date=latest_date,
        earliest_date=earliest_date,
        date_provenance=date_provenance,
        date_confidence=date_confidence,
        slack_channels=channels,
        ticket_ids=sorted(tickets_set),
        email_addresses=emails,
    )


def enrich_dataframe_with_metadata(normalized_df: pd.DataFrame) -> pd.DataFrame:
    """Attach deterministic metadata columns to a normalized documents DataFrame."""
    meta_rows: List[Dict[str, Any]] = []
    for row in normalized_df.to_dict(orient="records"):
        meta = extract_metadata_for_record(
            record_id=row["record_id"],
            doc_id=row["doc_id"],
            raw_title=row["raw_title"],
            raw_content=row["raw_content"],
        )
        d = meta.to_dict()
        d.pop("record_id")
        d.pop("doc_id")
        meta_rows.append({**row, **d})
    return pd.DataFrame(meta_rows)
