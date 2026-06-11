"""Per-tenant token usage, extracted at the checkpoint boundary.

LangGraph already persists `usage_metadata` on every AI message inside
`checkpoint["channel_values"]` — it is just never indexed or aggregated.
Since the tenant-scoped checkpointer sees every checkpoint anyway, it can
pull usage out and attribute it to the tenant for free.

Messages are deduplicated by message id, so re-checkpointing the same
conversation does not double-count.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Protocol


@dataclass(frozen=True)
class UsageRecord:
    message_id: str
    model: str | None
    input_tokens: int
    output_tokens: int
    total_tokens: int


class UsageLedger(Protocol):
    """Anything that can receive per-tenant usage records.

    Swap the in-memory default for one backed by your own Postgres table,
    StatsD, OpenMeter, etc.
    """

    def record(self, tenant_id: str, record: UsageRecord) -> None: ...


@dataclass
class TenantUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    messages: int = 0
    by_model: dict[str, int] = field(default_factory=lambda: defaultdict(int))


class InMemoryUsageLedger:
    """Reference ledger: per-tenant totals, deduped by message id."""

    def __init__(self) -> None:
        self._seen: set[tuple[str, str]] = set()
        self._totals: dict[str, TenantUsage] = defaultdict(TenantUsage)
        self._lock = Lock()

    def record(self, tenant_id: str, record: UsageRecord) -> None:
        with self._lock:
            key = (tenant_id, record.message_id)
            if key in self._seen:
                return
            self._seen.add(key)
            usage = self._totals[tenant_id]
            usage.input_tokens += record.input_tokens
            usage.output_tokens += record.output_tokens
            usage.total_tokens += record.total_tokens
            usage.messages += 1
            if record.model:
                usage.by_model[record.model] += record.total_tokens

    def totals(self, tenant_id: str) -> TenantUsage:
        with self._lock:
            return self._totals[tenant_id]


def extract_usage(checkpoint: dict[str, Any]) -> list[UsageRecord]:
    """Pull usage records out of message objects in a checkpoint's channels."""
    records: list[UsageRecord] = []
    for value in (checkpoint.get("channel_values") or {}).values():
        items = value if isinstance(value, (list, tuple)) else [value]
        for item in items:
            usage = getattr(item, "usage_metadata", None)
            message_id = getattr(item, "id", None)
            if not usage or not message_id:
                continue
            model = (getattr(item, "response_metadata", None) or {}).get("model_name")
            records.append(
                UsageRecord(
                    message_id=message_id,
                    model=model,
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                )
            )
    return records
