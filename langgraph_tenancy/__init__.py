"""Tenant isolation for LangGraph persistence.

LangGraph's own threat model: "Without application-level auth, any caller
with a valid thread_id can access that thread's state." This package is that
application-level wall, as a drop-in wrapper:

```python
from langgraph_tenancy import (
    TenantScopedCheckpointer, TenantScopedStore, InMemoryUsageLedger,
)

ledger = InMemoryUsageLedger()
checkpointer = TenantScopedCheckpointer(PostgresSaver(...), usage_ledger=ledger)
store = TenantScopedStore(InMemoryStore())

graph = builder.compile(checkpointer=checkpointer, store=store)

graph.invoke(
    {"messages": [...]},
    config={"configurable": {"thread_id": "t1", "tenant_id": "acme"}},
)

ledger.totals("acme")  # per-tenant token usage, for free
```

Guarantees:
- No tenant_id in config -> the call raises; nothing is read or written.
- Thread ids and store namespaces are physically tenant-prefixed in storage.
- `list(None)` (enumerate all tenants' threads) is refused.
- Maintenance ops (delete/copy/prune) require an explicit `for_tenant()` handle.
"""

from langgraph_tenancy._checkpointer import (
    TenantCheckpointerHandle,
    TenantScopedCheckpointer,
)
from langgraph_tenancy._errors import (
    InvalidTenantError,
    TenancyError,
    TenantRequiredError,
    UnscopedAccessError,
)
from langgraph_tenancy._store import TenantScopedStore
from langgraph_tenancy._usage import (
    InMemoryUsageLedger,
    TenantUsage,
    UsageLedger,
    UsageRecord,
    extract_usage,
)

__all__ = [
    "InMemoryUsageLedger",
    "InvalidTenantError",
    "TenancyError",
    "TenantCheckpointerHandle",
    "TenantRequiredError",
    "TenantScopedCheckpointer",
    "TenantScopedStore",
    "TenantUsage",
    "UnscopedAccessError",
    "UsageLedger",
    "UsageRecord",
    "extract_usage",
]
