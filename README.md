# langgraph-tenancy

[![CI](https://github.com/ac12644/langgraph-tenancy/actions/workflows/ci.yml/badge.svg)](https://github.com/ac12644/langgraph-tenancy/actions/workflows/ci.yml)
[![PyPI - Version](https://img.shields.io/pypi/v/langgraph-tenancy.svg)](https://pypi.org/project/langgraph-tenancy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Tenant isolation for LangGraph persistence — as a drop-in wrapper.**

> Using LangGraph.js? Same package, same guarantees:
> [ac12644/langgraph-tenancy-js](https://github.com/ac12644/langgraph-tenancy-js) ·
> [npm](https://www.npmjs.com/package/langgraph-tenancy)

LangGraph's own [threat model](https://github.com/langchain-ai/langgraph/blob/main/.github/THREAT_MODEL.md) says it plainly:

> Checkpoint savers index by `thread_id`. Without application-level auth, any
> caller with a valid thread_id can access that thread's state. [...] Users
> embedding LangGraph directly must implement their own access controls.

If you run a multi-tenant product on open-source LangGraph, the only thing
between Customer A's agent state and Customer B's is a query filter in your
application code. This package replaces that convention with enforcement.

## Install

```bash
pip install langgraph-tenancy
```

## Usage

Wrap your existing checkpointer and store. Nothing else changes.

```python
from langgraph_tenancy import (
    TenantScopedCheckpointer,
    TenantScopedStore,
    InMemoryUsageLedger,
)

ledger = InMemoryUsageLedger()
checkpointer = TenantScopedCheckpointer(PostgresSaver(...), usage_ledger=ledger)
store = TenantScopedStore(InMemoryStore())

graph = builder.compile(checkpointer=checkpointer, store=store)

# tenant_id is now REQUIRED on every invocation
graph.invoke(
    {"messages": ["hello"]},
    config={"configurable": {"thread_id": "t1", "tenant_id": "acme"}},
)

# free per-tenant token metering, extracted from checkpointed messages
ledger.totals("acme")  # TenantUsage(input_tokens=..., output_tokens=..., by_model={...})
```

## What it enforces

| Raw LangGraph behavior | With `langgraph-tenancy` |
|---|---|
| Any caller with a `thread_id` reads that thread | Threads are physically keyed `tenant::thread`; wrong-thread_id bugs cannot cross tenants |
| Missing filter → silent unscoped query | Missing `tenant_id` → `TenantRequiredError`, nothing read or written |
| `checkpointer.list(None)` enumerates **every** tenant's threads | Refused with `UnscopedAccessError` |
| Store namespaces are convention; any node can read any namespace | Every op is rooted at the tenant segment, resolved from the run config automatically |
| `delete_thread("t1")` deletes whoever owns `t1` | Requires an explicit `for_tenant("acme").delete_thread("t1")` handle |
| `usage_metadata` buried in checkpoint blobs, unqueryable | Aggregated per tenant (and per model), deduped by message id |

## No magic

The entire mechanism is key prefixing plus mandatory-context checks, in two
small files you can audit in ten minutes:

- thread ids become `"{tenant_id}::{thread_id}"` before reaching your
  database; the prefix is stripped from everything returned.
- store namespaces `("memories",)` become `("{tenant_id}", "memories")`.
- tenant ids containing the separator are rejected, so `acme` can never craft
  a key that collides with another tenant's space.

It composes with any `BaseCheckpointSaver` / `BaseStore` implementation —
Postgres, SQLite, Redis, MongoDB, in-memory — because it never touches
storage itself.

## What it is not

- Not authentication. You decide which tenant a request belongs to; this
  package guarantees that decision is enforced everywhere downstream.
- Not encryption. Combine with `EncryptedSerializer` for at-rest encryption.
- Not a replacement for database-level controls in high-assurance setups
  (RLS, schema-per-tenant) — it's the layer that makes your *application*
  unable to leak, whatever the database allows.

## Tested

The adversarial test suite — every test attempts a cross-tenant access the
raw LangGraph API allows — runs against `InMemorySaver` **and** a real
`PostgresSaver` in CI. The isolation guarantees are proven on actual SQL
storage, not just the in-memory reference.

## Development

```bash
uv venv && uv pip install -e ".[test]"
uv run pytest                 # postgres tests skip if no server is reachable

# to run the postgres leg locally:
export LG_TENANCY_PG_URI=postgresql://user@localhost:5432/langgraph_tenancy_test
uv run pytest
```

## Status

Early (0.1.x). Covered today: sync + async checkpointer paths, sync store
paths, in-memory and Postgres backends. Not yet covered: subgraph
`checkpoint_ns` edge cases, `AsyncPostgresSaver`, `PostgresStore`, store TTL
ops. Issues and PRs welcome.

## License

[MIT](LICENSE)
