"""Checkpointer backends for the isolation suite.

Every checkpointer test runs against each backend, so the isolation
guarantees are proven on real storage, not just the in-memory reference.
Postgres tests skip automatically if no server is reachable.
"""

import os

import pytest

PG_URI = os.environ.get(
    "LG_TENANCY_PG_URI",
    "postgresql://postgres:postgres@localhost:5432/langgraph_tenancy_test",
)
# In CI we want a missing database to FAIL the postgres leg, not skip it —
# otherwise a service misconfiguration silently halves the suite.
PG_REQUIRED = os.environ.get("LG_TENANCY_PG_REQUIRED") == "1"


@pytest.fixture(params=["memory", "postgres"])
def make_inner(request):
    """Factory producing fresh inner savers on a clean backend."""
    if request.param == "memory":
        from langgraph.checkpoint.memory import InMemorySaver

        yield InMemorySaver
        return

    psycopg = pytest.importorskip("psycopg")
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row

    try:
        conn = psycopg.Connection.connect(
            PG_URI, autocommit=True, prepare_threshold=0, row_factory=dict_row
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        if PG_REQUIRED:
            raise
        pytest.skip(f"postgres unavailable: {exc}")

    PostgresSaver(conn).setup()
    # clean slate per test (keep the migrations table)
    tables = conn.execute(
        "select tablename from pg_tables where schemaname='public' "
        "and tablename like 'checkpoint%' and tablename not like '%migrations'"
    ).fetchall()
    for row in tables:
        conn.execute(f'TRUNCATE "{row["tablename"]}"')

    yield lambda: PostgresSaver(conn)
    conn.close()
