"""Adversarial isolation tests.

Every test here attempts a cross-tenant access the raw LangGraph API allows,
and asserts the wrapper makes it either impossible or an explicit error.
Run against the real InMemorySaver/InMemoryStore from langgraph.
"""

import operator
from typing import Annotated, TypedDict
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_store
from langgraph.graph import StateGraph
from langgraph.store.memory import InMemoryStore

from langgraph_tenancy import (
    InMemoryUsageLedger,
    InvalidTenantError,
    TenantRequiredError,
    TenantScopedCheckpointer,
    TenantScopedStore,
    UnscopedAccessError,
)


class State(TypedDict):
    messages: Annotated[list, operator.add]


def make_graph(checkpointer, store=None, model_name="claude-sonnet-4-6"):
    """One-node graph that fakes an LLM reply (with usage) and writes a memory."""

    def agent(state: State) -> State:
        # real chat models always set an id; the ledger dedupes on it
        reply = AIMessage(
            id=str(uuid4()),
            content=f"echo: {state['messages'][-1]}",
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
            response_metadata={"model_name": model_name},
        )
        if store is not None:
            get_store().put(
                ("memories",),
                f"note-{len(state['messages'])}",
                {"last": str(state["messages"][-1])},
            )
        return {"messages": [reply]}

    builder = StateGraph(State)
    builder.add_node("agent", agent)
    builder.set_entry_point("agent")
    builder.set_finish_point("agent")
    return builder.compile(checkpointer=checkpointer, store=store)


def cfg(tenant, thread="t1"):
    return {"configurable": {"thread_id": thread, "tenant_id": tenant}}


# --- checkpointer isolation ---------------------------------------------------


def test_same_thread_id_different_tenants_do_not_collide(make_inner):
    graph = make_graph(TenantScopedCheckpointer(make_inner()))
    graph.invoke({"messages": ["from acme"]}, cfg("acme"))
    graph.invoke({"messages": ["from globex"]}, cfg("globex"))

    acme = graph.get_state(cfg("acme")).values["messages"]
    globex = graph.get_state(cfg("globex")).values["messages"]
    assert acme[0] == "from acme" and len(acme) == 2
    assert globex[0] == "from globex" and len(globex) == 2
    assert not any("globex" in str(m) for m in acme)


def test_missing_tenant_id_raises_instead_of_leaking(make_inner):
    graph = make_graph(TenantScopedCheckpointer(make_inner()))
    with pytest.raises(TenantRequiredError):
        graph.invoke({"messages": ["hi"]}, {"configurable": {"thread_id": "t1"}})


def test_tenant_id_cannot_contain_separator(make_inner):
    graph = make_graph(TenantScopedCheckpointer(make_inner()))
    with pytest.raises(InvalidTenantError):
        graph.invoke({"messages": ["hi"]}, cfg("acme::evil"))


def test_list_none_is_refused(make_inner):
    saver = TenantScopedCheckpointer(make_inner())
    make_graph(saver).invoke({"messages": ["hi"]}, cfg("acme"))
    with pytest.raises(UnscopedAccessError):
        list(saver.list(None))


def test_list_only_sees_own_tenant(make_inner):
    saver = TenantScopedCheckpointer(make_inner())
    graph = make_graph(saver)
    graph.invoke({"messages": ["a"]}, cfg("acme"))
    graph.invoke({"messages": ["b"]}, cfg("globex"))

    seen = [t.config["configurable"]["thread_id"] for t in saver.list(cfg("acme"))]
    assert seen and all(t == "t1" for t in seen)
    values = [t for t in saver.list(cfg("acme"))]
    assert not any("globex" in str(t.checkpoint.get("channel_values")) for t in values)


def test_delete_thread_requires_tenant_handle(make_inner):
    saver = TenantScopedCheckpointer(make_inner())
    graph = make_graph(saver)
    graph.invoke({"messages": ["a"]}, cfg("acme"))
    graph.invoke({"messages": ["b"]}, cfg("globex"))

    with pytest.raises(UnscopedAccessError):
        saver.delete_thread("t1")

    saver.for_tenant("acme").delete_thread("t1")
    assert saver.get_tuple(cfg("acme")) is None
    assert saver.get_tuple(cfg("globex")) is not None  # other tenant untouched


def test_storage_keys_are_physically_tenant_prefixed(make_inner):
    inner = make_inner()
    make_graph(TenantScopedCheckpointer(inner)).invoke({"messages": ["a"]}, cfg("acme"))
    raw_threads = {t.config["configurable"]["thread_id"] for t in inner.list(None)}
    assert raw_threads == {"acme::t1"}


# --- store isolation ----------------------------------------------------------


def test_store_namespaces_are_isolated_per_tenant():
    store = TenantScopedStore(InMemoryStore())
    graph = make_graph(TenantScopedCheckpointer(InMemorySaver()), store=store)
    graph.invoke({"messages": ["secret-acme"]}, cfg("acme"))
    graph.invoke({"messages": ["secret-globex"]}, cfg("globex"))

    acme_items = store.for_tenant("acme").search(("memories",))
    globex_items = store.for_tenant("globex").search(("memories",))
    assert [i.value["last"] for i in acme_items] == ["secret-acme"]
    assert [i.value["last"] for i in globex_items] == ["secret-globex"]
    # returned namespaces are unprefixed — the tenant segment never leaks out
    assert acme_items[0].namespace == ("memories",)


def test_store_outside_run_without_tenant_raises():
    store = TenantScopedStore(InMemoryStore())
    with pytest.raises(TenantRequiredError):
        store.search(("memories",))


def test_list_namespaces_only_sees_own_tenant():
    store = TenantScopedStore(InMemoryStore())
    store.for_tenant("acme").put(("memories", "work"), "k", {"v": 1})
    store.for_tenant("globex").put(("memories", "home"), "k", {"v": 2})
    assert store.for_tenant("acme").list_namespaces() == [("memories", "work")]


# --- usage metering -------------------------------------------------------------


def test_usage_attributed_per_tenant_and_deduped(make_inner):
    ledger = InMemoryUsageLedger()
    graph = make_graph(TenantScopedCheckpointer(make_inner(), usage_ledger=ledger))

    graph.invoke({"messages": ["q1"]}, cfg("acme"))
    graph.invoke({"messages": ["q2"]}, cfg("acme"))  # 2nd turn, same thread
    graph.invoke({"messages": ["q1"]}, cfg("globex"))

    acme, globex = ledger.totals("acme"), ledger.totals("globex")
    # two LLM calls for acme, one for globex; old messages re-checkpointed
    # on turn 2 must not double-count
    assert acme.messages == 2 and acme.total_tokens == 30
    assert globex.messages == 1 and globex.total_tokens == 15
    assert acme.by_model == {"claude-sonnet-4-6": 30}
