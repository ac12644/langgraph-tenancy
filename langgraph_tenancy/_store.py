"""Tenant-scoped wrapper around any `BaseStore`.

Raw `BaseStore` namespaces are pure convention — any caller can `get()`,
`search()`, or `list_namespaces()` across all of them. This wrapper prepends
the tenant id as the root namespace segment on every operation and strips it
from every result, so two tenants using the identical namespace tuple
(e.g. ``("memories",)``) land in physically distinct locations.

Tenant resolution, in order:
1. A pinned tenant from `.for_tenant(tenant_id)` (out-of-band/admin access).
2. The ambient run config via `langgraph.config.get_config()` — inside a node,
   the `tenant_id` you passed to `graph.invoke(...)` is picked up automatically.
3. Otherwise: `TenantRequiredError`. Never an unscoped operation.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from langgraph.store.base import (
    BaseStore,
    GetOp,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchOp,
)

from langgraph_tenancy._errors import InvalidTenantError, TenantRequiredError

SEP = "::"


class TenantScopedStore(BaseStore):
    def __init__(self, inner: BaseStore, *, _tenant: str | None = None) -> None:
        self.inner = inner
        self._tenant = _tenant

    def for_tenant(self, tenant_id: str) -> TenantScopedStore:
        """A view of the store pinned to one tenant, for use outside a run."""
        self._validate(tenant_id)
        return TenantScopedStore(self.inner, _tenant=tenant_id)

    # -- tenant resolution ----------------------------------------------------

    @staticmethod
    def _validate(tenant: Any) -> str:
        if not isinstance(tenant, str) or not tenant:
            raise TenantRequiredError("store access")
        if SEP in tenant:
            raise InvalidTenantError(f"tenant_id may not contain '{SEP}': {tenant!r}")
        return tenant

    def _current_tenant(self) -> str:
        if self._tenant is not None:
            return self._tenant
        try:
            from langgraph.config import get_config

            config = get_config()
        except Exception:
            config = None
        tenant = ((config or {}).get("configurable") or {}).get("tenant_id")
        return self._validate(tenant)

    # -- op rewriting -----------------------------------------------------------

    def _scope_op(self, tenant: str, op: Op) -> Op:
        if isinstance(op, (GetOp, PutOp)):
            return op._replace(namespace=(tenant, *op.namespace))
        if isinstance(op, SearchOp):
            return op._replace(namespace_prefix=(tenant, *op.namespace_prefix))
        if isinstance(op, ListNamespacesOp):
            conditions = list(op.match_conditions or ())
            for i, cond in enumerate(conditions):
                if cond.match_type == "prefix":
                    conditions[i] = MatchCondition(
                        match_type="prefix", path=(tenant, *cond.path)
                    )
                    break
            else:
                conditions.insert(
                    0, MatchCondition(match_type="prefix", path=(tenant,))
                )
            return op._replace(
                match_conditions=tuple(conditions),
                max_depth=None if op.max_depth is None else op.max_depth + 1,
            )
        raise TypeError(f"unsupported op: {op!r}")

    def _unscope_result(self, tenant: str, op: Op, result: Result) -> Result:
        if isinstance(op, GetOp) and result is not None:
            result.namespace = tuple(result.namespace[1:])
            return result
        if isinstance(op, SearchOp):
            for item in result:
                item.namespace = tuple(item.namespace[1:])
            return result
        if isinstance(op, ListNamespacesOp):
            return [tuple(ns[1:]) for ns in result if ns and ns[0] == tenant]
        return result

    # -- BaseStore protocol -------------------------------------------------------

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        tenant = self._current_tenant()
        ops = list(ops)
        results = self.inner.batch([self._scope_op(tenant, op) for op in ops])
        return [
            self._unscope_result(tenant, op, result)
            for op, result in zip(ops, results, strict=True)
        ]

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        tenant = self._current_tenant()
        ops = list(ops)
        results = await self.inner.abatch([self._scope_op(tenant, op) for op in ops])
        return [
            self._unscope_result(tenant, op, result)
            for op, result in zip(ops, results, strict=True)
        ]
