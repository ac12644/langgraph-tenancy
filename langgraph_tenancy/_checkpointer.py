"""Tenant-scoped wrapper around any `BaseCheckpointSaver`.

Design:
- Reads `tenant_id` from `config["configurable"]` on every call. Missing
  tenant -> `TenantRequiredError`. There is no unscoped fallback.
- Physically prefixes `thread_id` with the tenant (``acme::thread-1``) before
  it reaches the inner saver, and strips the prefix from everything returned.
  A wrong-thread_id bug in app code therefore cannot cross a tenant boundary:
  the key the database sees is always tenant-qualified.
- Blocks the dangerous raw-API escape hatches: `list(None)` and
  `delete_thread(...)` without a tenant.
- Optionally records per-tenant token usage from checkpointed messages into a
  `UsageLedger` (see `_usage.py`) — same integration point, free metering.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)

from langgraph_tenancy._errors import (
    InvalidTenantError,
    TenantRequiredError,
    UnscopedAccessError,
)
from langgraph_tenancy._usage import UsageLedger, extract_usage

SEP = "::"


def _validate_tenant(tenant: Any, where: str) -> str:
    if not isinstance(tenant, str) or not tenant:
        raise TenantRequiredError(where)
    if SEP in tenant:
        raise InvalidTenantError(f"tenant_id may not contain '{SEP}': {tenant!r}")
    return tenant


class TenantScopedCheckpointer(BaseCheckpointSaver):
    """Wrap a checkpointer so every operation is scoped to one tenant."""

    def __init__(
        self,
        inner: BaseCheckpointSaver,
        *,
        usage_ledger: UsageLedger | None = None,
    ) -> None:
        super().__init__(serde=inner.serde)
        self.inner = inner
        self.usage_ledger = usage_ledger

    # -- scoping helpers ----------------------------------------------------

    def _scope(self, config: RunnableConfig, where: str) -> tuple[str, RunnableConfig]:
        conf = config.get("configurable") or {}
        tenant = _validate_tenant(conf.get("tenant_id"), where)
        thread_id = conf.get("thread_id")
        scoped_thread = f"{tenant}{SEP}{thread_id}"
        return tenant, {
            **config,
            "configurable": {**conf, "thread_id": scoped_thread},
        }

    def _unscope_config(
        self, tenant: str, config: RunnableConfig | None
    ) -> RunnableConfig | None:
        if config is None:
            return None
        conf = config.get("configurable") or {}
        thread_id = conf.get("thread_id")
        prefix = f"{tenant}{SEP}"
        if isinstance(thread_id, str) and thread_id.startswith(prefix):
            conf = {**conf, "thread_id": thread_id[len(prefix) :], "tenant_id": tenant}
        return {**config, "configurable": conf}

    def _unscope_tuple(
        self, tenant: str, tup: CheckpointTuple | None
    ) -> CheckpointTuple | None:
        if tup is None:
            return None
        return CheckpointTuple(
            config=self._unscope_config(tenant, tup.config),
            checkpoint=tup.checkpoint,
            metadata=tup.metadata,
            parent_config=self._unscope_config(tenant, tup.parent_config),
            pending_writes=tup.pending_writes,
        )

    # -- core protocol (sync) -----------------------------------------------

    @property
    def config_specs(self) -> list:
        return self.inner.config_specs

    def get_next_version(self, current: Any, channel: None = None) -> Any:
        return self.inner.get_next_version(current, channel)

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        tenant, scoped = self._scope(config, "get_tuple()")
        return self._unscope_tuple(tenant, self.inner.get_tuple(scoped))

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        if config is None:
            raise UnscopedAccessError(
                "list(None) would enumerate every tenant's threads; "
                "pass a config with tenant_id (and optionally thread_id)."
            )
        tenant, scoped = self._scope(config, "list()")
        scoped_before = self._scope(before, "list(before=...)")[1] if before else None
        for tup in self.inner.list(
            scoped, filter=filter, before=scoped_before, limit=limit
        ):
            yield self._unscope_tuple(tenant, tup)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        tenant, scoped = self._scope(config, "put()")
        if self.usage_ledger is not None:
            for record in extract_usage(checkpoint):
                self.usage_ledger.record(tenant, record)
        result = self.inner.put(scoped, checkpoint, metadata, new_versions)
        return self._unscope_config(tenant, result)

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        _, scoped = self._scope(config, "put_writes()")
        self.inner.put_writes(scoped, writes, task_id, task_path)

    # -- blocked / redirected escape hatches ----------------------------------

    def delete_thread(self, thread_id: str) -> None:
        raise UnscopedAccessError(
            "delete_thread() has no tenant context; "
            "use for_tenant(tenant_id).delete_thread(thread_id)."
        )

    def delete_for_runs(self, run_ids: Sequence[str]) -> None:
        raise UnscopedAccessError(
            "delete_for_runs() cannot be tenant-scoped (run ids are global); "
            "call it on the inner saver explicitly if you accept that."
        )

    def for_tenant(self, tenant_id: str) -> TenantCheckpointerHandle:
        """Admin/maintenance handle pinned to one tenant."""
        return TenantCheckpointerHandle(
            self, _validate_tenant(tenant_id, "for_tenant()")
        )

    # -- core protocol (async) ------------------------------------------------

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        tenant, scoped = self._scope(config, "aget_tuple()")
        return self._unscope_tuple(tenant, await self.inner.aget_tuple(scoped))

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        if config is None:
            raise UnscopedAccessError(
                "alist(None) would enumerate every tenant's threads; "
                "pass a config with tenant_id (and optionally thread_id)."
            )
        tenant, scoped = self._scope(config, "alist()")
        scoped_before = self._scope(before, "alist(before=...)")[1] if before else None
        async for tup in self.inner.alist(
            scoped, filter=filter, before=scoped_before, limit=limit
        ):
            yield self._unscope_tuple(tenant, tup)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        tenant, scoped = self._scope(config, "aput()")
        if self.usage_ledger is not None:
            for record in extract_usage(checkpoint):
                self.usage_ledger.record(tenant, record)
        result = await self.inner.aput(scoped, checkpoint, metadata, new_versions)
        return self._unscope_config(tenant, result)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        _, scoped = self._scope(config, "aput_writes()")
        await self.inner.aput_writes(scoped, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        self.delete_thread(thread_id)


class TenantCheckpointerHandle:
    """Maintenance operations pre-bound to a single tenant.

    Exists because `BaseCheckpointSaver.delete_thread/copy_thread/prune` take
    bare thread ids with no config, so there is no per-call tenant to read.
    """

    def __init__(self, parent: TenantScopedCheckpointer, tenant: str) -> None:
        self._inner = parent.inner
        self._tenant = tenant

    def _scoped(self, thread_id: str) -> str:
        return f"{self._tenant}{SEP}{thread_id}"

    def delete_thread(self, thread_id: str) -> None:
        self._inner.delete_thread(self._scoped(thread_id))

    def copy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        self._inner.copy_thread(
            self._scoped(source_thread_id), self._scoped(target_thread_id)
        )

    def prune(
        self, thread_ids: Sequence[str], *, strategy: str = "keep_latest"
    ) -> None:
        self._inner.prune([self._scoped(t) for t in thread_ids], strategy=strategy)
