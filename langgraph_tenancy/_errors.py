"""Errors raised by langgraph-tenancy.

Every error here exists to turn a silent cross-tenant leak into a loud failure.
"""


class TenancyError(Exception):
    """Base class for all tenancy errors."""


class TenantRequiredError(TenancyError):
    """Raised when an operation runs without a tenant_id in scope.

    The wrapper never falls back to an unscoped read or write: no tenant, no data.
    """

    def __init__(self, where: str) -> None:
        super().__init__(
            f"{where} requires a tenant. Pass it in the run config: "
            "config={'configurable': {'thread_id': ..., 'tenant_id': ...}} "
            "or use .for_tenant(tenant_id) for out-of-band access."
        )


class UnscopedAccessError(TenancyError):
    """Raised for operations that would touch data across tenant boundaries.

    Example: `checkpointer.list(None)` on a raw saver enumerates every
    customer's threads. This wrapper refuses that call instead.
    """


class InvalidTenantError(TenancyError):
    """Raised when a tenant_id could be used to escape its scope.

    The tenant id becomes part of storage keys, so it must not contain the
    separator (or be empty) — otherwise tenant "a" could craft ids that
    collide with tenant "a::b".
    """
