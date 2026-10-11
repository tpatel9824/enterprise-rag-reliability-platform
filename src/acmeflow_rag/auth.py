"""Authentication boundary and development-only local identity provider."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class AuthenticationError(PermissionError):
    pass


@dataclass(frozen=True)
class Principal:
    subject: str
    roles: frozenset[str]
    customer_id: str | None = None


class IdentityProvider(Protocol):
    def authenticate(self, credential: str | None) -> Principal: ...


class LocalIdentityProvider:
    """Static token mapping for local development and tests; never a production IdP."""

    def __init__(self, identities: dict[str, Principal]):
        self._identities = dict(identities)

    @classmethod
    def from_json(cls, path: Path) -> "LocalIdentityProvider":
        payload = json.loads(path.read_text(encoding="utf-8"))
        identities: dict[str, Principal] = {}
        for token, value in payload.get("tokens", {}).items():
            if not token or not isinstance(value.get("roles"), list):
                raise ValueError("Local identity file contains an invalid token entry")
            identities[token] = Principal(
                subject=value["subject"],
                roles=frozenset(value["roles"]),
                customer_id=value.get("customer_id"),
            )
        return cls(identities)

    def authenticate(self, credential: str | None) -> Principal:
        if not credential:
            raise AuthenticationError("Missing authentication credential")
        principal = self._identities.get(credential)
        if principal is None:
            raise AuthenticationError("Invalid authentication credential")
        return principal
