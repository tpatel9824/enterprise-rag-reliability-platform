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


class OIDCIdentityProvider:
    """Validates signed OIDC JWTs and derives authorization attributes from verified claims."""

    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        jwks_url: str,
        roles_claim: str = "roles",
        customer_claim: str = "customer_id",
        decoder=None,
    ):
        if not issuer or not audience or not jwks_url:
            raise ValueError("OIDC issuer, audience, and JWKS URL are required")
        self.issuer = issuer.rstrip("/")
        self.audience = audience
        self.roles_claim = roles_claim
        self.customer_claim = customer_claim
        if decoder is None:
            import jwt

            jwks_client = jwt.PyJWKClient(jwks_url)

            def decoder(token):
                signing_key = jwks_client.get_signing_key_from_jwt(token)
                return jwt.decode(
                    token,
                    signing_key.key,
                    algorithms=["RS256"],
                    audience=self.audience,
                    issuer=self.issuer,
                    options={"require": ["exp", "iat", "iss", "aud", "sub"]},
                )
        self._decoder = decoder

    def authenticate(self, credential: str | None) -> Principal:
        if not credential:
            raise AuthenticationError("Missing authentication credential")
        try:
            claims = self._decoder(credential)
        except Exception as exc:
            raise AuthenticationError("Invalid OIDC credential") from exc
        roles = claims.get(self.roles_claim, [])
        if not isinstance(roles, list) or not all(isinstance(role, str) and role for role in roles):
            raise AuthenticationError("Verified token has no valid role entitlement claim")
        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject:
            raise AuthenticationError("Verified token has no subject")
        customer = claims.get(self.customer_claim)
        if customer is not None and not isinstance(customer, str):
            raise AuthenticationError("Verified token has an invalid customer claim")
        return Principal(subject, frozenset(roles), customer)


def build_identity_provider(settings):
    mode = settings.auth_mode.lower()
    if settings.app_environment == "production" and mode == "local":
        raise RuntimeError("Development token authentication is disabled in production")
    if mode == "local":
        return LocalIdentityProvider.from_json(settings.local_identities_path)
    if mode == "oidc":
        return OIDCIdentityProvider(
            issuer=settings.oidc_issuer,
            audience=settings.oidc_audience,
            jwks_url=settings.oidc_jwks_url,
            roles_claim=settings.oidc_roles_claim,
            customer_claim=settings.oidc_customer_claim,
        )
    raise ValueError(f"Unsupported AUTH_MODE: {settings.auth_mode!r}")
