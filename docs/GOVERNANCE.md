# Lifecycle, precedence, and authorization

## Lifecycle selection

The loader validates explicit effective ranges and supersession references across the complete
corpus. At request time, `select_policies` chooses documents effective on the supplied `as_of`
date. The default is the server's current date. Historical requests must supply a date explicitly;
natural-language dates are not parsed or trusted.

Within a standard policy family, the eligible document with the latest effective start is selected.
A signed customer-scoped document is eligible only when its customer identifier matches the
server-verified principal and its date range includes `as_of`. It then takes precedence over the
standard family document. More than one applicable customer document is ambiguous and causes a
no-answer decision. The system never constructs contract terms that are not present in an eligible
document.

The current precedence unit is a whole policy family, not an individual clause. Therefore a
customer override excludes the standard document even for questions about unmodified clauses.
That conservative limitation avoids mixing conflicting terms but can cause incomplete answers.

## Authentication boundary

`IdentityProvider` converts an opaque credential into a principal containing a subject, roles, and
optional customer scope. `LocalIdentityProvider` reads static development fixtures. API request
bodies forbid extra fields, including caller-supplied roles, and missing or invalid bearer tokens
return HTTP 401. A request with no accessible evidence in its selected department returns HTTP 403.

A production adapter must validate signed credentials with a trusted issuer and enforce audience,
expiry, revocation, tenant binding, and authoritative entitlements. None of those properties are
provided by the local adapter, so this project does not claim production-grade authentication or
RBAC.

## Ordering and leakage boundary

The request sequence is:

1. authenticate and derive roles/customer scope server-side;
2. select effective and precedence-eligible document IDs;
3. apply role, department, and eligible-document filters inside sparse/dense retrieval;
4. fuse and optionally rerank only filtered candidates;
5. assess evidence and optionally generate;
6. validate citations against that same authorized context;
7. emit a metadata-only audit event.

Audit events never contain query text, document text, chunk text, answers, or credentials. They are
useful for decision tracing but are not tamper-proof compliance logs.
