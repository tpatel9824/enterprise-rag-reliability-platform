---
document_id: sec_access_control_2026_v2
title: Access Control Standard
department: Security
version: "2.5"
effective_date: "2026-01-01"
effective_from: "2026-01-01"
effective_until: null
supersedes: []
status: active
policy_family: security_access_control
access_roles: [security, engineering, it, manager, executive]
tags: [access, least-privilege, iam]
---
# Access Control Standard

## Provisioning

Production access requires documented business justification, manager approval, and system-owner approval. Permissions must follow least privilege and use a named identity. Shared administrator accounts are prohibited except for controlled emergency accounts.

## Authentication

Privileged access requires multi-factor authentication and approved credential storage. Service identities should use short-lived credentials when the platform supports them. Secrets must not be committed to source control or placed in application logs.

## Review and Removal

System owners review privileged access quarterly. Managers report role changes promptly, and access no longer required must be removed within one business day. Human Resources termination events trigger the established offboarding workflow.

## Emergency Access

Emergency access is time-limited, logged, and reviewed after use. The requester records the incident or change reference and explains why normal access was insufficient. Emergency access does not remove the requirement to preserve audit evidence.
