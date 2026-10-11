---
document_id: sec_incident_standard_2026_v2
title: Security Incident Standard
department: Security
version: "3.2"
effective_date: "2026-01-10"
effective_from: "2026-01-10"
effective_until: null
supersedes: []
status: active
policy_family: security_incident
access_roles: [security, engineering, support, compliance, manager, executive]
tags: [incident, credentials, evidence]
---
# Security Incident Standard

## Reporting

Suspected credential compromise, unauthorized access, malware, or loss of a managed device must be reported immediately through the security incident channel. The reporter preserves relevant identifiers and times but should not conduct an independent forensic investigation.

## Containment

Security coordinates credential rotation, session revocation, device isolation, and other containment actions. Engineering assists with service-level controls under the Engineering Incident Response Guide. Actions should minimize evidence destruction and unnecessary customer impact.

## Evidence and Communication

Responders preserve relevant logs, alerts, and change records with documented custody. Support communicates only facts approved by the incident commander or Security lead. Potential legal or privacy obligations are escalated to Compliance.

## Closure

Security records scope, affected systems, containment, recovery, and follow-up actions. High-severity incidents require a formal review. Lessons that affect release controls are tracked through the Production Release Standard.
