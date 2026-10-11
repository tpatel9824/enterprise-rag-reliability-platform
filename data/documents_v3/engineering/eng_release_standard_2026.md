---
document_id: eng_release_standard_2026_v2
title: Production Release Standard
department: Engineering
version: "2.0"
effective_date: "2026-01-15"
effective_from: "2026-01-15"
effective_until: null
supersedes: []
status: active
policy_family: engineering_release
access_roles: [engineering, product, security, manager, executive]
tags: [release, deployment, testing]
---
# Production Release Standard

## Required Evidence

A production release requires reviewed code, passing automated tests, an immutable artifact identifier, and documented rollback steps. The release record links those items to the approved change. Known failed tests require an explicit risk acceptance from the responsible engineering manager.

## Risk Review

Changes affecting authentication, authorization, encryption, or regulated data require Security review before production. High-risk changes use a phased rollout when the platform supports it. The owner defines health signals and a decision point for continuing or stopping the rollout.

## Deployment and Verification

Deployments run through the approved continuous-delivery pipeline. The operator verifies service health, error rate, latency, and critical business checks after each stage. If safety thresholds fail, the operator pauses or rolls back instead of proceeding to the next stage.

## Emergency Changes

An incident commander may authorize an emergency production change when delay creates greater customer risk. The team still records the artifact, approver, validation evidence, and rollback result. A retrospective review must occur within two business days.
