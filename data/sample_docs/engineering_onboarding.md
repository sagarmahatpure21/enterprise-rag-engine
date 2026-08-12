# Engineering Onboarding Guide

## Week 1: Environment Setup
New engineers should install the standard toolchain: Git, Docker, VS Code,
and the internal CLI (`corp-cli`). Access to GitHub, the internal package
registry, and the staging cluster is provisioned automatically on day one
via the identity management system.

## Code Review Standards
Every pull request requires at least one approval before merging. PRs
touching production infrastructure require two approvals, one of which
must come from a senior engineer or tech lead. CI checks (lint, unit
tests, security scan) must pass before merge is allowed.

## Deployment Process
Deployments follow a trunk-based development model. Merges to main trigger
an automatic deployment to staging. Production deployments require a
manual approval gate and are rolled out using a canary strategy: 5% of
traffic, then 25%, then 100%, with automated rollback if error rates
exceed 1%.

## On-call Rotation
Engineers join the on-call rotation after their first 90 days. Rotations
are weekly, and on-call engineers are expected to acknowledge pages within
15 minutes during business hours and 30 minutes outside business hours.

## Architecture Overview
The platform is built on a microservices architecture running on
Kubernetes, with services communicating over gRPC internally and REST at
the edge. Data pipelines use Kafka for streaming ingestion and Spark for
batch processing.
