# Data Pipeline Reliability Guide

- **Document Id:** eng_data_pipeline_2026
- **Title:** Data Pipeline Reliability Guide
- **Department:** Engineering
- **Version:** 1.0
- **Effective Date:** 2026-03-01
- **Access Roles:** ['engineering', 'security', 'product', 'manager', 'executive']
- **Tags:** ['data', 'pipeline']

## Policy / Guidance

Scheduled pipelines should expose freshness, row-count, and failure metrics. Retries should use bounded backoff. Pipelines processing customer data must respect retention and access controls.
