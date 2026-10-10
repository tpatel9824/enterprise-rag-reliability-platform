# API Authentication Guide

- **Document Id:** eng_api_auth_2026
- **Title:** API Authentication Guide
- **Department:** Engineering
- **Version:** 3.0
- **Effective Date:** 2026-01-01
- **Access Roles:** ['employee', 'engineering', 'support', 'security', 'product', 'manager', 'executive']
- **Tags:** ['api', 'authentication']

## Policy / Guidance

API keys must be stored in secret-management tooling and never committed to source control. Service-to-service calls should use short-lived credentials when supported. Suspected compromise requires immediate rotation.
