# API Limits

- **Document Id:** product_api_limits_2026
- **Title:** API Limits
- **Department:** Product
- **Version:** 2.0
- **Effective Date:** 2026-01-01
- **Access Roles:** ['employee', 'support', 'engineering', 'product', 'sales', 'manager', 'executive']
- **Tags:** ['api', 'limits']

## Policy / Guidance

The standard API tier permits 60 requests per minute per API key. Enterprise customers may have negotiated higher limits. Clients receiving HTTP 429 should use exponential backoff with jitter.
