# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | Yes                |

## Reporting a Security Issue

If you discover a security issue in this project, please report it responsibly.

**Please do NOT open a public GitHub issue for security problems.**

Instead, email the maintainer directly at: **138163850+CameronScarpati@users.noreply.github.com**

Include the following in your report:

- Description of the issue
- Steps to reproduce (if applicable)
- Potential impact
- Suggested fix (if you have one)

## Response Timeline

- **Acknowledgment**: Within 48 hours of your report
- **Assessment**: Within 1 week
- **Fix or mitigation**: Depends on severity, but we aim for prompt resolution

## Scope

This policy covers the `lob-regime-scanner` Python package and its dependencies as configured in `pyproject.toml`. It does not cover third-party data sources (e.g., Tardis.dev) or exchanges.

## Best Practices for Users

- Never commit API keys or credentials to the repository
- Use environment variables (e.g., `TARDIS_API_KEY`) for sensitive configuration
- Keep dependencies updated to their latest compatible versions
