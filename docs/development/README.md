# Development Documentation

This directory contains technical documentation for developers working on Stockula.

## Contents

### [CI/CD](ci-cd.md)

Complete guide to the continuous integration and deployment setup, including GitHub Actions workflows, release automation, and Docker builds.

### [AutoTS Threading Considerations](autots-threading-considerations.md)

Guidelines for working with AutoTS's threading limitations and recommendations for implementing reliable forecasting.

### [SQLModel Migration](sqlmodel-migration.md)

Documentation about the migration from SQLAlchemy to SQLModel for improved type safety and validation.

### [Testing](testing.md)

Comprehensive guide for testing in Stockula, including testing strategy, best practices, coverage improvements, and common patterns.

### [Test Coverage Status](test-coverage-status.md)

Detailed test coverage metrics, recent improvements, and important warnings about untested modules.

### [YAML Formatting](yaml-formatting.md)

Configuration and usage guide for YAML formatting with yamlfmt and yamllint validation.

### [Docker Validation](docker-validation.md)

Comprehensive Docker setup validation script for ensuring proper Docker configuration and functionality.

### [Migration Guide: AutoTS Validator](migration-autots-validator.md)

Guide for migrating from the old AutoTSModelValidator class to the new database-driven validation system.

## Quick Links

- [Project README](../../README.md)
- [User Guide](../user-guide/)
- [API Reference](../api/)

## Contributing

When adding new development documentation:

1. Create a new Markdown file in this directory
1. Use clear, descriptive filenames
1. Add a link and description in this README
1. Follow the existing documentation style
1. Include code examples where appropriate
